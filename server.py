#!/usr/bin/env python3
"""
3D Reconstruction Pipeline Server — FastAPI implementation
Run alongside Flask for parallel testing:
    uvicorn server_fastapi:app --host 0.0.0.0 --port 8001 --reload

Production start command (after Flask removal):
    uvicorn server:app --host 0.0.0.0 --port $PORT
"""
import asyncio
import glob
import os
import time
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import boto3
from botocore.config import Config

load_dotenv()

# ── Directory setup ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRAMES_DIR = os.path.join(BASE_DIR, "test_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

for _d in (UPLOAD_DIR, FRAMES_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Global pipeline state ────────────────────────────────────────────────────
state: dict = {
    "stage":             "idle",   # idle | extracting | extracted | reconstructing | done | error
    "message":           "Ready.",
    "frame_count":       0,
    "error":             None,
    "carbon_estimation": None,
    "overlap_warning":   None,
    "cancel_requested":  False,
}

def upd(stage: str, msg: str, **kw: Any) -> None:
    state.update({"stage": stage, "message": msg, **kw})

# ── Pydantic request / response models ───────────────────────────────────────

class ReconstructRequest(BaseModel):
    """Optional JSON body for POST /reconstruct."""
    tree_code: Optional[str] = None
    remove_background: Optional[bool] = True

class StatusResponse(BaseModel):
    stage: str
    message: str
    frame_count: int
    error: Optional[str]
    carbon_estimation: Optional[Any]
    overlap_warning: Optional[str]
    frames: List[str]
    has_result: bool

class HistoryResponse(BaseModel):
    success: bool
    tree_code: str
    history: List[Any]

class ScansResponse(BaseModel):
    success: bool
    scans: List[Any]

# ── Helper: load scale_factor from calibration.json (scan-id-aware) ─────────
import json as _json
import logging as _logging

_calib_logger = _logging.getLogger("calibration")

def _load_scale_factor_for_scan(scan_id: str = None) -> float:
    """
    Looks up scale_factor from calibration.json.
    Priority: scan_id entry → 'default' entry → hardcoded 1.0 (with WARNING).
    """
    calib_path = os.path.join(BASE_DIR, "calibration.json")
    if os.path.exists(calib_path):
        try:
            with open(calib_path, "r") as fh:
                registry = _json.load(fh)
            # 1. Try scan-specific entry first
            if scan_id and scan_id in registry:
                sf = float(registry[scan_id]["scale_factor"])
                _calib_logger.info(
                    f"[CALIBRATION] Loaded scan-specific scale_factor={sf:.8f} "
                    f"for scan_id='{scan_id}' from {calib_path}"
                )
                return sf
            # 2. Fall back to global 'default' entry if present
            if "default" in registry:
                sf = float(registry["default"]["scale_factor"])
                _calib_logger.info(
                    f"[CALIBRATION] No scan-specific calibration for '{scan_id}'. "
                    f"Using global 'default' scale_factor={sf:.8f} from {calib_path}"
                )
                return sf
            _calib_logger.warning(
                f"[CALIBRATION] calibration.json exists at {calib_path} but contains "
                f"no entry for scan_id='{scan_id}' and no 'default' key. "
                f"Falling back to scale_factor=1.0 — DBH/height values will be UNCALIBRATED."
            )
        except Exception as e:
            _calib_logger.warning(
                f"[CALIBRATION] Failed to read {calib_path}: {e}. "
                f"Falling back to scale_factor=1.0 — DBH/height values will be UNCALIBRATED."
            )
    else:
        _calib_logger.warning(
            "[CALIBRATION] ⚠ WARNING: calibration.json NOT FOUND. "
            "Using scale_factor=1.0 (uncalibrated default). "
            "DBH and height measurements will be in arbitrary PLY units, NOT real-world meters. "
            "Run calibrate_scale.py to create a calibration file."
        )
    return 1.0


# ── Helper: carbon analysis (sync, CPU-bound — always called inside thread) ──
def run_carbon_analysis(ply_path: str, scan_id: str = None) -> dict:
    try:
        from carbon.allometric import estimate_carbon
        from carbon.dbh_extractor import extract_dbh

        # Load scale_factor from calibration.json (scan-id-aware, with fallback + warnings)
        scale_factor = _load_scale_factor_for_scan(scan_id)

        dbh_result = extract_dbh(
            ply_path=ply_path, scale_factor=scale_factor,
            vertical_axis="z", breast_height=1.3,
        )
        if "error" in dbh_result:
            return {"error": dbh_result["error"]}

        carbon_result = estimate_carbon(
            dbh_cm=dbh_result["dbh_cm"],
            height_m=dbh_result["height_m"],
            wood_density=0.6,
        )
        return {
            "dbh_cm":                  dbh_result["dbh_cm"],
            "height_m":                dbh_result["height_m"],
            "confidence":              dbh_result["confidence_note"],
            "method":                  dbh_result["method"],
            "slice_points_count":      dbh_result["slice_points_count"],
            "mean_fit_error_cm":       dbh_result["mean_fit_error_cm"],
            "scale_factor_used":       scale_factor,
            "calibrated":              scale_factor != 1.0,
            "biomass_kg":              carbon_result["total_biomass_kg"],
            "above_ground_biomass_kg": carbon_result["above_ground_biomass_kg"],
            "below_ground_biomass_kg": carbon_result["below_ground_biomass_kg"],
            "carbon_kg":               carbon_result["carbon_kg"],
            "co2e_kg":                 carbon_result["co2e_kg"],
            "disclaimer":              carbon_result["disclaimer"],
        }
    except Exception as exc:
        return {"error": f"Failed to compute carbon metrics: {exc}"}

# ── Helper: check overlap between selected frames and dynamically resample ───
def _check_overlap_and_resample(candidates: list, initial_idxs: list, threshold: float = 0.15) -> list:
    """
    Checks overlap between consecutive frames in initial_idxs.
    If overlap is less than threshold (e.g. 15%), dynamically inserts the middle
    candidate frame from the candidates list to heal the coverage gap.
    """
    import cv2
    import numpy as np

    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    current_idxs = list(initial_idxs)
    added_count = 0
    max_added = 10  # Cap the number of additional frames
    i = 0
    gaps_detected = []

    print(f"[OVERLAP] Starting overlap validation for {len(current_idxs)} frames...")

    while i < len(current_idxs) - 1 and added_count < max_added:
        idx_a = current_idxs[i]
        idx_b = current_idxs[i+1]

        # Cannot insert if they are already adjacent candidates
        if idx_b - idx_a <= 1:
            i += 1
            continue

        # Get frame matrices (handling both numpy array and encoded jpeg bytes)
        raw_a = candidates[idx_a][2]
        raw_b = candidates[idx_b][2]

        if isinstance(raw_a, np.ndarray):
            frame_a = raw_a
        else:
            frame_a = cv2.imdecode(raw_a, cv2.IMREAD_COLOR)

        if isinstance(raw_b, np.ndarray):
            frame_b = raw_b
        else:
            frame_b = cv2.imdecode(raw_b, cv2.IMREAD_COLOR)

        # Convert to grayscale safely
        gray_a = frame_a if (len(frame_a.shape) == 2 or frame_a.shape[2] == 1) else cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = frame_b if (len(frame_b.shape) == 2 or frame_b.shape[2] == 1) else cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        # Downscale for performance during overlap matching
        h_a, w_a = gray_a.shape
        if w_a > 640:
            gray_a = cv2.resize(gray_a, (640, int(h_a * 640 / w_a)))
        h_b, w_b = gray_b.shape
        if w_b > 640:
            gray_b = cv2.resize(gray_b, (640, int(h_b * 640 / w_b)))

        kp1, des1 = orb.detectAndCompute(gray_a, None)
        kp2, des2 = orb.detectAndCompute(gray_b, None)

        if des1 is None or des2 is None:
            ratio = 0.0
        else:
            matches = bf.match(des1, des2)
            good_matches = [m for m in matches if m.distance < 50]
            min_features = min(len(kp1), len(kp2))
            ratio = len(good_matches) / max(1, min_features)

        print(f"[OVERLAP] Pair {i:02d} (candidate {idx_a} -> {idx_b}): overlap = {ratio*100:.1f}%")

        if ratio < threshold:
            idx_mid = (idx_a + idx_b) // 2
            print(f"[OVERLAP] ⚠ Low overlap ({ratio*100:.1f}% < {threshold*100:.1f}%). Inserting candidate {idx_mid} between {idx_a} and {idx_b}")
            current_idxs.insert(i + 1, idx_mid)
            added_count += 1
            gaps_detected.append(f"Gap between frames {i} and {i+1} ({ratio*100:.1f}% overlap)")
            i += 2  # Skip testing the newly inserted frame in this pass to prevent loops
        else:
            i += 1

    if gaps_detected:
        warning_msg = f"⚠ Low overlap warning: {len(gaps_detected)} gaps detected. Resampled +{added_count} frames. Try slower/steadier capture next time."
        state["overlap_warning"] = warning_msg
        print(f"[OVERLAP] {warning_msg}")
    else:
        state["overlap_warning"] = None
        print(f"[OVERLAP] All pairs satisfy overlap threshold of {threshold*100:.0f}%")

    return current_idxs


# ── Background thread: frame extraction (sync, CPU+IO heavy) ─────────────────
def _extract_thread(video_path: str, target: int, blur_thresh: int) -> None:
    cap = None
    try:
        upd("extracting", "Opening video file…")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video — try MP4 / MOV / AVI / WEBM / MKV format")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration     = total_frames / fps
        step         = max(1, int(fps / 8))
        
        print(f"[EXTRACT] Start extraction from {video_path}")
        print(f"[EXTRACT] Original Resolution: {orig_w}x{orig_h}, FPS: {fps:.2f}, Duration: {duration:.2f}s, Total Frames: {total_frames}")
        
        upd("extracting", f"Scanning {total_frames} frames ({duration:.1f}s at {fps:.0f}fps)…")

        def _blur(frame):
            h, w = frame.shape[:2]
            if w > 960:
                frame = cv2.resize(frame, (960, int(h * 960 / w)))
            gray = frame if (len(frame.shape) == 2 or frame.shape[2] == 1) else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()

        candidates, fi = [], 0
        while True:
            if state.get("cancel_requested", False):
                break
            ok, frame = cap.read()
            if not ok:
                break
            
            # Resize 4K to 1920px immediately after reading to free up memory
            h, w = frame.shape[:2]
            if w > 1920:
                frame = cv2.resize(frame, (1920, int(h * 1920 / w)))

            if fi % step == 0:
                blur_score = _blur(frame)
                if blur_score >= blur_thresh:
                    ok_enc, encoded_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok_enc:
                        candidates.append((fi, blur_score, encoded_img))
            fi += 1
            
    except Exception as exc:
        print(f"[EXTRACT ERROR] During video read: {exc}")
        upd("error", str(exc), error=str(exc))
        return
    finally:
        if cap is not None:
            cap.release()

    # Handle cancellation cleanly outside the cv2.VideoCapture block
    try:
        if state.get("cancel_requested", False):
            print("[EXTRACT] Cancel requested by user. Aborting...")
            upd("idle", "Ready.")
            state["cancel_requested"] = False
            return

        print(f"[EXTRACT] Found {len(candidates)} sharp candidate frames (blur_thresh={blur_thresh})")

        if not candidates:
            raise ValueError(
                f"No sharp frames found (blur_thresh={blur_thresh}). "
                "Try a slower, steadier recording."
            )

        n    = min(target, len(candidates))
        idxs = np.linspace(0, len(candidates) - 1, n, dtype=int)
        
        # Overlap Check & Dynamic Resampling (Fase 3)
        upd("extracting", f"Checking overlap & resampling frames...")
        if state.get("cancel_requested", False):
            print("[EXTRACT] Cancel requested by user. Aborting...")
            upd("idle", "Ready.")
            state["cancel_requested"] = False
            return
            
        resampled_idxs = _check_overlap_and_resample(candidates, idxs, threshold=0.15)
        n = len(resampled_idxs)
        
        upd("extracting", f"Selecting {n} frames (including dynamically resampled ones)…")

        for f in glob.glob(os.path.join(FRAMES_DIR, "*")):
            os.remove(f)

        for j, (_, _score, raw_frame) in enumerate([candidates[i] for i in resampled_idxs]):
            if state.get("cancel_requested", False):
                print("[EXTRACT] Cancel requested by user. Aborting...")
                upd("idle", "Ready.")
                state["cancel_requested"] = False
                return
            
            if isinstance(raw_frame, np.ndarray):
                frame = raw_frame
            else:
                frame = cv2.imdecode(raw_frame, cv2.IMREAD_COLOR)

            h, w = frame.shape[:2]
            if w > 1920:
                frame = cv2.resize(frame, (1920, int(h * 1920 / w)))
            cv2.imwrite(
                os.path.join(FRAMES_DIR, f"{j:04d}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            if j == 0 or j == n - 1 or j == n // 2:
                print(f"[EXTRACT] Saved frame {j:04d}.jpg - Output resolution: {frame.shape[1]}x{frame.shape[0]}")

        state["frame_count"] = n
        upd("extracted", f"\u2713 {n} sharp frames ready")
        print(f"[EXTRACT] Completed. {n} frames written to {FRAMES_DIR}")

    except Exception as exc:
        print(f"[EXTRACT ERROR] During frame processing: {exc}")
        upd("error", str(exc), error=str(exc))

# ── Background thread: GPU reconstruction + R2/D1 persistence (sync, IO heavy) ─
def _reconstruct_thread(tree_code: str, remove_background: bool = False) -> None:
    try:
        if state.get("cancel_requested", False):
            raise RuntimeError("Job cancelled by user")
        upd("reconstructing", "Connecting to Modal…")
        import modal

        files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
        if not files:
            raise ValueError("No frames found in test_images/")

        imgs = []
        if remove_background:
            try:
                from rembg import remove
                from PIL import Image
                import io
                print(f"[RECONSTRUCT] Running background removal using rembg on {len(files)} frames...")
                for idx, f in enumerate(files):
                    upd("reconstructing", f"Removing background: frame {idx+1}/{len(files)}...")
                    with open(f, "rb") as fh:
                        file_bytes = fh.read()
                    
                    # Process with rembg
                    input_img = Image.open(io.BytesIO(file_bytes))
                    output_img = remove(input_img)
                    
                    # Composite on a solid black background
                    if output_img.mode == "RGBA":
                        background = Image.new("RGBA", output_img.size, (0, 0, 0, 255))
                        composited = Image.alpha_composite(background, output_img).convert("RGB")
                    else:
                        composited = output_img.convert("RGB")
                    
                    # Convert back to jpeg bytes
                    out_io = io.BytesIO()
                    composited.save(out_io, format="JPEG", quality=95)
                    imgs.append(out_io.getvalue())
                print(f"[RECONSTRUCT] Background removal complete.")
            except Exception as bg_err:
                print(f"[RECONSTRUCT ERROR] Background removal failed: {bg_err}. Falling back to original frames.")
                imgs = []
                for f in files:
                    with open(f, "rb") as fh:
                        imgs.append(fh.read())
        else:
            for f in files:
                with open(f, "rb") as fh:
                    imgs.append(fh.read())

        upd("reconstructing", f"Sending {len(imgs)} frames to Modal A10G GPU…")
        
        t0 = time.time()
        print(f"[RECONSTRUCT] Connecting to Modal pipeline for tree_code '{tree_code}' at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}")
        print(f"[RECONSTRUCT] Uploading {len(imgs)} frames to GPU cloud (background removed: {remove_background})...")
        
        if state.get("cancel_requested", False):
            raise RuntimeError("Job cancelled by user")
        fn     = modal.Function.from_name("instantsplat-app", "run_reconstruction")
        
        t_remote_start = time.time()
        result = fn.remote(imgs)
        t_remote_end = time.time()
        
        elapsed_remote = t_remote_end - t_remote_start
        print(f"[RECONSTRUCT] GPU Reconstruction remote call completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_remote_end))}")
        print(f"[RECONSTRUCT] Remote duration: {elapsed_remote:.2f} seconds")

        out = os.path.join(OUTPUT_DIR, "result.ply")
        with open(out, "wb") as f:
            f.write(result)

        elapsed = time.time() - t0
        mb      = len(result) / 1024 / 1024
        print(f"[RECONSTRUCT] Saved output PLY: {out} ({mb:.2f} MB)")
        print(f"[RECONSTRUCT] Total pipeline runtime: {elapsed:.2f} seconds")

        upd("reconstructing", "\u2713 Reconstruction done. Estimating DBH and Carbon...")
        carbon_est = run_carbon_analysis(out, scan_id=tree_code)
        state["carbon_estimation"] = carbon_est

        if carbon_est and "error" not in carbon_est:
            try:
                upd("reconstructing", "Uploading splat file to Cloudflare R2...")
                from storage.r2_client import upload_splat, upload_thumbnail
                splat_url = upload_splat(out, tree_code)

                # Select middle representative frame as thumbnail
                thumbnail_url = None
                if files:
                    mid_idx = len(files) // 2
                    representative_frame = files[mid_idx]
                    try:
                        upd("reconstructing", "Uploading representative frame as thumbnail to R2...")
                        thumbnail_url = upload_thumbnail(representative_frame, tree_code)
                    except Exception as thumb_err:
                        print(f"Thumbnail upload error: {thumb_err}")

                upd("reconstructing", "Saving scan results to Cloudflare D1...")
                from storage.d1_client import save_scan_result
                save_scan_result(
                    tree_code=tree_code,
                    dbh_cm=carbon_est.get("dbh_cm"),
                    tinggi_m=carbon_est.get("height_m"),
                    biomassa_kg=carbon_est.get("biomass_kg"),
                    karbon_kg=carbon_est.get("carbon_kg"),
                    co2e_kg=carbon_est.get("co2e_kg"),
                    splat_file_url=splat_url,
                    confidence_note=carbon_est.get("confidence"),
                    thumbnail_url=thumbnail_url,
                )
                upd("done", f"\u2713 Done in {elapsed:.0f}s \u2014 {mb:.1f} MB Gaussian Splat ready! (Tree code: {tree_code})")
            except Exception as exc:
                print(f"Persistence error: {exc}")
                upd("error", f"Reconstruction done, but failed to save: {exc}", error=str(exc))
        else:
            err = (carbon_est or {}).get("error", "Unknown carbon analysis error")
            upd("error", f"Reconstruction done, but carbon analysis failed: {err}", error=err)

    except BaseException as exc:
        if state.get("cancel_requested", False):
            print("[RECONSTRUCT] Cancel requested by user. Aborting...")
            upd("idle", "Ready.")
            state["cancel_requested"] = False
        else:
            print(f"[RECONSTRUCT ERROR] Critical pipeline failure: {exc}")
            upd("error", str(exc), error=str(exc))

# ── Application lifespan: pre-calculate carbon if result.ply already exists ──
@asynccontextmanager
async def lifespan(application: FastAPI):
    existing_ply = os.path.join(OUTPUT_DIR, "result.ply")
    if os.path.exists(existing_ply):
        print("Found existing result.ply. Pre-calculating carbon metrics (no scan_id context at startup)...")
        # At server startup we don't know which scan_id was last processed,
        # so pass None — _load_scale_factor_for_scan will warn and use default/global.
        state["carbon_estimation"] = await asyncio.to_thread(run_carbon_analysis, existing_ply, None)

    print("=" * 50)
    print("  3D Reconstruction Pipeline (FastAPI)")
    print("=" * 50)
    yield

# ── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI(
    title="3D Tree Reconstruction Pipeline",
    description="Reconstruct 3D Gaussian Splats from video/photos and estimate tree carbon metrics.",
    version="2.0.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

# CORS Configuration
# allow_credentials=False means we can safely use allow_origins=["*"].
# Listed origins are documented for reference; the wildcard covers all of them.
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://vora-frontend-six.vercel.app",  # production Vercel deployment
    "https://vora-frontend.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    # Using wildcard because this API is credential-free (no cookies / auth headers).
    # Switch to the explicit `origins` list above if you add credential-based auth.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static / file-serving routes ─────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/viewer", include_in_schema=False)
@app.get("/viewer.html", include_in_schema=False)
async def viewer():
    return FileResponse(os.path.join(BASE_DIR, "viewer.html"))

@app.get("/gaussian-splats-3d.umd.js", include_in_schema=False)
async def splat_js():
    return FileResponse(os.path.join(BASE_DIR, "gaussian-splats-3d.umd.js"))

@app.get("/output/{fn:path}", include_in_schema=False)
async def output_file(fn: str):
    path = os.path.join(OUTPUT_DIR, fn)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)

@app.get("/frames/{fn}", include_in_schema=False)
async def frame_file(fn: str):
    path = os.path.join(FRAMES_DIR, fn)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(path)

# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/status", response_model=StatusResponse, summary="Poll pipeline state")
async def status():
    """Returns the current pipeline stage and all associated metadata."""
    frames = (
        sorted(f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(".jpg"))
        if os.path.exists(FRAMES_DIR)
        else []
    )
    return {
        **state,
        "frames":     frames,
        "has_result": os.path.exists(os.path.join(OUTPUT_DIR, "result.ply")),
    }

@app.post("/upload_video", summary="Upload a video and start frame extraction")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file (MP4/MOV/AVI/WEBM/MKV, up to 4 GB)"),
    frames: int = Form(default=25, description="Target number of frames to extract"),
    blur_thresh: int = Form(default=80, description="Minimum Laplacian blur score to keep a frame"),
):
    """
    Accepts a video upload, saves it to disk, then starts smart frame extraction
    asynchronously. Poll `/status` for progress.
    """
    allowed_extensions = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    path = os.path.join(UPLOAD_DIR, f"input{ext}")

    # Stream to disk in 1 MB chunks — safe for very large (4 GB) files
    with open(path, "wb") as out:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    state["error"] = None
    state["cancel_requested"] = False
    upd("extracting", "Video received, starting smart extraction…")
    # BackgroundTasks dispatches sync callables to thread pool automatically
    background_tasks.add_task(_extract_thread, path, frames, blur_thresh)
    return {"queued": True}

@app.post("/use_photos", summary="Upload photos directly as extracted frames")
async def use_photos(photos: List[UploadFile] = File(...)):
    """
    Accepts a list of photos as direct input (skips video extraction step).
    Poll `/status` for updated state.
    """
    if not photos:
        raise HTTPException(status_code=400, detail="No photos provided")

    for f in glob.glob(os.path.join(FRAMES_DIR, "*")):
        os.remove(f)

    for i, p in enumerate(photos):
        data = await p.read()
        with open(os.path.join(FRAMES_DIR, f"{i:04d}.jpg"), "wb") as out:
            out.write(data)

    n = len(photos)
    state["frame_count"] = n
    state["error"] = None
    state["cancel_requested"] = False
    upd("extracted", f"\u2713 {n} photos loaded")
    return {"success": True, "count": n}

@app.post("/reconstruct", summary="Start GPU reconstruction on extracted frames")
async def reconstruct(
    background_tasks: BackgroundTasks,
    # Accept tree_code and remove_background from JSON body OR query string for maximum flexibility
    body: Optional[ReconstructRequest] = Body(default=None),
    tree_code_query: Optional[str] = Query(default=None, alias="tree_code"),
    remove_bg_query: Optional[bool] = Query(default=None, alias="remove_background"),
):
    """
    Dispatches the GPU reconstruction job (via Modal) as a background task.
    Returns immediately with `tree_code` so the client can track this scan.
    If no `tree_code` is provided, one is auto-generated in format `POHON-XXXX`.
    """
    if state["stage"] not in ("extracted", "done", "error"):
        raise HTTPException(status_code=400, detail="Not ready — extract frames first")

    # Force remove_background to False as requested by user
    remove_bg = False

    # Reset cancellation request
    state["cancel_requested"] = False

    # Generate or resolve tree code
    import random
    final_code = tree_code_query or (body.tree_code if body else None) or f"POHON-{random.randint(1000, 9999)}"
    final_code = final_code.strip().upper()

    state["error"] = None
    upd("reconstructing", "Queuing reconstruction…")
    background_tasks.add_task(_reconstruct_thread, final_code, remove_bg)
    return {"started": True, "tree_code": final_code}


@app.post("/cancel", summary="Cancel active pipeline job")
async def cancel_job():
    """Signals cancellation to background threads and resets state to idle."""
    state["cancel_requested"] = True
    upd("idle", "Ready (Previous job cancelled).")
    state["overlap_warning"] = None
    state["error"] = None
    return {"success": True, "message": "Cancellation request registered."}

@app.get(
    "/history/{tree_code}",
    response_model=HistoryResponse,
    summary="Fetch scan history for a tree code",
)
async def history(tree_code: str):
    """
    Returns all historical scan records for `tree_code` from Cloudflare D1,
    ordered by scan_date descending. The underlying `requests` call is wrapped
    in `asyncio.to_thread` so it does not block the event loop.
    """
    try:
        from storage.d1_client import get_scan_history
        records = await asyncio.to_thread(get_scan_history, tree_code)
        return {"success": True, "tree_code": tree_code, "history": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get(
    "/scans",
    response_model=ScansResponse,
    summary="Fetch all scan history with pagination",
)
async def get_scans(limit: int = 20, offset: int = 0):
    """
    Returns all scan records from Cloudflare D1 with optional limit and offset,
    ordered by scan_date descending.
    """
    try:
        from storage.d1_client import get_all_scans
        records = await asyncio.to_thread(get_all_scans, limit, offset)
        return {"success": True, "scans": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get(
    "/splat-proxy/{tree_code}/{filename}",
    summary="Proxy and stream splat files from Cloudflare R2",
)
async def splat_proxy(tree_code: str, filename: str):
    """
    Proxies splat/ply files from Cloudflare R2, streaming them back to the client.
    Bypasses DNS/SSL blocks on R2.dev subdomains by using the direct S3 API.
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("R2_BUCKET_NAME")

    if not all([account_id, access_key, secret_key, bucket_name]):
        raise HTTPException(status_code=500, detail="R2 storage credentials not configured")

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    object_key = f"tree_scans/{tree_code}/{filename}"

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name="auto"
        )
        # Fetch object metadata and stream from R2 (boto3 call is blocking, so run in thread pool)
        response = await asyncio.to_thread(
            s3.get_object,
            Bucket=bucket_name,
            Key=object_key
        )
    except Exception as exc:
        err_msg = str(exc)
        if "NoSuchKey" in err_msg:
            raise HTTPException(status_code=404, detail="File not found in storage")
        raise HTTPException(status_code=500, detail=f"Failed to fetch from R2: {err_msg}")

    body = response["Body"]

    def iter_chunks():
        try:
            while True:
                # Read 1 MB chunk (blocking call, but executed safely in thread by StreamingResponse)
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            body.close()

    media_type = "application/octet-stream"
    if filename.endswith(".ply"):
        media_type = "application/x-ply"

    headers = {
        "Cache-Control": "public, max-age=31536000, immutable",
        "Content-Length": str(response.get("ContentLength", ""))
    }

    return StreamingResponse(iter_chunks(), media_type=media_type, headers=headers)

# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
