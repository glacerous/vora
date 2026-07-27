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
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

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
}

def upd(stage: str, msg: str, **kw: Any) -> None:
    state.update({"stage": stage, "message": msg, **kw})

# ── Pydantic request / response models ───────────────────────────────────────

class ReconstructRequest(BaseModel):
    """Optional JSON body for POST /reconstruct."""
    tree_code: Optional[str] = None

class StatusResponse(BaseModel):
    stage: str
    message: str
    frame_count: int
    error: Optional[str]
    carbon_estimation: Optional[Any]
    frames: List[str]
    has_result: bool

class HistoryResponse(BaseModel):
    success: bool
    tree_code: str
    history: List[Any]

# ── Helper: carbon analysis (sync, CPU-bound — always called inside thread) ──
def run_carbon_analysis(ply_path: str) -> dict:
    try:
        from carbon.allometric import estimate_carbon
        from carbon.dbh_extractor import extract_dbh

        dbh_result = extract_dbh(
            ply_path=ply_path, scale_factor=1.0,
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
            "biomass_kg":              carbon_result["total_biomass_kg"],
            "above_ground_biomass_kg": carbon_result["above_ground_biomass_kg"],
            "below_ground_biomass_kg": carbon_result["below_ground_biomass_kg"],
            "carbon_kg":               carbon_result["carbon_kg"],
            "co2e_kg":                 carbon_result["co2e_kg"],
            "disclaimer":              carbon_result["disclaimer"],
        }
    except Exception as exc:
        return {"error": f"Failed to compute carbon metrics: {exc}"}

# ── Background thread: frame extraction (sync, CPU+IO heavy) ─────────────────
def _extract_thread(video_path: str, target: int, blur_thresh: int) -> None:
    try:
        upd("extracting", "Opening video file…")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video — try MP4 / MOV / AVI format")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration     = total_frames / fps
        step         = max(1, int(fps / 8))
        upd("extracting", f"Scanning {total_frames} frames ({duration:.1f}s at {fps:.0f}fps)…")

        def _blur(frame):
            return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

        candidates, fi = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fi % step == 0 and _blur(frame) >= blur_thresh:
                candidates.append((fi, _blur(frame), frame.copy()))
            fi += 1
        cap.release()

        if not candidates:
            raise ValueError(
                f"No sharp frames found (blur_thresh={blur_thresh}). "
                "Try a slower, steadier recording."
            )

        n    = min(target, len(candidates))
        idxs = np.linspace(0, len(candidates) - 1, n, dtype=int)
        upd("extracting", f"Selecting {n} frames from {len(candidates)} sharp candidates…")

        for f in glob.glob(os.path.join(FRAMES_DIR, "*")):
            os.remove(f)

        for j, (_, _score, frame) in enumerate([candidates[i] for i in idxs]):
            h, w = frame.shape[:2]
            if w > 1920:
                frame = cv2.resize(frame, (1920, int(h * 1920 / w)))
            cv2.imwrite(
                os.path.join(FRAMES_DIR, f"{j:04d}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

        state["frame_count"] = n
        upd("extracted", f"\u2713 {n} sharp frames ready")

    except Exception as exc:
        upd("error", str(exc), error=str(exc))

# ── Background thread: GPU reconstruction + R2/D1 persistence (sync, IO heavy) ─
def _reconstruct_thread(tree_code: str) -> None:
    try:
        upd("reconstructing", "Connecting to Modal…")
        import modal

        files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
        if not files:
            raise ValueError("No frames found in test_images/")

        imgs = []
        for f in files:
            with open(f, "rb") as fh:
                imgs.append(fh.read())

        upd("reconstructing", f"Sending {len(imgs)} frames to Modal A10G GPU…")
        t0 = time.time()

        fn     = modal.Function.from_name("instantsplat-app", "run_reconstruction")
        result = fn.remote(imgs)

        out = os.path.join(OUTPUT_DIR, "result.ply")
        with open(out, "wb") as f:
            f.write(result)

        elapsed = time.time() - t0
        mb      = len(result) / 1024 / 1024

        upd("reconstructing", "\u2713 Reconstruction done. Estimating DBH and Carbon...")
        carbon_est = run_carbon_analysis(out)
        state["carbon_estimation"] = carbon_est

        if carbon_est and "error" not in carbon_est:
            try:
                upd("reconstructing", "Uploading splat file to Cloudflare R2...")
                from storage.r2_client import upload_splat
                splat_url = upload_splat(out, tree_code)

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
                )
                upd("done", f"\u2713 Done in {elapsed:.0f}s \u2014 {mb:.1f} MB Gaussian Splat ready! (Tree code: {tree_code})")
            except Exception as exc:
                print(f"Persistence error: {exc}")
                upd("error", f"Reconstruction done, but failed to save: {exc}", error=str(exc))
        else:
            err = (carbon_est or {}).get("error", "Unknown carbon analysis error")
            upd("error", f"Reconstruction done, but carbon analysis failed: {err}", error=err)

    except Exception as exc:
        upd("error", str(exc), error=str(exc))

# ── Application lifespan: pre-calculate carbon if result.ply already exists ──
@asynccontextmanager
async def lifespan(application: FastAPI):
    existing_ply = os.path.join(OUTPUT_DIR, "result.ply")
    if os.path.exists(existing_ply):
        print("Found existing result.ply. Pre-calculating carbon metrics...")
        state["carbon_estimation"] = await asyncio.to_thread(run_carbon_analysis, existing_ply)

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
# WARNING: Narrow this list down to your specific production Vercel frontend domains before submission.
origins = [
    "http://localhost:3000",
    "http://localhost:5173",  # Common Vite dev port
    # "https://your-frontend.vercel.app",  # Add your production Vercel domain here
]

app.add_middleware(
    CORSMiddleware,
    # For initial loose development, allow_origins=["*"] is used.
    # Note: To support credentials (cookies, authorization headers), allow_origins MUST be set to
    # an explicit list of origins (like the 'origins' variable above), and allow_credentials must be True.
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
    video: UploadFile = File(..., description="Video file (MP4/MOV/AVI, up to 4 GB)"),
    frames: int = Form(default=25, description="Target number of frames to extract"),
    blur_thresh: int = Form(default=80, description="Minimum Laplacian blur score to keep a frame"),
):
    """
    Accepts a video upload, saves it to disk, then starts smart frame extraction
    asynchronously. Poll `/status` for progress.
    """
    ext  = os.path.splitext(video.filename or ".mp4")[1].lower() or ".mp4"
    path = os.path.join(UPLOAD_DIR, f"input{ext}")

    # Stream to disk in 1 MB chunks — safe for very large (4 GB) files
    with open(path, "wb") as out:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    state["error"] = None
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
    upd("extracted", f"\u2713 {n} photos loaded")
    return {"success": True, "count": n}

@app.post("/reconstruct", summary="Start GPU reconstruction on extracted frames")
async def reconstruct(
    background_tasks: BackgroundTasks,
    # Accept tree_code from JSON body OR query string for maximum flexibility
    body: Optional[ReconstructRequest] = Body(default=None),
    tree_code_query: Optional[str] = Query(default=None, alias="tree_code"),
):
    """
    Dispatches the GPU reconstruction job (via Modal) as a background task.
    Returns immediately with `tree_code` so the client can track this scan.
    If no `tree_code` is provided, one is auto-generated in format `POHON-XXXX`.
    """
    if state["stage"] not in ("extracted", "done", "error"):
        raise HTTPException(status_code=400, detail="Not ready — extract frames first")

    tree_code = (body.tree_code if body else None) or tree_code_query
    if not tree_code:
        from storage.d1_client import generate_tree_code
        tree_code = generate_tree_code()

    state["error"] = None
    upd("reconstructing", "Queuing reconstruction…")
    background_tasks.add_task(_reconstruct_thread, tree_code)
    return {"started": True, "tree_code": tree_code}

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

# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_fastapi:app", host="0.0.0.0", port=8001, reload=False)
