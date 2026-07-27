#!/usr/bin/env python3
"""
3D Reconstruction Pipeline Server
Run with: py server.py
Open:     http://localhost:8000
"""
import os
from dotenv import load_dotenv
load_dotenv()

import glob
import threading
import time
import shutil

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRAMES_DIR = os.path.join(BASE_DIR, "test_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

for d in (UPLOAD_DIR, FRAMES_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB max upload

# ── Global pipeline state ───────────────────────────────────────────────────
state = {
    "stage":       "idle",   # idle | extracting | extracted | reconstructing | done | error
    "message":     "Ready.",
    "frame_count": 0,
    "error":       None,
    "carbon_estimation": None,
}

def upd(stage, msg, **kw):
    state.update({"stage": stage, "message": msg, **kw})

# ── Smart frame extraction ──────────────────────────────────────────────────
def laplacian_blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _extract_thread(video_path, target, blur_thresh):
    try:
        upd("extracting", "Opening video file…")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video — try MP4 / MOV / AVI format")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration     = total_frames / fps

        # Sample at ~8 fps to build candidates (avoids processing every single frame)
        step = max(1, int(fps / 8))
        upd("extracting", f"Scanning {total_frames} frames ({duration:.1f}s at {fps:.0f}fps)…")

        candidates = []
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fi % step == 0:
                score = laplacian_blur_score(frame)
                if score >= blur_thresh:
                    candidates.append((fi, score, frame.copy()))
            fi += 1
        cap.release()

        if not candidates:
            raise ValueError(
                f"No sharp frames found (blur_thresh={blur_thresh}). "
                "Try a slower, steadier recording."
            )

        n = min(target, len(candidates))
        upd("extracting", f"Selecting {n} frames from {len(candidates)} sharp candidates…")

        # Evenly spaced across the timeline for maximum scene coverage
        idxs     = np.linspace(0, len(candidates) - 1, n, dtype=int)
        selected = [candidates[i] for i in idxs]

        # Clear previous frames
        for f in glob.glob(os.path.join(FRAMES_DIR, "*")):
            os.remove(f)

        # Save — resize to max 1920 px wide for faster GPU processing
        for j, (_, score, frame) in enumerate(selected):
            h, w = frame.shape[:2]
            if w > 1920:
                frame = cv2.resize(frame, (1920, int(h * 1920 / w)))
            out = os.path.join(FRAMES_DIR, f"{j:04d}.jpg")
            cv2.imwrite(out, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        state["frame_count"] = n
        upd("extracted", f"✓ {n} sharp frames ready")

    except Exception as e:
        upd("error", str(e), error=str(e))


def run_carbon_analysis(ply_path):
    try:
        from carbon.dbh_extractor import extract_dbh
        from carbon.allometric import estimate_carbon
        
        # Default scale_factor is 1.0 (will load calibration.json if present)
        dbh_result = extract_dbh(
            ply_path=ply_path, 
            scale_factor=1.0, 
            vertical_axis='z', 
            breast_height=1.3
        )
        
        if "error" in dbh_result:
            return {"error": dbh_result["error"]}
            
        carbon_result = estimate_carbon(
            dbh_cm=dbh_result['dbh_cm'], 
            height_m=dbh_result['height_m'], 
            wood_density=0.6
        )
        
        return {
            "dbh_cm": dbh_result["dbh_cm"],
            "height_m": dbh_result["height_m"],
            "confidence": dbh_result["confidence_note"],
            "method": dbh_result["method"],
            "slice_points_count": dbh_result["slice_points_count"],
            "mean_fit_error_cm": dbh_result["mean_fit_error_cm"],
            "biomass_kg": carbon_result["total_biomass_kg"],
            "above_ground_biomass_kg": carbon_result["above_ground_biomass_kg"],
            "below_ground_biomass_kg": carbon_result["below_ground_biomass_kg"],
            "carbon_kg": carbon_result["carbon_kg"],
            "co2e_kg": carbon_result["co2e_kg"],
            "disclaimer": carbon_result["disclaimer"]
        }
    except Exception as e:
        return {"error": f"Failed to compute carbon metrics: {e}"}


def _reconstruct_thread(tree_code):
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
        mb = len(result) / 1024 / 1024
        
        # Run DBH and carbon estimation
        upd("reconstructing", "✓ Reconstruction done. Estimating DBH and Carbon...")
        carbon_est = run_carbon_analysis(out)
        state["carbon_estimation"] = carbon_est
        
        # Persist results to Cloudflare R2 and D1
        if carbon_est and "error" not in carbon_est:
            try:
                upd("reconstructing", "Uploading splat file to Cloudflare R2...")
                from storage.r2_client import upload_splat
                splat_file_url = upload_splat(out, tree_code)
                
                upd("reconstructing", "Saving scan results to Cloudflare D1...")
                from storage.d1_client import save_scan_result
                save_scan_result(
                    tree_code=tree_code,
                    dbh_cm=carbon_est.get("dbh_cm"),
                    tinggi_m=carbon_est.get("height_m"),
                    biomassa_kg=carbon_est.get("biomass_kg"),
                    karbon_kg=carbon_est.get("carbon_kg"),
                    co2e_kg=carbon_est.get("co2e_kg"),
                    splat_file_url=splat_file_url,
                    confidence_note=carbon_est.get("confidence")
                )
                upd("done", f"✓ Done in {elapsed:.0f}s — {mb:.1f} MB Gaussian Splat ready! (Tree code: {tree_code})")
            except Exception as e:
                print(f"Persistence error: {e}")
                upd("error", f"Reconstruction done, but failed to save: {e}", error=str(e))
        else:
            error_msg = carbon_est.get("error") if carbon_est else "Unknown carbon analysis error"
            upd("error", f"Reconstruction done, but carbon analysis failed: {error_msg}", error=error_msg)

    except Exception as e:
        upd("error", str(e), error=str(e))


# ── HTTP Routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/viewer")
@app.route("/viewer.html")
def viewer():
    return send_from_directory(BASE_DIR, "viewer.html")

@app.route("/gaussian-splats-3d.umd.js")
def splat_js():
    return send_from_directory(BASE_DIR, "gaussian-splats-3d.umd.js")

@app.route("/output/<path:fn>")
def output_file(fn):
    return send_from_directory(OUTPUT_DIR, fn)

@app.route("/frames/<fn>")
def frame_file(fn):
    return send_from_directory(FRAMES_DIR, fn)

@app.route("/status")
def status():
    frames = (
        sorted(f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(".jpg"))
        if os.path.exists(FRAMES_DIR) else []
    )
    return jsonify({
        **state,
        "frames":     frames,
        "has_result": os.path.exists(os.path.join(OUTPUT_DIR, "result.ply")),
    })

@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    f      = request.files["video"]
    target = int(request.form.get("frames", 25))
    blur   = int(request.form.get("blur_thresh", 80))

    ext  = os.path.splitext(f.filename or ".mp4")[1].lower() or ".mp4"
    path = os.path.join(UPLOAD_DIR, f"input{ext}")
    f.save(path)

    state["error"] = None
    upd("extracting", "Video received, starting smart extraction…")
    threading.Thread(target=_extract_thread, args=(path, target, blur), daemon=True).start()
    return jsonify({"queued": True})

@app.route("/use_photos", methods=["POST"])
def use_photos():
    if "photos" not in request.files:
        return jsonify({"error": "No photos"}), 400

    photos = request.files.getlist("photos")
    for f in glob.glob(os.path.join(FRAMES_DIR, "*")):
        os.remove(f)

    for i, p in enumerate(photos):
        p.save(os.path.join(FRAMES_DIR, f"{i:04d}.jpg"))

    n = len(photos)
    state["frame_count"] = n
    state["error"] = None
    upd("extracted", f"✓ {n} photos loaded")
    return jsonify({"success": True, "count": n})

@app.route("/reconstruct", methods=["POST"])
def reconstruct():
    if state["stage"] not in ("extracted", "done", "error"):
        return jsonify({"error": "Not ready — extract frames first"}), 400
    
    # Read tree_code from request json, form, or args
    tree_code = None
    if request.is_json:
        data = request.get_json() or {}
        tree_code = data.get("tree_code")
    else:
        tree_code = request.form.get("tree_code") or request.args.get("tree_code")
        
    if not tree_code:
        from storage.d1_client import generate_tree_code
        tree_code = generate_tree_code()
        
    state["error"] = None
    upd("reconstructing", "Queuing reconstruction…")
    threading.Thread(target=_reconstruct_thread, args=(tree_code,), daemon=True).start()
    return jsonify({"started": True, "tree_code": tree_code})

@app.route("/history/<tree_code>", methods=["GET"])
def history(tree_code):
    try:
        from storage.d1_client import get_scan_history
        records = get_scan_history(tree_code)
        return jsonify({"success": True, "tree_code": tree_code, "history": records})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    import socket
    def find_free_port(start_port=8000):
        port = start_port
        while True:
            try:
                # Try binding to 127.0.0.1 first because on Windows, binding to 0.0.0.0 might succeed
                # even if 127.0.0.1 is occupied by another process.
                # Do NOT set SO_REUSEADDR on Windows to avoid false positive bindings on busy ports.
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", port))
                s.close()
                
                # Also try 0.0.0.0 just in case
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("0.0.0.0", port))
                s.close()
                
                return port
            except OSError:
                port += 1

    port = find_free_port(8000)
    
    # Pre-calculate carbon metrics if result.ply already exists
    existing_ply = os.path.join(OUTPUT_DIR, "result.ply")
    if os.path.exists(existing_ply):
        print("Found existing result.ply. Pre-calculating carbon metrics...")
        state["carbon_estimation"] = run_carbon_analysis(existing_ply)
        
    print("=" * 50)
    print("  3D Reconstruction Pipeline")
    print(f"  http://localhost:{port}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
