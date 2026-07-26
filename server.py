#!/usr/bin/env python3
"""
3D Reconstruction Pipeline Server
Run with: py server.py
Open:     http://localhost:8000
"""
import os
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


def _reconstruct_thread():
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
        upd("done", f"✓ Done in {elapsed:.0f}s — {mb:.1f} MB Gaussian Splat ready!")

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
    state["error"] = None
    upd("reconstructing", "Queuing reconstruction…")
    threading.Thread(target=_reconstruct_thread, daemon=True).start()
    return jsonify({"started": True})

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
    print("=" * 50)
    print("  3D Reconstruction Pipeline")
    print(f"  http://localhost:{port}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
