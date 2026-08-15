import os
import time
import modal

global_import_time = time.time()

app = modal.App("instantsplat-app")
progress_dict = modal.Dict.from_name("instantsplat-progress-dict", create_if_missing=True)

# GPU Configuration - easy to swap to "a100" or "h100" if "a10g" triggers CUDA Out Of Memory (OOM) errors
GPU_CONFIG = "a10g"

# Clone the repository recursively and download the MASt3R checkpoint into it during image build
image = (
    modal.Image.from_registry("dockerzhiwen/instantsplat_public:2.0")
    .run_commands(
        "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git libgl1-mesa-glx libglib2.0-0 ffmpeg",
        "git clone --recursive https://github.com/NVlabs/InstantSplat.git /workspace/InstantSplat",
        "/opt/conda/bin/pip install 'numpy==1.26.0' open3d plyfile icecream pyquaternion configargparse rembg opencv-python-headless boto3",
        "TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9' /opt/conda/bin/pip install --no-build-isolation --no-deps /workspace/InstantSplat/submodules/simple-knn",
        "TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9' /opt/conda/bin/pip install --no-build-isolation --no-deps /workspace/InstantSplat/submodules/diff-gaussian-rasterization",
        "TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9' /opt/conda/bin/pip install --no-build-isolation --no-deps /workspace/InstantSplat/submodules/fused-ssim",
        "python3 -c \"import urllib.request, os; "
        "url = 'https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'; "
        "fn = 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'; "
        "p = '/workspace/InstantSplat/mast3r/checkpoints'; "
        "os.makedirs(p, exist_ok=True); "
        "t = os.path.join(p, fn); "
        "print('Checking/Downloading checkpoint...'); "
        "urllib.request.urlretrieve(url, t) if not os.path.exists(t) else print('exists');\"",
        "python3 -c \"import urllib.request, os; "
        "url = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx'; "
        "fn = 'u2net.onnx'; "
        "p = '/root/.u2net'; "
        "os.makedirs(p, exist_ok=True); "
        "t = os.path.join(p, fn); "
        "print('Downloading u2net.onnx...'); "
        "urllib.request.urlretrieve(url, t) if not os.path.exists(t) else print('exists');\""
    )
    .run_commands(
        "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl",
        "curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs",
        "git clone https://github.com/mkkellogg/GaussianSplats3D.git /workspace/GaussianSplats3D && cd /workspace/GaussianSplats3D && npm install && npm run build"
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: MASt3R Geometric Scale Prior
# Reads COLMAP camera centres from init_geo.py output (MASt3R metric space)
# and verifies the coordinate system is plausibly in metres.
# MediaPipe Pose calibration has been removed — it was incompatible with tree
# scan workflows where a full-body person is almost never in frame.
# ─────────────────────────────────────────────────────────────────────────────

def _read_colmap_camera_centers(images_bin_path):
    """
    Parse COLMAP binary images.bin and return camera centre positions (Nx3 float64).
    Camera centre = -R^T @ t  (COLMAP convention: t is camera-in-world translation).
    Returns numpy array or None on failure.
    """
    import struct
    import numpy as np

    def _quat_to_rot(qw, qx, qy, qz):
        return np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz),   2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz),   1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),   2*(qy*qz + qw*qx),   1 - 2*(qx**2 + qy**2)],
        ], dtype=np.float64)

    centers = []
    try:
        with open(images_bin_path, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                f.read(4)                           # image_id (uint32)
                qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
                tx, ty, tz     = struct.unpack("<3d", f.read(24))
                f.read(4)                           # camera_id (uint32)
                # Read null-terminated filename
                while True:
                    c = f.read(1)
                    if c in (b"\x00", b""):
                        break
                # Skip 2-D point observations
                num_pts2d = struct.unpack("<Q", f.read(8))[0]
                f.read(num_pts2d * 24)              # x(8) + y(8) + point3d_id(8)
                R = _quat_to_rot(qw, qx, qy, qz)
                t = np.array([tx, ty, tz])
                centers.append(-R.T @ t)
    except Exception as exc:
        print(f"[SCALE-PRIOR] Failed to parse {images_bin_path}: {exc}")
        return None
    return np.array(centers, dtype=np.float64) if centers else None


def _read_colmap_images_txt(images_txt_path):
    """
    Fallback: parse COLMAP text images.txt (same camera-centre derivation).
    Returns numpy array or None.
    """
    import numpy as np

    def _quat_to_rot(qw, qx, qy, qz):
        return np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz),   2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz),   1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),   2*(qy*qz + qw*qx),   1 - 2*(qx**2 + qy**2)],
        ], dtype=np.float64)

    centers = []
    try:
        with open(images_txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                try:
                    qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    tx, ty, tz     = float(parts[5]), float(parts[6]), float(parts[7])
                except ValueError:
                    continue
                R = _quat_to_rot(qw, qx, qy, qz)
                t = np.array([tx, ty, tz])
                centers.append(-R.T @ t)
                next(f, None)  # skip POINTS2D line
    except Exception as exc:
        print(f"[SCALE-PRIOR] Failed to parse {images_txt_path}: {exc}")
        return None
    return np.array(centers, dtype=np.float64) if centers else None


def _derive_mast3r_scale_prior(source_path, detected_n_views):
    """
    Derive a geometric scale prior from MASt3R's init_geo.py COLMAP output.

    Strategy:
      1. Read camera centres from sparse_N/images.bin (MASt3R metric space, metres).
      2. Compute mean pairwise distance between camera centres.
      3. If spacing is plausible for a tree-scan walkabout (0.05 – 15 m),
         the coordinate system IS already in metres → scale_factor = 1.0.
      4. Return a scale_calibration dict with source='estimated_geometric_prior'.

    This does NOT require a person in frame, a reference object, or any extra
    hardware.  Typical accuracy: ~5-9% relative error on absolute scale.
    """
    import numpy as np
    import os

    sparse_dir = os.path.join(source_path, f"sparse_{detected_n_views}")
    images_bin = os.path.join(sparse_dir, "images.bin")
    images_txt = os.path.join(sparse_dir, "images.txt")

    # List what's actually in the sparse dir for diagnostics
    if os.path.exists(sparse_dir):
        contents = os.listdir(sparse_dir)
        print(f"[SCALE-PRIOR] sparse_{detected_n_views}/ contains: {contents}")
    else:
        print(f"[SCALE-PRIOR] sparse_{detected_n_views}/ not found at {sparse_dir}")
        return {
            "is_calibrated": False,
            "source": "uncalibrated",
            "scale_factor": 1.0,
            "reason": f"sparse_{detected_n_views}/ directory not found after init_geo",
        }

    # Read camera centres
    centers = None
    if os.path.exists(images_bin):
        print(f"[SCALE-PRIOR] Reading COLMAP binary images.bin...")
        centers = _read_colmap_camera_centers(images_bin)
    elif os.path.exists(images_txt):
        print(f"[SCALE-PRIOR] Reading COLMAP text images.txt...")
        centers = _read_colmap_images_txt(images_txt)
    else:
        print(f"[SCALE-PRIOR] No images.bin or images.txt found in {sparse_dir}")
        return {
            "is_calibrated": False,
            "source": "uncalibrated",
            "scale_factor": 1.0,
            "reason": "COLMAP images file not found in init_geo sparse output",
        }

    if centers is None or len(centers) < 2:
        return {
            "is_calibrated": False,
            "source": "uncalibrated",
            "scale_factor": 1.0,
            "reason": f"Could not extract camera centres (found {len(centers) if centers is not None else 0} cameras)",
        }

    # Mean pairwise distance between camera centres
    n = len(centers)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(centers[i] - centers[j])))
    mean_dist = float(np.mean(dists))
    min_dist  = float(np.min(dists))
    max_dist  = float(np.max(dists))

    print(f"[SCALE-PRIOR] {n} cameras | pairwise dist: mean={mean_dist:.4f} min={min_dist:.4f} max={max_dist:.4f} units")

    # Sanity gate: for a tree walk-around, camera spacing should be 0.05-15 m.
    # Outside this range the coordinate system is NOT metric (or data is degenerate).
    PLAUSIBLE_MIN = 0.05   # metres — closer than this means reconstruction collapsed
    PLAUSIBLE_MAX = 15.0   # metres — farther means units are not metres
    if not (PLAUSIBLE_MIN <= mean_dist <= PLAUSIBLE_MAX):
        print(f"[SCALE-PRIOR] Camera spacing {mean_dist:.4f} outside plausible metric range "
              f"[{PLAUSIBLE_MIN}, {PLAUSIBLE_MAX}] — staying uncalibrated")
        return {
            "is_calibrated": False,
            "source": "uncalibrated",
            "scale_factor": 1.0,
            "reason": (
                f"MASt3R camera spacing ({mean_dist:.3f} units) outside plausible metric range "
                f"— coordinate system may not be in metres"
            ),
            "mean_camera_spacing": mean_dist,
        }

    print(f"[SCALE-PRIOR] Geometric prior accepted: scale_factor=1.0 "
          f"(MASt3R metric checkpoint, mean_camera_spacing={mean_dist:.3f}m, {n} cameras)")
    return {
        "is_calibrated": True,
        "source": "estimated_geometric_prior",
        "scale_factor": 1.0,   # MASt3R _metric checkpoint outputs metres directly
        "reason": (
            f"geometri MASt3R init_geo ({n} kamera, jarak rata-rata={mean_dist:.3f}m) — "
            f"estimasi prior, akurasi ~5-9% relatif"
        ),
        "mean_camera_spacing_m": mean_dist,
        "n_cameras": n,
    }

def _derive_scale_from_vio_poses(camera_poses, source_path, detected_n_views):
    """
    Derive scale factor by comparing the total path length of the phone's
    VIO trajectory (ground truth in metres) against the reconstructed MASt3R
    camera centers path.
    """
    import numpy as np
    import os

    if not camera_poses or len(camera_poses) < 2:
        return {
            "is_calibrated": False,
            "source": "uncalibrated",
            "scale_factor": 1.0,
            "reason": "VIO poses too sparse or empty",
        }

    # 1. Compute VIO camera path length
    vio_pts = []
    for pose in camera_poses:
        if "x" in pose and "y" in pose and "z" in pose:
            vio_pts.append([float(pose["x"]), float(pose["y"]), float(pose["z"])])
    if len(vio_pts) < 2:
        return {
            "is_calibrated": False,
            "source": "uncalibrated",
            "scale_factor": 1.0,
            "reason": "Invalid coordinate keys in VIO poses",
        }
    
    vio_pts = np.array(vio_pts)
    vio_dists = np.linalg.norm(np.diff(vio_pts, axis=0), axis=1)
    vio_path_len = float(np.sum(vio_dists))

    # 2. Get Reconstruction camera centers in chronological order
    sparse_dir = os.path.join(source_path, f"sparse_{detected_n_views}")
    images_bin = os.path.join(sparse_dir, "images.bin")
    images_txt = os.path.join(sparse_dir, "images.txt")

    # Read camera poses and map filename to position
    poses_map = {}
    
    def _quat_to_rot(qw, qx, qy, qz):
        return np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz),   2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz),   1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),   2*(qy*qz + qw*qx),   1 - 2*(qx**2 + qy**2)],
        ], dtype=np.float64)

    try:
        if os.path.exists(images_bin):
            import struct
            with open(images_bin, "rb") as f:
                num_images = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num_images):
                    f.read(4)                           # image_id
                    qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
                    tx, ty, tz     = struct.unpack("<3d", f.read(24))
                    f.read(4)                           # camera_id
                    # filename
                    fn_chars = []
                    while True:
                        c = f.read(1)
                        if c in (b"\x00", b""):
                            break
                        fn_chars.append(c.decode("ascii", errors="ignore"))
                    filename = "".join(fn_chars)
                    num_pts2d = struct.unpack("<Q", f.read(8))[0]
                    f.read(num_pts2d * 24)
                    
                    R = _quat_to_rot(qw, qx, qy, qz)
                    t = np.array([tx, ty, tz])
                    center = -R.T @ t
                    poses_map[filename] = center
        elif os.path.exists(images_txt):
            with open(images_txt, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    try:
                        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        tx, ty, tz     = float(parts[5]), float(parts[6]), float(parts[7])
                        filename = parts[9]
                    except ValueError:
                        continue
                    R = _quat_to_rot(qw, qx, qy, qz)
                    t = np.array([tx, ty, tz])
                    center = -R.T @ t
                    poses_map[filename] = center
                    next(f, None)
    except Exception as exc:
        print(f"[ARCORE-VIO] Failed to parse COLMAP sparse folder: {exc}")
        return {
            "is_calibrated": False,
            "source": "uncalibrated",
            "scale_factor": 1.0,
            "reason": f"Exception parsing sparse folder: {exc}",
        }

    if len(poses_map) < 2:
        return {
            "is_calibrated": False,
            "source": "uncalibrated",
            "scale_factor": 1.0,
            "reason": f"Insufficient reconstructed cameras in sparse folder (found {len(poses_map)})",
        }

    # Sort filenames alphabetically/chronologically: '000.jpg', '001.jpg', etc.
    sorted_filenames = sorted(poses_map.keys())
    recon_pts = np.array([poses_map[fn] for fn in sorted_filenames])
    recon_dists = np.linalg.norm(np.diff(recon_pts, axis=0), axis=1)
    recon_path_len = float(np.sum(recon_dists))

    if recon_path_len <= 0:
        return {
            "is_calibrated": False,
            "source": "uncalibrated",
            "scale_factor": 1.0,
            "reason": "Reconstructed path length is zero",
        }

    scale_factor = vio_path_len / recon_path_len
    
    print(f"[ARCORE-VIO] Success! VIO path: {vio_path_len:.3f}m | Recon path: {recon_path_len:.3f} units | scale_factor: {scale_factor:.6f}")
    return {
        "is_calibrated": True,
        "source": "arcore_vio",
        "scale_factor": scale_factor,
        "reason": f"skala dihitung via rasio lintasan kamera VIO ({vio_path_len:.2f}m / {recon_path_len:.2f} unit)",
        "vio_path_length_m": vio_path_len,
        "recon_path_length": recon_path_len,
    }

def parse_ply_coords(ply_path):
    import numpy as np
    try:
        with open(ply_path, "rb") as f:
            num_vertices = 0
            is_binary = False
            raw_props = []
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("format binary_little_endian"):
                    is_binary = True
                elif line.startswith("element vertex"):
                    num_vertices = int(line.split()[-1])
                elif line.startswith("property"):
                    parts = line.split()
                    if len(parts) >= 3:
                        raw_props.append((parts[1], parts[2]))
                elif line == "end_header":
                    break
            if num_vertices <= 0 or not is_binary:
                return None
            dtype_map = []
            for p_type, p_name in raw_props:
                if p_type in ("float", "float32"):   dtype_map.append((p_name, "<f4"))
                elif p_type in ("int", "int32", "uint"): dtype_map.append((p_name, "<i4"))
                elif p_type in ("uchar", "uint8"):   dtype_map.append((p_name, "u1"))
                else:                                 dtype_map.append((p_name, "<f4"))
            vertex_data = np.fromfile(f, dtype=np.dtype(dtype_map), count=num_vertices)
            return np.column_stack((vertex_data["x"], vertex_data["y"], vertex_data["z"])).astype(np.float64)
    except Exception as e:
        print(f"[MODAL-CALIB-ERROR] Failed to parse PLY: {e}")
        return None

def _validate_early_geometry(source_path: str, detected_n_views: int):
    """
    Gate 2: Early MASt3R point cloud density & camera path parallax validation.
    Aborts execution before the expensive 2000-iteration 3D Gaussian Splatting optimization
    if the scene geometry is invalid, empty, or lacks parallax (non-orbit).
    """
    import struct
    import numpy as np

    sparse_candidates = [
        os.path.join(source_path, f"sparse_{detected_n_views}", "0"),
        os.path.join(source_path, f"sparse_{detected_n_views}"),
        os.path.join(source_path, "sparse", "0"),
    ]
    sparse_dir = None
    for c in sparse_candidates:
        if os.path.isdir(c):
            sparse_dir = c
            break

    if not sparse_dir:
        raise RuntimeError("Early geometry validation failed: No sparse reconstruction folder found after MASt3R initialization.")

    # 1. Count points in points3D.bin or points3D.txt or points3D.ply
    points_bin = os.path.join(sparse_dir, "points3D.bin")
    points_txt = os.path.join(sparse_dir, "points3D.txt")
    points_ply = os.path.join(sparse_dir, "points3D.ply")
    
    num_points = 0
    if os.path.exists(points_bin):
        try:
            with open(points_bin, "rb") as f:
                num_points = struct.unpack("<Q", f.read(8))[0]
        except Exception:
            pass
    elif os.path.exists(points_txt):
        try:
            with open(points_txt, "r") as f:
                num_points = sum(1 for line in f if line.strip() and not line.startswith("#"))
        except Exception:
            pass
    elif os.path.exists(points_ply):
        pts = parse_ply_coords(points_ply)
        if pts is not None:
            num_points = len(pts)

    print(f"[GATE-2-CHECK] MASt3R triangulated point count: {num_points}")
    MIN_REQUIRED_POINTS = 250
    if num_points > 0 and num_points < MIN_REQUIRED_POINTS:
        raise RuntimeError(
            f"Early geometry validation failed: Only {num_points} 3D points were reconstructed from the video (minimum {MIN_REQUIRED_POINTS} required). "
            f"The video lacks sufficient texture or clear tree trunk features. Please rescan in good lighting."
        )

    # 2. Check camera parallax
    images_bin = os.path.join(sparse_dir, "images.bin")
    images_txt = os.path.join(sparse_dir, "images.txt")
    centers = None
    if os.path.exists(images_bin):
        centers = _read_colmap_camera_centers(images_bin)
    elif os.path.exists(images_txt):
        centers = _read_colmap_images_txt(images_txt)

    if centers is not None and len(centers) >= 2:
        diffs = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        path_length = float(np.sum(diffs))
        print(f"[GATE-2-CHECK] Camera path length: {path_length:.4f} units across {len(centers)} views")
        MIN_REQUIRED_PATH = 0.10
        if path_length < MIN_REQUIRED_PATH:
            raise RuntimeError(
                f"Early geometry validation failed: Insufficient camera movement/parallax detected (path length {path_length:.3f} < {MIN_REQUIRED_PATH}). "
                f"Please record a smooth orbit walking around the tree trunk instead of standing still."
            )

    print("[GATE-2-CHECK] [OK] Early geometry validation passed! Proceeding to 3D Gaussian Splatting optimization.")


def upload_to_r2(file_path: str, tree_code: str, custom_timestamp: int = None, is_thumbnail: bool = False, custom_filename: str = None) -> str:
    import os
    import time
    import boto3
    from botocore.config import Config
    
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("R2_BUCKET_NAME")
    
    if not all([account_id, access_key, secret_key, bucket_name]):
        print("[MODAL-R2-ERROR] Missing R2 configuration variables.")
        return ""
        
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto"
    )
    
    file_name = custom_filename if custom_filename is not None else os.path.basename(file_path)
    ts = custom_timestamp if custom_timestamp is not None else int(time.time())
    
    if is_thumbnail:
        object_key = f"thumbnails/{tree_code}/{ts}_{file_name}"
    else:
        object_key = f"tree_scans/{tree_code}/{ts}_{file_name}"
        
    content_type = "application/octet-stream"
    if file_name.endswith(".ply"):
        content_type = "application/x-ply"
    elif file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
        content_type = "image/jpeg"
        
    s3.upload_file(
        file_path,
        bucket_name,
        object_key,
        ExtraArgs={"ContentType": content_type}
    )
    
    public_url_prefix = os.environ.get("R2_PUBLIC_URL_PREFIX")
    if public_url_prefix:
        public_url_prefix = public_url_prefix.rstrip("/")
        return f"{public_url_prefix}/{object_key}"
        
    presigned_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": object_key},
        ExpiresIn=604800
    )
    return presigned_url

def _clean_ply_on_modal(ply_path: str) -> None:
    """Cleans background air floaters and ground noise directly on Modal using RANSAC + SOR."""
    import os
    import numpy as np
    from scipy.spatial import KDTree

    if not os.path.exists(ply_path):
        return

    with open(ply_path, "rb") as f:
        header_lines = []
        num_vertices = 0
        properties = []
        is_binary = False
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line.startswith("format binary_little_endian"):
                is_binary = True
            elif line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3:
                    properties.append((parts[1], parts[2]))
            elif line == "end_header":
                break

        if num_vertices < 20 or not is_binary:
            return

        dtype_map = []
        for p_type, p_name in properties:
            if p_type in ("float", "float32"):
                dtype_map.append((p_name, "<f4"))
            elif p_type in ("int", "int32", "uint"):
                dtype_map.append((p_name, "<i4"))
            elif p_type in ("uchar", "uint8"):
                dtype_map.append((p_name, "u1"))
            else:
                dtype_map.append((p_name, "<f4"))

        vertex_data = np.fromfile(f, dtype=np.dtype(dtype_map), count=num_vertices)

    pts = np.column_stack((vertex_data["x"], vertex_data["y"], vertex_data["z"]))

    # 1. RANSAC ground plane isolation
    sample_size = min(len(pts), 10000)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(pts), sample_size, replace=False)
    sample_pts = pts[sample_idx]

    max_iter = 100
    thresh = 0.06
    r_gen = np.random.default_rng(42)
    samples = r_gen.choice(sample_size, size=(max_iter, 3), replace=True)
    best_in = np.zeros(sample_size, dtype=bool)
    best_pl = None
    for s in samples:
        p1, p2, p3 = sample_pts[s[0]], sample_pts[s[1]], sample_pts[s[2]]
        n = np.cross(p2 - p1, p3 - p1)
        nl = np.linalg.norm(n)
        if nl < 1e-6:
            continue
        n = n / nl
        d = -np.dot(n, p1)
        inliers = np.abs(np.dot(sample_pts, n) + d) < thresh
        if np.sum(inliers) > np.sum(best_in):
            best_in = inliers
            best_pl = (n, d)

    if best_pl is not None:
        n_g, d_g = best_pl
        h_g = np.dot(sample_pts, n_g) + d_g
        if np.median(h_g) < 0:
            n_g, d_g = -n_g, -d_g
        fg_pts = sample_pts[(np.dot(sample_pts, n_g) + d_g) > 0.04]
    else:
        n_g = np.array([0.0, -1.0, 0.0])
        fg_pts = sample_pts

    if len(fg_pts) < 20:
        fg_pts = sample_pts

    ref = np.array([1.0, 0.0, 0.0]) if abs(n_g[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u1 = np.cross(n_g, ref)
    u1 = u1 / (np.linalg.norm(u1) + 1e-9)
    u2 = np.cross(n_g, u1)

    p_u1 = np.dot(fg_pts, u1)
    p_u2 = np.dot(fg_pts, u2)
    hist, xedges, yedges = np.histogram2d(p_u1, p_u2, bins=35)
    max_idx = np.unravel_index(np.argmax(hist), hist.shape)
    peak_u1 = 0.5 * (xedges[max_idx[0]] + xedges[max_idx[0] + 1])
    peak_u2 = 0.5 * (yedges[max_idx[1]] + yedges[max_idx[1] + 1])

    # 2. Crop to cylinder around primary trunk
    p_u1_all = np.dot(pts, u1)
    p_u2_all = np.dot(pts, u2)
    dist_sq = (p_u1_all - peak_u1) ** 2 + (p_u2_all - peak_u2) ** 2
    CROP_RADIUS = 0.85
    crop_mask = dist_sq <= (CROP_RADIUS ** 2)

    filtered_vertex_data = vertex_data[crop_mask]
    filtered_xyz = pts[crop_mask]

    # 3. Statistical Outlier Removal (SOR)
    if len(filtered_xyz) >= 20:
        tree = KDTree(filtered_xyz)
        dists, _ = tree.query(filtered_xyz, k=21, workers=-1)
        mean_dists = dists[:, 1:].mean(axis=1)
        g_mean = mean_dists.mean()
        g_std = mean_dists.std()
        inlier_mask = mean_dists <= (g_mean + 2.0 * g_std)
        filtered_vertex_data = filtered_vertex_data[inlier_mask]

    # 4. Overwrite PLY in place
    n_out = len(filtered_vertex_data)
    h_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n_out}",
    ]
    for p_type, p_name in properties:
        h_lines.append(f"property {p_type} {p_name}")
    h_lines.append("end_header\n")
    header_bytes = "\n".join(h_lines).encode("ascii")

    with open(ply_path, "wb") as f_out:
        f_out.write(header_bytes)
        f_out.write(filtered_vertex_data.tobytes())

@app.function(
    gpu=GPU_CONFIG,
    timeout=1800,  # 30 minutes
    image=image
)
def run_reconstruction(images_bytes: list[bytes] = None, tree_code: str = "Unknown", remove_background: bool = False, r2_config: dict = None, iterations: int = 2000, camera_poses: list = None, r2_frames_prefix: str = None) -> dict:
    import os
    import time
    import shutil
    import subprocess
    import glob
    from concurrent.futures import ThreadPoolExecutor
    
    if r2_config:
        for k, v in r2_config.items():
            if v:
                os.environ[k] = str(v)
    
    # 1. Locate the repository directory
    repo_path = "/workspace/InstantSplat"
    if not os.path.exists(os.path.join(repo_path, "init_geo.py")):
        raise Exception(f"Could not find InstantSplat repository at {repo_path}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found InstantSplat at: {repo_path}")
    os.chdir(repo_path)
    
    # 2. Prepare the input directory
    # InstantSplat expects images to be in assets/examples/<scene_name>/images
    scene_name = "test_scene"
    input_dir = os.path.join(repo_path, "assets", "examples", scene_name, "images")
    output_dir = os.path.join(repo_path, "output_infer", scene_name)
    
    # Clean previous input/output directories to ensure a clean run
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir, exist_ok=True)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Direct Modal-to-R2 frame loading (eliminates 50-70MB outbound bandwidth from Render)
    if r2_frames_prefix and r2_config:
        import boto3
        from botocore.config import Config as BotoConfig
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{r2_config['CLOUDFLARE_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=r2_config["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=r2_config["R2_SECRET_ACCESS_KEY"],
            config=BotoConfig(signature_version="s3v4"),
            region_name="auto",
        )
        bucket = r2_config["R2_BUCKET_NAME"]
        res = s3.list_objects_v2(Bucket=bucket, Prefix=r2_frames_prefix)
        frame_keys = sorted([obj["Key"] for obj in res.get("Contents", []) if obj["Key"].lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found {len(frame_keys)} frames in R2 prefix '{r2_frames_prefix}'")
        if not frame_keys:
            raise ValueError(f"No frames found in R2 prefix '{r2_frames_prefix}'! Extraction upload may have failed.")
            
        def dl_frame(args):
            idx, key = args
            dest = os.path.join(input_dir, f"{idx:04d}.jpg")
            s3.download_file(bucket, key, dest)
            return dest
            
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(dl_frame, enumerate(frame_keys)))
            
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Downloaded {len(frame_keys)} frames to {input_dir} directly from R2!")
    elif images_bytes:
        # Legacy fallback: images sent over RPC
        for i, img_bytes in enumerate(images_bytes):
            img_path = os.path.join(input_dir, f"{i:03d}.jpg")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved {len(images_bytes)} images from RPC to {input_dir}")
    else:
        raise ValueError("Neither r2_frames_prefix nor images_bytes provided to run_reconstruction")
    
    # ── Background removal on Modal ──
    if remove_background:
        try:
            from rembg import remove, new_session
            from PIL import Image
            import io
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initialising rembg session (u2net)...")
            bg_session = new_session("u2net")
            frame_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running background removal using rembg on {len(frame_files)} frames...")
            for idx, f_path in enumerate(frame_files):
                input_img = Image.open(f_path)
                output_img = remove(input_img, session=bg_session)
                if output_img.mode == "RGBA":
                    background = Image.new("RGBA", output_img.size, (0, 0, 0, 255))
                    composited = Image.alpha_composite(background, output_img).convert("RGB")
                else:
                    composited = output_img.convert("RGB")
                composited.save(f_path, format="JPEG", quality=95)
                if (idx + 1) % 5 == 0:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]   rembg: processed {idx + 1}/{len(frame_files)} frames")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Background removal complete on Modal.")
        except Exception as bg_err:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Background removal failed on Modal: {bg_err}")
    
    # ── Deterministic seed: controls PYTHONHASHSEED, CUBLAS, PyTorch, NumPy, etc. ──
    # Fix 3D alignment inconsistency: same input images → same output every time.
    # References: https://pytorch.org/docs/stable/notes/randomness.html
    FIXED_SEED = 42
    deterministic_env = {
        **os.environ,
        # Python built-in hash randomisation
        "PYTHONHASHSEED": str(FIXED_SEED),
        # CUDA deterministic algorithms (allocates some workspace; acceptable overhead)
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        # PyTorch Lightning / generic seed env picked up by many frameworks
        "PL_GLOBAL_SEED": str(FIXED_SEED),
        # Force single-threaded OpenBLAS to avoid non-deterministic threading order
        "OPENBLAS_NUM_THREADS": "1",
    }

    # Helper to run a shell command with real-time log printing
    def run_command(cmd_args, stage_name):
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting {stage_name} ---")
        start = time.time()
        
        # Locate conda or environment's python executable directly
        has_conda = False
        conda_exec = "conda"
        python_exec = "python3"
        
        # Try to locate instantsplat python directly
        for p_path in [
            "/opt/conda/bin/python3",
            "/opt/conda/envs/instantsplat/bin/python3",
            "/root/miniconda3/envs/instantsplat/bin/python3",
            "/opt/miniconda/envs/instantsplat/bin/python3",
            "/miniconda/envs/instantsplat/bin/python3",
        ]:
            if os.path.exists(p_path):
                python_exec = p_path
                has_conda = True
                break
                
        if has_conda:
            if cmd_args[0] in ("python3", "python"):
                full_cmd = [python_exec] + cmd_args[1:]
            else:
                full_cmd = cmd_args
        else:
            # Fallback to finding conda and running via conda run
            for c_path in ["conda", "/opt/conda/bin/conda", "/opt/miniconda/bin/conda", "/root/miniconda3/bin/conda", "/miniconda/bin/conda"]:
                try:
                    res = subprocess.run([c_path, "--version"], capture_output=True)
                    if res.returncode == 0:
                        conda_exec = c_path
                        break
                except Exception:
                    continue
            try:
                res = subprocess.run([conda_exec, "info", "--envs"], capture_output=True, text=True)
                if "instantsplat" in res.stdout:
                    full_cmd = [conda_exec, "run", "-n", "instantsplat"] + cmd_args
                else:
                    full_cmd = cmd_args
            except Exception:
                full_cmd = cmd_args
            
        print(f"Running command: {' '.join(full_cmd)}")
        
        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=deterministic_env,  # <-- inject deterministic seed environment
        )
        
        # Stream output in real time
        log_lines = []
        for line in process.stdout:
            print(line, end="")
            log_lines.append(line)
            
        process.wait()
        duration = time.time() - start
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Finished {stage_name} in {duration:.2f} seconds ---")
        
        if process.returncode != 0:
            full_logs = "".join(log_lines)
            if "out of memory" in full_logs.lower() or "oom" in full_logs.lower():
                raise RuntimeError(
                    f"CUDA Out of Memory (OOM) error occurred during {stage_name}! "
                    f"Please edit the GPU_CONFIG parameter in modal_app.py to 'a100' or 'h100'."
                )
            else:
                tail = "".join(log_lines[-50:])
                raise RuntimeError(
                    f"Command failed during {stage_name} with exit code {process.returncode}.\n"
                    f"--- Last output ---\n{tail}"
                )
                
    # 3. Stage 1: Geometric Initialization (init_geo.py)
    progress_dict[tree_code] = "Initializing geometry (MASt3R)"
    # Run WITHOUT --n_views — let init_geo pick its default number of views.
    # It will create a sparse_{N}/ folder inside source_path which we detect next.
    #
    # Determinism strategy:
    #   - Env vars (PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG) injected via run_command.
    #   - Try passing --seed if init_geo.py supports it; silently fall back if not.
    #   - The seed wrapper script below seeds Python / NumPy / PyTorch before the
    #     actual script runs inside the same process via exec, so it works regardless
    #     of whether init_geo.py exposes its own --seed flag.
    seed_wrapper_path = os.path.join(repo_path, "_seed_wrapper.py")
    with open(seed_wrapper_path, "w") as _sw:
        _sw.write(
            f"import random, os, sys\n"
            f"import numpy as np\n"
            f"random.seed({FIXED_SEED})\n"
            f"np.random.seed({FIXED_SEED})\n"
            f"try:\n"
            f"    import torch\n"
            f"    torch.manual_seed({FIXED_SEED})\n"
            f"    torch.cuda.manual_seed_all({FIXED_SEED})\n"
            f"    torch.backends.cudnn.deterministic = True\n"
            f"    torch.backends.cudnn.benchmark = False\n"
            f"    torch.use_deterministic_algorithms(True, warn_only=True)\n"
            f"except Exception:\n"
            f"    pass\n"
            f"# Run the actual target script with its original __file__ so that\n"
            f"# any os.path.dirname(__file__) calls inside the target resolve correctly.\n"
            f"script = os.path.abspath(sys.argv[1])\n"
            f"sys.argv = sys.argv[1:]\n"
            f"with open(script) as _f:\n"
            f"    exec(compile(_f.read(), script, 'exec'), {{\"__name__\": \"__main__\", \"__file__\": script}})\n"
        )
    print(f"[SEED] Written seed wrapper → {seed_wrapper_path} (seed={FIXED_SEED})")

    init_cmd = [
        "python3", seed_wrapper_path, "init_geo.py",
        "--source_path", os.path.join(repo_path, "assets", "examples", scene_name),
        "--model_path", output_dir,
        "--niter", "300"
    ]
    run_command(init_cmd, "Geometric Initialization (init_geo.py)")

    # Auto-detect which sparse_{N} folder init_geo.py actually created
    # so we can pass the correct --n_views to train.py
    source_path = os.path.join(repo_path, "assets", "examples", scene_name)
    detected_n_views = None
    for entry in os.listdir(source_path):
        if entry.startswith("sparse_") and os.path.isdir(os.path.join(source_path, entry)):
            try:
                candidate = int(entry.split("_")[1])
                if detected_n_views is None or candidate > detected_n_views:
                    detected_n_views = candidate
            except ValueError:
                pass
    if detected_n_views is None:
        raise RuntimeError("Could not find any sparse_N folder created by init_geo.py")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] init_geo created sparse_{detected_n_views}/ — using n_views={detected_n_views} for train.py")

    # Gate 2: Early MASt3R point cloud density & camera path parallax validation
    progress_dict[tree_code] = "Validating 3D geometry"
    _validate_early_geometry(source_path, detected_n_views)

    # 4. Stage 2: Fast 3D-Gaussian Optimization (train.py)
    progress_dict[tree_code] = "Training Gaussians"
    train_cmd = [
        "python3", seed_wrapper_path, "train.py",
        "--source_path", source_path,
        "--model_path", output_dir,
        "--iterations", str(iterations),
        "--n_views", str(detected_n_views),
        "--optim_pose",
        "--test_iterations", str(iterations)
    ]
    run_command(train_cmd, f"Fast 3D-Gaussian Optimization (train.py) with {iterations} iterations")
    
    # 5. Exporting results
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Locating output file ---")
    
    # Always list ALL generated files so we can see the full structure
    all_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            rel = os.path.relpath(os.path.join(root, file), output_dir)
            size = os.path.getsize(os.path.join(root, file))
            all_files.append((rel, size))
    print(f"All files in output directory ({len(all_files)} total):")
    for rel, size in sorted(all_files):
        print(f"  {size:>12,} bytes  {rel}")
    
    # Prefer the Gaussian Splat PLY produced by train.py — it is named point_cloud.ply
    # and lives somewhere under point_cloud/iteration_*/
    # It has opacity, scale, rotation, f_dc properties needed by the Gaussian Splat viewer.
    gaussian_ply_files = sorted(glob.glob(
        os.path.join(output_dir, "**/point_cloud.ply"),
        recursive=True
    ), reverse=True)  # sort descending so highest iteration comes first
    
    # result.ply from init_geo is just a plain xyz+rgb point cloud — skip it
    gaussian_ply_files = [f for f in gaussian_ply_files if "result.ply" not in f]
    
    output_file_path = None
    if gaussian_ply_files:
        output_file_path = gaussian_ply_files[0]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using Gaussian Splat PLY: {output_file_path}")
    else:
        # Fallback: any .splat or .ply (excluding the plain result.ply from init_geo)
        ply_files = [f for f in glob.glob(os.path.join(output_dir, "**/*.ply"), recursive=True)
                     if "result.ply" not in f]
        splat_files = glob.glob(os.path.join(output_dir, "**/*.splat"), recursive=True)
        if splat_files:
            output_file_path = splat_files[0]
        elif ply_files:
            output_file_path = ply_files[0]
        else:
            # Last resort: grab result.ply even though it's a plain point cloud
            all_ply = glob.glob(os.path.join(output_dir, "**/*.ply"), recursive=True)
            if all_ply:
                output_file_path = all_ply[0]
        
    if not output_file_path or not os.path.exists(output_file_path):
        raise FileNotFoundError(f"Could not find any output file in: {output_dir}")
        
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sending: {output_file_path} ({os.path.getsize(output_file_path):,} bytes)")
    
    # 6. Post-processing: remove outlier / floater Gaussians before returning
    progress_dict[tree_code] = "Extracting point cloud"
    # This runs on the Modal GPU machine which has scipy (numpy is always available).
    # Parameters are deliberately conservative to avoid over-pruning valid splats.
    try:
        import numpy as np
        from scipy.spatial import KDTree

        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting post-processing outlier removal ---")

        # Read full Gaussian PLY as structured numpy array
        with open(output_file_path, "rb") as pf:
            raw_props = []
            num_vertices = 0
            while True:
                hline = pf.readline().decode("ascii", errors="ignore").strip()
                if hline.startswith("element vertex"):
                    num_vertices = int(hline.split()[-1])
                elif hline.startswith("property"):
                    hp = hline.split()
                    if len(hp) >= 3:
                        raw_props.append((hp[1], hp[2]))
                elif hline == "end_header":
                    break
            dtype_map = []
            for p_type, p_name in raw_props:
                if p_type in ("float", "float32"):   dtype_map.append((p_name, "<f4"))
                elif p_type in ("int", "int32", "uint"): dtype_map.append((p_name, "<i4"))
                elif p_type in ("uchar", "uint8"):   dtype_map.append((p_name, "u1"))
                else:                                 dtype_map.append((p_name, "<f4"))
            vertex_data = np.fromfile(pf, dtype=np.dtype(dtype_map), count=num_vertices)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loaded {num_vertices:,} Gaussians for outlier removal")

        # Combined inlier mask — start all True
        combined_mask = np.ones(num_vertices, dtype=bool)

        # Pass 1: Horizontal crop around the trunk cluster peak
        # Reduces point count by removing distant background/clutter, which makes KDTree processing extremely fast.
        # Find peak using only the lower 35% of points to avoid canopy/branch bias.
        xyz = np.column_stack((vertex_data["x"], vertex_data["y"], vertex_data["z"])).astype(np.float64)
        # Force Y as the vertical axis (axis index 1)
        rough_axis_idx = 1
        proj_axes = [0, 2]

        # Stage 1: Rough Horizontal Peak using entire cloud
        h1_all = xyz[:, proj_axes[0]]
        h2_all = xyz[:, proj_axes[1]]
        hist, xedges, yedges = np.histogram2d(h1_all, h2_all, bins=30)
        max_idx = np.unravel_index(np.argmax(hist), hist.shape)
        rough_peak_h1 = 0.5 * (xedges[max_idx[0]] + xedges[max_idx[0] + 1])
        rough_peak_h2 = 0.5 * (yedges[max_idx[1]] + yedges[max_idx[1] + 1])

        # Rough crop to remove background (2.2 meters radius in real world)
        ROUGH_CROP_RADIUS = 2.2
        dist_sq_rough = (xyz[:, proj_axes[0]] - rough_peak_h1)**2 + (xyz[:, proj_axes[1]] - rough_peak_h2)**2
        rough_cropped = xyz[dist_sq_rough <= ROUGH_CROP_RADIUS**2]
        if len(rough_cropped) < 100:
            rough_cropped = xyz

        # Stage 2: Refined Peak using lower 35% of rough cropped points (Y-down)
        rough_y = rough_cropped[:, rough_axis_idx]
        y_min = np.percentile(rough_y, 1)
        y_max = np.percentile(rough_y, 99)
        y_height = y_max - y_min

        # Lower 35% height in Y-down convention: Y is close to y_max
        lower_mask = rough_y >= (y_max - y_height * 0.35)
        lower_xyz = rough_cropped[lower_mask]
        if len(lower_xyz) < 100:
            lower_xyz = rough_cropped

        h1 = lower_xyz[:, proj_axes[0]]
        h2 = lower_xyz[:, proj_axes[1]]

        hist_ref, xedges_ref, yedges_ref = np.histogram2d(h1, h2, bins=30)
        max_idx_ref = np.unravel_index(np.argmax(hist_ref), hist_ref.shape)
        peak_h1 = 0.5 * (xedges_ref[max_idx_ref[0]] + xedges_ref[max_idx_ref[0] + 1])
        peak_h2 = 0.5 * (yedges_ref[max_idx_ref[1]] + yedges_ref[max_idx_ref[1] + 1])

        dist_sq = (xyz[:, proj_axes[0]] - peak_h1)**2 + (xyz[:, proj_axes[1]] - peak_h2)**2
        CROP_RADIUS = 2.2
        crop_mask = dist_sq <= CROP_RADIUS**2
        
        # If crop yields too few points, fallback to keeping all points
        if crop_mask.sum() < 1000:
            crop_mask = np.ones(num_vertices, dtype=bool)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Crop mask resulted in too few points (< 1000), skipping crop.")
        else:
            combined_mask &= crop_mask
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pass 1 (horizontal crop): removed {num_vertices - combined_mask.sum():,} points | remaining {combined_mask.sum():,}")

        # Pass 1.5: Spatial Statistical Outlier Removal (KNN-based KDTree)
        # Using std_ratio=1.5 to aggressively prune floating noise.
        # Running on cropped subset first ensures KDTree completes in <10 seconds.
        n_spatial = 0
        if combined_mask.sum() >= 20:
            cropped_xyz = xyz[combined_mask]
            tree = KDTree(cropped_xyz)
            nb_neighbors = 20
            std_ratio = 1.5
            distances, _ = tree.query(cropped_xyz, k=nb_neighbors + 1, workers=-1)
            mean_dists = distances[:, 1:].mean(axis=1)

            global_mean = mean_dists.mean()
            global_std  = mean_dists.std()
            threshold   = global_mean + std_ratio * global_std

            spatial_inliers = mean_dists <= threshold
            # Map the inlier mask back to the global index
            spatial_mask = np.zeros(num_vertices, dtype=bool)
            spatial_mask[combined_mask] = spatial_inliers
            
            combined_mask &= spatial_mask
            n_spatial = int((~spatial_inliers).sum())
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pass 1.5 (spatial KNN): removed {n_spatial:,} | remaining {combined_mask.sum():,}")

        # Pass 2: Low-opacity removal
        # sigmoid(-2.2) ≈ 0.10 — removes splats with <10% effective opacity.
        # The previous threshold of -4.0 (1.8%) left semi-transparent "smoke"
        # Gaussians intact. Raising to -2.2 eliminates the fog/smoke artifact
        # while preserving dense trunk/branch structure (which has opacity > 0.5).
        MIN_OPACITY_LOGIT = -2.2
        if "opacity" in vertex_data.dtype.names:
            opacity_mask = vertex_data["opacity"] >= MIN_OPACITY_LOGIT
            combined_mask &= opacity_mask
            n_opacity = int((~opacity_mask).sum())
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pass 2 (opacity): removed {n_opacity:,} | remaining {combined_mask.sum():,}")

        # Pass 3: Oversized Gaussian removal (log-scale filter)
        # exp(-1.5) ≈ 0.22 units — tighter than the previous -1.0 (0.37 units).
        # Large transparent Gaussians are the primary cause of the smoke effect:
        # they cover large areas with little colour contribution and look like fog.
        MAX_LOG_SCALE = -1.5
        scale_names = [n for n in vertex_data.dtype.names if n.startswith("scale_")]
        if scale_names:
            scales = np.column_stack([vertex_data[n] for n in scale_names])
            max_scales = scales.max(axis=1)
            scale_mask = max_scales <= MAX_LOG_SCALE
            combined_mask &= scale_mask
            n_scale = int((~scale_mask).sum())
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pass 3 (scale):   removed {n_scale:,} | remaining {combined_mask.sum():,}")

        # Pass 3.5: Remove "smoke pattern" — low-opacity AND medium-large splats.
        # Smoke Gaussians: opacity logit in (-3.0, -1.5) AND max_scale > -2.5.
        # Even after Pass 2+3, some medium-size translucent splats slip through.
        if "opacity" in vertex_data.dtype.names and scale_names:
            smoke_opacity_mask = vertex_data["opacity"] < -1.5   # sigmoid < 18%
            smoke_scale_mask   = np.column_stack([vertex_data[n] for n in scale_names]).max(axis=1) > -2.5
            smoke_combined     = smoke_opacity_mask & smoke_scale_mask
            combined_mask &= ~smoke_combined
            n_smoke = int(smoke_combined.sum())
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pass 3.5 (smoke): removed {n_smoke:,} translucent-large splats | remaining {combined_mask.sum():,}")

        # Pass 3.6: Remove spiky needles (extreme aspect ratio scale outliers)
        # In log-space, scale_0, scale_1, scale_2 are the log-scales.
        # An extreme aspect ratio (needle-like) splat has max(scales) - min(scales) > 4.5 (approx 90x ratio).
        MAX_SCALE_DIFF = 4.5
        if scale_names:
            scales = np.column_stack([vertex_data[n] for n in scale_names])
            scale_diff = scales.max(axis=1) - scales.min(axis=1)
            spiky_mask = scale_diff <= MAX_SCALE_DIFF
            combined_mask &= spiky_mask
            n_spiky = int((~spiky_mask).sum())
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pass 3.6 (spiky):  removed {n_spiky:,} spiky needles | remaining {combined_mask.sum():,}")


        n_kept    = int(combined_mask.sum())
        n_removed = num_vertices - n_kept
        pct       = n_removed / num_vertices * 100
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Outlier removal complete: {n_removed:,} removed ({pct:.1f}%) | {n_kept:,} Gaussians kept")

        # Write filtered PLY back to same file
        filtered = vertex_data[combined_mask].copy()  # make writable

        # Pass 4: Scale inflation — enlarge remaining Gaussians to fill visual gaps.
        # In log-space, adding a constant = multiplying actual size by exp(constant).
        # +0.35 → each Gaussian becomes ~42% larger in every dimension.
        # Cap at -0.5 (exp(-0.5) ≈ 0.61 units) so we don't create new large floaters.
        INFLATE_AMOUNT = 0.35
        INFLATE_CAP    = -0.5
        scale_cols = [n for n in filtered.dtype.names if n.startswith("scale_")]
        if scale_cols:
            for sc in scale_cols:
                filtered[sc] = np.minimum(
                    filtered[sc] + INFLATE_AMOUNT, INFLATE_CAP
                ).astype(filtered[sc].dtype)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pass 4 (inflate):  scales +{INFLATE_AMOUNT} log-units (cap {INFLATE_CAP}) on {len(scale_cols)} channels | {n_kept:,} Gaussians")
        header_lines = ["ply", "format binary_little_endian 1.0", f"element vertex {n_kept}"]
        for p_type, p_name in raw_props:
            header_lines.append(f"property {p_type} {p_name}")
        header_lines.append("end_header")
        header = "\n".join(header_lines) + "\n"
        with open(output_file_path, "wb") as wf:
            wf.write(header.encode("ascii"))
            filtered.tofile(wf)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Post-processing complete ---\n")

    except Exception as cleanup_err:
        # Non-fatal: if cleanup fails, we still return the unfiltered PLY
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: outlier removal failed ({cleanup_err}), returning unfiltered PLY")

    # Convert PLY to KSPLAT
    ksplat_path = output_file_path.replace(".ply", ".ksplat")
    splat_data = b""
    try:
        conv_cmd = [
            "node", "/workspace/GaussianSplats3D/util/create-ksplat.js",
            output_file_path, ksplat_path,
            "1", "1"
        ]
        conv_res = subprocess.run(conv_cmd, capture_output=True, text=True)
        if conv_res.returncode == 0:
            with open(ksplat_path, "rb") as f:
                splat_data = f.read()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] KSPLAT conversion successful: {len(splat_data):,} bytes")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] KSPLAT conversion failed: {conv_res.stderr}. Returning raw PLY instead.")
            with open(output_file_path, "rb") as f:
                splat_data = f.read()
    except Exception as conv_err:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] KSPLAT conversion exception: {conv_err}. Returning raw PLY instead.")
        with open(output_file_path, "rb") as f:
            splat_data = f.read()

    # Locate the MASt3R plain xyz+rgb point cloud produced by init_geo.py.
    # init_geo may save it as "result.ply" in output_dir or source_path,
    # or as "points3d.ply" inside sparse_N folders.
    points3d_data = None
    
    # Robust case-insensitive search across relevant directories
    search_dirs = [output_dir, source_path, repo_path]
    target_names_lower = {"points3d.ply", "result.ply"}
    
    raw_candidates = []
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            for root, _, files in os.walk(s_dir):
                for file in files:
                    if file.lower() in target_names_lower:
                        raw_candidates.append(os.path.join(root, file))
                        
    # Remove duplicates
    seen = set()
    mast3r_candidates = [c for c in raw_candidates if not (c in seen or seen.add(c))]
    
    # Exclude the gaussian splat PLY we already selected (it's a training artifact, not a plain point cloud)
    if output_file_path:
        mast3r_candidates = [c for c in mast3r_candidates if os.path.abspath(c) != os.path.abspath(output_file_path)]
        
    # Prioritize candidate files: 
    # 1. contains "sparse_" and filename is case-insensitively "points3d.ply"
    # 2. filename is case-insensitively "points3d.ply"
    # 3. filename is case-insensitively "result.ply"
    def candidate_priority(path):
        fname = os.path.basename(path).lower()
        has_sparse = "sparse_" in path
        if fname == "points3d.ply":
            return (0 if has_sparse else 1)
        return 2

    mast3r_candidates.sort(key=candidate_priority)
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Searching for MASt3R point cloud in {len(mast3r_candidates)} case-insensitive candidates...")
    for candidate in mast3r_candidates:
        if os.path.exists(candidate):
            fsize = os.path.getsize(candidate)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking candidate: {candidate} ({fsize:,} bytes)")
            if fsize < 1024:  # skip empty/tiny files
                continue
            with open(candidate, "rb") as f:
                points3d_data = f.read()
            size_kb = len(points3d_data) / 1024
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found MASt3R point cloud: {candidate} ({size_kb:.1f} KB)")
            break

    if points3d_data is None:
        searched_paths = [os.path.abspath(d) for d in search_dirs]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: points3D.ply not found in expected paths: {searched_paths}")

    # Search and load points3D_all.npy
    points3d_all_data = None
    target_npy_names_lower = {"points3d_all.npy"}
    npy_candidates = []
    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            for root, _, files in os.walk(s_dir):
                for file in files:
                    if file.lower() in target_npy_names_lower:
                        npy_candidates.append(os.path.join(root, file))

    for candidate in npy_candidates:
        if os.path.exists(candidate):
            fsize = os.path.getsize(candidate)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking NPY candidate: {candidate} ({fsize:,} bytes)")
            if fsize < 1024:
                continue
            with open(candidate, "rb") as f:
                points3d_all_data = f.read()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found dense pointmap: {candidate} ({len(points3d_all_data)/1024/1024:.1f} MB)")
            break

    # 7. Scale calibration from Modal (VIO camera path or MASt3R geometric prior)
    scale_calibration = None
    if camera_poses and len(camera_poses) >= 2:
        try:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Deriving VIO scale factor from camera poses ({len(camera_poses)} entries)...")
            scale_calibration = _derive_scale_from_vio_poses(camera_poses, source_path, detected_n_views)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] VIO scale result: {scale_calibration}")
        except Exception as vio_err:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] VIO scale derivation failed: {vio_err}")

    if not scale_calibration or not scale_calibration.get("is_calibrated"):
        try:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Deriving geometric scale prior from MASt3R init_geo output...")
            scale_calibration = _derive_mast3r_scale_prior(source_path, detected_n_views)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Geometric scale prior result: {scale_calibration}")
        except Exception as cal_err:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Geometric scale prior failed: {cal_err}")
            scale_calibration = {
                "is_calibrated": False,
                "source": "uncalibrated",
                "scale_factor": 1.0,
                "reason": f"Exception during geometric scale derivation: {cal_err}",
            }

    # 8. Upload files directly to R2 if config provided
    splat_url = ""
    points3d_url = ""
    points3d_all_url = ""
    thumbnail_url = ""
    ts = int(time.time())
    
    if r2_config:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Uploading files directly to Cloudflare R2 from Modal...")
        try:
            if output_file_path and os.path.exists(output_file_path):
                # Convert PLY to KSPLAT on Modal
                ksplat_path = output_file_path.replace(".ply", ".ksplat")
                try:
                    import subprocess
                    conv_cmd = [
                        "node", "/workspace/GaussianSplats3D/util/create-ksplat.js",
                        output_file_path, ksplat_path,
                        "1", "1"  # compression level = 1, alpha removal threshold = 1
                    ]
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Converting splat PLY to KSPLAT: {' '.join(conv_cmd)}")
                    conv_res = subprocess.run(conv_cmd, capture_output=True, text=True)
                    if conv_res.returncode == 0:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] KSPLAT conversion successful: {os.path.getsize(ksplat_path):,} bytes")
                        splat_url = upload_to_r2(ksplat_path, tree_code, custom_timestamp=ts, custom_filename="result.ksplat")
                    else:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] KSPLAT conversion failed: {conv_res.stderr}. Uploading raw PLY instead.")
                        splat_url = upload_to_r2(output_file_path, tree_code, custom_timestamp=ts, custom_filename="result.ply")
                except Exception as conv_err:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] KSPLAT conversion exception: {conv_err}. Uploading raw PLY instead.")
                    splat_url = upload_to_r2(output_file_path, tree_code, custom_timestamp=ts, custom_filename="result.ply")
            if len(mast3r_candidates) > 0 and os.path.exists(mast3r_candidates[0]):
                try:
                    _clean_ply_on_modal(mast3r_candidates[0])
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cleaned floaters from MASt3R point cloud before R2 upload")
                except Exception as clean_err:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Modal PLY floater cleaning exception: {clean_err}")
                points3d_url = upload_to_r2(mast3r_candidates[0], tree_code, custom_timestamp=ts, custom_filename="points3d.ply")
            npy_path = None
            for candidate in npy_candidates:
                if os.path.exists(candidate) and os.path.getsize(candidate) >= 1024:
                    npy_path = candidate
                    break
            if npy_path:
                points3d_all_url = upload_to_r2(npy_path, tree_code, custom_timestamp=ts, custom_filename="points3D_all.npy")
            try:
                frame_files = sorted([
                    os.path.join(input_dir, f) for f in os.listdir(input_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])
                target_thumb_idx = 0
                if npy_path and os.path.exists(npy_path):
                    try:
                        pts3d_loaded = np.load(npy_path)
                        N_f = pts3d_loaded.shape[0]
                        valid_counts = [
                            np.sum(~np.all(pts3d_loaded[i] == 0, axis=-1) & ~np.any(np.isnan(pts3d_loaded[i]), axis=-1))
                            for i in range(N_f)
                        ]
                        target_thumb_idx = int(np.argmax(valid_counts))
                    except Exception as count_err:
                        print(f"[MODAL-R2] Could not calculate argmax valid counts: {count_err}")
                        target_thumb_idx = 0

                if frame_files and target_thumb_idx < len(frame_files):
                    thumbnail_url = upload_to_r2(frame_files[target_thumb_idx], tree_code, custom_timestamp=ts, is_thumbnail=True)
                    print(f"[MODAL-R2] Uploaded thumbnail frame {target_thumb_idx}/{len(frame_files)} matching primary pointmap.")
                elif frame_files:
                    thumbnail_url = upload_to_r2(frame_files[0], tree_code, custom_timestamp=ts, is_thumbnail=True)
            except Exception as thumb_err:
                print(f"[MODAL-R2-ERROR] Thumbnail upload failed: {thumb_err}")
        except Exception as r2_err:
            print(f"[MODAL-R2-ERROR] Direct R2 uploads failed: {r2_err}")

        # Write completion marker to Modal Dict so the server can recover even
        # if its fn.remote() connection was dropped (e.g. Render restart mid-job).
        # Key: "{tree_code}_complete", TTL ~1 hour (server reads it once then deletes it).
        try:
            completion_payload = {
                "splat_url": splat_url,
                "points3d_url": points3d_url,
                "points3d_all_url": points3d_all_url,
                "thumbnail_url": thumbnail_url,
                "timestamp": ts,
                "scale_calibration": scale_calibration,
            }
            progress_dict[f"{tree_code}_complete"] = completion_payload
            print(f"[MODAL-COMPLETE] Wrote completion marker for {tree_code} to Modal Dict.")
        except Exception as dict_err:
            print(f"[MODAL-COMPLETE] Failed to write completion marker: {dict_err}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Export completed successfully ---")
    return {
        "uploaded": True if r2_config else False,
        "splat": splat_data if not r2_config else b"",
        "points3d": points3d_data if not r2_config else b"",
        "points3d_all": points3d_all_data if not r2_config else b"",
        "scale_calibration": scale_calibration,
        "splat_url": splat_url,
        "points3d_url": points3d_url,
        "points3d_all_url": points3d_all_url,
        "thumbnail_url": thumbnail_url,
        "timestamp": ts,
    }


@app.function(
    image=image,
    timeout=300,
    cpu=2.0
)
def extract_video_frames_modal(
    video_bytes: bytes,
    target: int,
    blur_thresh: int,
    t_server_before_call: float = None,
    r2_key: str = None,
    r2_config: dict = None,
    tree_code: str = None,
) -> dict:
    """Extract sharp, well-overlapping frames from a video.

    Two modes:
    - Legacy (video_bytes not None): receives raw bytes serialised through Modal call.
    - New (r2_key + r2_config): downloads directly from R2, skips the serialisation overhead.

    After downloading, 4K (>1080p height) footage is pre-downscaled to 1080p via ffmpeg
    before the CV2 frame-scoring loop, cutting compute time ~3-4x.
    """
    import time
    t_modal_enter = time.time()
    import os
    import cv2
    import numpy as np
    import shutil
    import tempfile
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "input_video.mp4")

    # --- Step 1: Obtain video from R2 or legacy bytes ---
    if r2_key and r2_config:
        import boto3
        from botocore.config import Config as BotoConfig
        t_dl_start = time.time()
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{r2_config['CLOUDFLARE_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=r2_config["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=r2_config["R2_SECRET_ACCESS_KEY"],
            config=BotoConfig(signature_version="s3v4"),
            region_name="auto",
        )
        s3.download_file(r2_config["R2_BUCKET_NAME"], r2_key, video_path)
        t_dl_end = time.time()
        file_mb = os.path.getsize(video_path) / 1024 / 1024
        print(f"[TIMING] R2 download to Modal: {t_dl_end - t_dl_start:.4f}s for {file_mb:.2f} MB")
    else:
        # Legacy: bytes passed directly through Modal serialisation
        with open(video_path, "wb") as f:
            f.write(video_bytes)

    # --- Step 2: Pre-downscale 4K -> 1080p with ffmpeg (Fix 2) ---
    # This reduces CV2 decode cost ~4x for 3840x2160 inputs at the cost of ~5s ffmpeg step.
    # 1080p is more than sufficient for MASt3R (internally works at 512px).
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height", "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            capture_output=True, text=True, timeout=30
        )
        native_h = int(probe.stdout.strip()) if probe.stdout.strip().isdigit() else 0
    except Exception:
        native_h = 0

    if native_h > 1080:
        downscaled_path = os.path.join(temp_dir, "input_1080p.mp4")
        t_ff_start = time.time()
        try:
            res = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", "scale=-2:1080",          # scale height to 1080, maintain AR
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "18",                    # near-lossless quality
                    "-an",                           # drop audio (not needed)
                    downscaled_path,
                ],
                capture_output=True, text=True, timeout=120
            )
            if res.returncode == 0:
                t_ff_end = time.time()
                print(f"[TIMING] ffmpeg downscale {native_h}p -> 1080p: {t_ff_end - t_ff_start:.4f}s")
                video_path = downscaled_path
            else:
                print(f"[WARNING] ffmpeg downscale failed with exit code {res.returncode}. Falling back to original video.")
                print(f"ffmpeg stdout: {res.stdout}")
                print(f"ffmpeg stderr: {res.stderr}")
        except Exception as e:
            print(f"[WARNING] ffmpeg downscale raised exception: {e}. Falling back to original video.")
    else:
        print(f"[TIMING] ffmpeg downscale skipped (native height={native_h}p <= 1080p)")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video in Modal container")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(total_frames / (target * 2.0)))

        # 1. Read frames sequentially in a single pass to avoid opening the video file multiple times
        frames_to_process = []
        fi = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if fi % step == 0:
                ok_ret, frame = cap.retrieve()
                if ok_ret and frame is not None:
                    # Resize immediately to max 1920 width to save memory
                    h, w = frame.shape[:2]
                    if w > 1920:
                        frame = cv2.resize(frame, (1920, int(h * 1920 / w)))
                    frames_to_process.append((fi, frame))
            fi += 1
        
        cap.release()

        def process_decoded_frame(args):
            frame_idx, frame = args
            h, w = frame.shape[:2]
            
            # Compute blur score
            if w > 960:
                f_resized = cv2.resize(frame, (960, int(h * 960 / w)), interpolation=cv2.INTER_NEAREST)
            else:
                f_resized = frame
            gray = cv2.cvtColor(f_resized, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_32F).var()
            
            if blur_score >= blur_thresh:
                ok_enc, encoded_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok_enc:
                    return (frame_idx, blur_score, encoded_img.tobytes())
            return None

        # Execute in parallel to speed up Laplacian scoring significantly
        candidates = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(process_decoded_frame, frames_to_process)
            for res in results:
                if res is not None:
                    candidates.append(res)

        # Sort candidates by frame index
        candidates.sort(key=lambda x: x[0])

        if not candidates:
            raise ValueError(f"No sharp frames found with blur_thresh={blur_thresh}")

        # 2. Overlap Validation & Resampling using ORB
        orb = cv2.ORB_create(nfeatures=1000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        n = min(target, len(candidates))
        idxs = list(np.linspace(0, len(candidates) - 1, n, dtype=int))

        current_idxs = list(idxs)
        added_count = 0
        max_added = 10
        i = 0
        gaps_detected = []
        threshold = 0.15

        while i < len(current_idxs) - 1 and added_count < max_added:
            idx_a = current_idxs[i]
            idx_b = current_idxs[i+1]

            if idx_b - idx_a <= 1:
                i += 1
                continue

            raw_a = candidates[idx_a][2]
            raw_b = candidates[idx_b][2]

            gray_a = cv2.imdecode(np.frombuffer(raw_a, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            gray_b = cv2.imdecode(np.frombuffer(raw_b, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

            h_a, w_a = gray_a.shape[:2]
            if w_a > 640:
                gray_a = cv2.resize(gray_a, (640, int(h_a * 640 / w_a)))
            h_b, w_b = gray_b.shape[:2]
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

            if ratio < threshold:
                idx_mid = (idx_a + idx_b) // 2
                current_idxs.insert(i + 1, idx_mid)
                added_count += 1
                gaps_detected.append(f"Gap between frames {i} and {i+1} ({ratio*100:.1f}% overlap)")
                i += 2
            else:
                i += 1

        overlap_warning = None
        if gaps_detected:
            overlap_warning = f"[WARNING] Low overlap warning: {len(gaps_detected)} gaps detected. Resampled +{added_count} frames. Try slower/steadier capture next time."

        final_frames_bytes = [candidates[idx][2] for idx in current_idxs]

        # Direct R2 frame upload from Modal (eliminates 50-70MB outbound bandwidth back to Render)
        r2_frames_prefix = None
        if not tree_code and r2_key:
            base_fname = os.path.basename(r2_key)
            if "_" in base_fname:
                tree_code = base_fname.split("_")[0]
            else:
                tree_code = os.path.splitext(base_fname)[0]

        if r2_config and tree_code:
            import boto3
            from botocore.config import Config as BotoConfig
            s3_frame_uploader = boto3.client(
                "s3",
                endpoint_url=f"https://{r2_config['CLOUDFLARE_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                aws_access_key_id=r2_config["R2_ACCESS_KEY_ID"],
                aws_secret_access_key=r2_config["R2_SECRET_ACCESS_KEY"],
                config=BotoConfig(signature_version="s3v4"),
                region_name="auto",
            )
            r2_frames_prefix = f"tree_scans/{tree_code}/frames/"
            bucket = r2_config["R2_BUCKET_NAME"]
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Uploading {len(final_frames_bytes)} frames directly to R2 prefix {r2_frames_prefix}...")
            
            def upload_single_frame(args):
                idx, f_bytes = args
                key = f"{r2_frames_prefix}{idx:04d}.jpg"
                s3_frame_uploader.put_object(Bucket=bucket, Key=key, Body=f_bytes, ContentType="image/jpeg")
                return key
                
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(upload_single_frame, enumerate(final_frames_bytes)))
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Uploaded {len(final_frames_bytes)} frames to R2 directly from Modal in parallel!")

            # Return only 3 representative frames to Render (first, middle, last) for Gate 1 & UI thumbnail preview
            # This cuts response payload from ~50MB to <500KB!
            sample_idxs = [0]
            if len(final_frames_bytes) >= 3:
                sample_idxs.extend([len(final_frames_bytes)//2, len(final_frames_bytes)-1])
            elif len(final_frames_bytes) == 2:
                sample_idxs.append(1)
            preview_frames = [final_frames_bytes[i] for i in sample_idxs]
        else:
            preview_frames = final_frames_bytes

        return {
            "frames": preview_frames,
            "num_frames": len(final_frames_bytes),
            "r2_frames_prefix": r2_frames_prefix,
            "overlap_warning": overlap_warning,
            "t_modal_enter": t_modal_enter,
            "t_modal_exit": time.time(),
            "global_import_time": global_import_time
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        # Clean up the temporary video_uploads/ key from R2 after successful processing
        if r2_key and r2_config:
            try:
                s3.delete_object(Bucket=r2_config["R2_BUCKET_NAME"], Key=r2_key)
                print(f"[CLEANUP] Deleted temp R2 key: {r2_key}")
            except Exception as cleanup_err:
                print(f"[CLEANUP] Failed to delete R2 key {r2_key}: {cleanup_err}")


@app.function(
    image=image,
    timeout=180,
    cpu=1.0
)
def align_and_filter_ply_modal(
    ply_bytes: bytes,
    points3d_all_bytes: bytes,
    p1: list[float] = None,
    p2: list[float] = None,
    width: int = None,
    height: int = None,
    frame_idx: int = None,
) -> dict:
    import os
    import io
    import numpy as np
    import tempfile
    import shutil
    from scipy.spatial import KDTree

    temp_dir = tempfile.mkdtemp()
    temp_ply_path = os.path.join(temp_dir, "temp_points3d.ply")
    with open(temp_ply_path, "wb") as f:
        f.write(ply_bytes)

    # 1. Parse raw PLY
    def parse_ply_points_local(ply_path):
        with open(ply_path, "rb") as f:
            raw_props = []
            num_vertices = 0
            is_binary = False
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("format binary_little_endian"):
                    is_binary = True
                elif line.startswith("element vertex"):
                    num_vertices = int(line.split()[-1])
                elif line.startswith("property"):
                    parts = line.split()
                    if len(parts) >= 3:
                        raw_props.append((parts[1], parts[2]))
                elif line == "end_header":
                    break
            if num_vertices <= 0:
                raise ValueError("No vertices in header")
            
            dtype_map = []
            for p_type, p_name in raw_props:
                if p_type in ("float", "float32"):   dtype_map.append((p_name, "<f4"))
                elif p_type in ("int", "int32", "uint"): dtype_map.append((p_name, "<i4"))
                elif p_type in ("uchar", "uint8"):   dtype_map.append((p_name, "u1"))
                else:                                 dtype_map.append((p_name, "<f4"))
            
            vertex_data = np.fromfile(f, dtype=np.dtype(dtype_map), count=num_vertices)
            return vertex_data, raw_props, num_vertices

    try:
        vertex_data, raw_props, num_vertices = parse_ply_points_local(temp_ply_path)
        x = vertex_data['x']
        y = vertex_data['y']
        z = vertex_data['z']
        pts_world_raw = np.column_stack((x, y, z))

        # 2. Map coordinates and align if p1 and p2 are provided and points3d_all_bytes is available
        P1_aligned = None
        P2_aligned = None
        R = np.eye(3)
        t = np.zeros(3)
        s = 1.0
        
        # Umeyama SVD fit function
        def umeyama_fit(A, B):
            n = A.shape[0]
            centroid_A = np.mean(A, axis=0)
            centroid_B = np.mean(B, axis=0)
            AA = A - centroid_A
            BB = B - centroid_B
            var_A = np.mean(np.sum(AA**2, axis=1))
            if var_A < 1e-8:
                return np.eye(3), np.zeros(3), 1.0
            H = (AA.T @ BB) / n
            U, S, Vt = np.linalg.svd(H)
            R_fit = Vt.T @ U.T
            if np.linalg.det(R_fit) < 0:
                Vt[2, :] *= -1
                R_fit = Vt.T @ U.T
            d = np.ones(3)
            if np.linalg.det(H) < 0:
                d[2] = -1
            s_fit = float(np.sum(S * d) / var_A)
            t_fit = centroid_B - s_fit * (R_fit @ centroid_A)
            return R_fit, t_fit, s_fit

        # ICP function
        def register_pointmap_to_world_local(pointmap, pts_world, max_iterations=25, subsample=1500):
            if len(pointmap.shape) == 3:
                valid_mask = ~np.all(pointmap == 0, axis=-1) & ~np.any(np.isnan(pointmap), axis=-1)
                pts_cam = pointmap[valid_mask]
            else:
                pts_cam = pointmap
            if len(pts_cam) < 10 or len(pts_world) < 10:
                return np.eye(3), np.zeros(3), 1.0
            if len(pts_cam) > subsample:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(pts_cam), size=subsample, replace=False)
                src = pts_cam[idx]
            else:
                src = pts_cam
            dst = pts_world
            tree = KDTree(dst)
            R_fit = np.eye(3)
            t_fit = np.mean(dst, axis=0) - np.mean(src, axis=0)
            s_fit = 1.0
            dist_threshold = 0.5
            for _ in range(max_iterations):
                src_transformed = s_fit * (src @ R_fit.T) + t_fit
                distances, indices = tree.query(src_transformed, k=1, workers=-1)
                valid = distances < dist_threshold
                if np.sum(valid) < 10:
                    break
                src_corr = src[valid]
                dst_corr = dst[indices[valid]]
                R_fit, t_fit, s_fit = umeyama_fit(src_corr, dst_corr)
            return R_fit, t_fit, s_fit

        def map_pixel_to_cropped_local(u_org, v_org, W1, H1, W_crop, H_crop, size=512):
            scale = size / max(W1, H1)
            W_resized = int(round(W1 * scale))
            H_resized = int(round(H1 * scale))
            cx, cy = W_resized // 2, H_resized // 2
            halfw = (cx // 8) * 8
            halfh = (cy // 8) * 8
            left = cx - halfw
            top = cy - halfh
            u_crop = u_org * scale - left
            v_crop = v_org * scale - top
            u_crop = max(0, min(W_crop - 1, int(round(u_crop))))
            v_crop = max(0, min(H_crop - 1, int(round(v_crop))))
            return u_crop, v_crop

        def get_robust_3d_point_local(pointmap, u, v, window=15):
            H, W, _ = pointmap.shape
            points = []
            for du in range(-window//2, window//2 + 1):
                for dv in range(-window//2, window//2 + 1):
                    nu, nv = u + du, v + dv
                    if 0 <= nu < W and 0 <= nv < H:
                        pt = pointmap[nv, nu]
                        if not np.all(pt == 0) and not np.any(np.isnan(pt)):
                            points.append(pt)
            if len(points) == 0:
                for r in range(1, 51):
                    found = False
                    for du in range(-r, r + 1):
                        for dv in [-r, r]:
                            nu, nv = u + du, v + dv
                            if 0 <= nu < W and 0 <= nv < H:
                                pt = pointmap[nv, nu]
                                if not np.all(pt == 0) and not np.any(np.isnan(pt)):
                                    points.append(pt)
                                    found = True
                    for dv in range(-r + 1, r):
                        for du in [-r, r]:
                            nu, nv = u + du, v + dv
                            if 0 <= nu < W and 0 <= nv < H:
                                pt = pointmap[nv, nu]
                                if not np.all(pt == 0) and not np.any(np.isnan(pt)):
                                    points.append(pt)
                                    found = True
                    if found:
                        break
            if len(points) == 0:
                return pointmap[v, u]
            points = sorted(points, key=lambda p: p[2])
            n_keep = max(1, int(len(points) * 0.3))
            return np.mean(points[:n_keep], axis=0)

        # Main processing flow inside Modal function
        center_x = None
        center_z = None
        
        if p1 is not None and p2 is not None and points3d_all_bytes and len(points3d_all_bytes) > 100:
            try:
                pts3d = np.load(io.BytesIO(points3d_all_bytes))
                N_f, H_crop, W_crop, _ = pts3d.shape
                if frame_idx is not None and 0 <= frame_idx < N_f:
                    target_idx = frame_idx
                else:
                    valid_counts = np.array([
                        np.sum(~np.all(pts3d[i] == 0, axis=-1) & ~np.any(np.isnan(pts3d[i]), axis=-1))
                        for i in range(N_f)
                    ])
                    target_idx = int(np.argmax(valid_counts))
                pointmap = pts3d[target_idx]

                u1_crop, v1_crop = map_pixel_to_cropped_local(p1[0], p1[1], width, height, W_crop, H_crop)
                u2_crop, v2_crop = map_pixel_to_cropped_local(p2[0], p2[1], width, height, W_crop, H_crop)

                P1_cam = get_robust_3d_point_local(pointmap, u1_crop, v1_crop)
                P2_cam = get_robust_3d_point_local(pointmap, u2_crop, v2_crop)

                R, t, s = register_pointmap_to_world_local(pointmap, pts_world_raw)
                P1_val = s * (P1_cam @ R.T) + t
                P2_val = s * (P2_cam @ R.T) + t
                P1_aligned = P1_val.tolist()
                P2_aligned = P2_val.tolist()
                center_x = P1_val[0]
                center_z = P1_val[2]
            except Exception as e:
                print(f"[MODAL-ICP-ERROR] ICP Alignment failed: {e}")

        # 3. Apply orientation-agnostic PLY filtering (ground plane separation & trunk crop)
        if len(pts_world_raw) > 10000:
            rng = np.random.default_rng(42)
            sample_pts = pts_world_raw[rng.choice(len(pts_world_raw), 10000, replace=False)]
        else:
            sample_pts = pts_world_raw

        # RANSAC ground plane detection
        def fit_plane_ransac_local(pts_in, max_iter=100, thresh=0.06):
            n_pts = len(pts_in)
            if n_pts < 10:
                return None, np.zeros(n_pts, dtype=bool)
            r_gen = np.random.default_rng(42)
            samples = r_gen.choice(n_pts, size=(max_iter, 3), replace=True)
            best_in = np.zeros(n_pts, dtype=bool)
            best_pl = None
            for s_idx in samples:
                p1_s, p2_s, p3_s = pts_in[s_idx[0]], pts_in[s_idx[1]], pts_in[s_idx[2]]
                n_vec = np.cross(p2_s - p1_s, p3_s - p1_s)
                n_len = np.linalg.norm(n_vec)
                if n_len < 1e-6:
                    continue
                n_vec = n_vec / n_len
                d_val = -np.dot(n_vec, p1_s)
                inliers_m = np.abs(np.dot(pts_in, n_vec) + d_val) < thresh
                if np.sum(inliers_m) > np.sum(best_in):
                    best_in = inliers_m
                    best_pl = (n_vec, d_val)
            return best_pl, best_in

        plane_res, grnd_mask = fit_plane_ransac_local(sample_pts)
        if plane_res is not None and np.sum(grnd_mask) > len(sample_pts) * 0.05:
            n_ground, d_ground = plane_res
            h_g = np.dot(sample_pts, n_ground) + d_ground
            if np.median(h_g) < 0:
                n_ground = -n_ground
                d_ground = -d_ground
                h_g = -h_g
            fg_pts_modal = sample_pts[h_g > 0.04]
        else:
            n_ground = np.array([0.0, -1.0, 0.0])
            fg_pts_modal = sample_pts

        if len(fg_pts_modal) < 20:
            fg_pts_modal = sample_pts

        ref = np.array([1.0, 0.0, 0.0]) if abs(n_ground[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u1 = np.cross(n_ground, ref)
        u1 = u1 / (np.linalg.norm(u1) + 1e-9)
        u2 = np.cross(n_ground, u1)

        p_u1 = np.dot(fg_pts_modal, u1)
        p_u2 = np.dot(fg_pts_modal, u2)

        hist, xedges, yedges = np.histogram2d(p_u1, p_u2, bins=35)
        max_idx = np.unravel_index(np.argmax(hist), hist.shape)
        peak_u1 = 0.5 * (xedges[max_idx[0]] + xedges[max_idx[0] + 1])
        peak_u2 = 0.5 * (yedges[max_idx[1]] + yedges[max_idx[1] + 1])

        if center_x is not None and center_z is not None:
            # When manual clicks exist, center crop on P1
            p_u1_all = np.dot(pts_world_raw - P1_val, u1)
            p_u2_all = np.dot(pts_world_raw - P1_val, u2)
            dist_sq = p_u1_all**2 + p_u2_all**2
        else:
            p_u1_all = np.dot(pts_world_raw, u1)
            p_u2_all = np.dot(pts_world_raw, u2)
            dist_sq = (p_u1_all - peak_u1)**2 + (p_u2_all - peak_u2)**2

        CROP_RADIUS = 0.85
        crop_mask = dist_sq <= CROP_RADIUS**2

        if np.sum(crop_mask) < 20:
            crop_mask = np.ones(len(pts_world_raw), dtype=bool)

        filtered_vertex_data = vertex_data[crop_mask]
        filtered_xyz = pts_world_raw[crop_mask]

        if len(filtered_xyz) >= 20:
            tree = KDTree(filtered_xyz)
            nb_neighbors = 20
            std_ratio = 2.0
            distances, _ = tree.query(filtered_xyz, k=nb_neighbors + 1, workers=-1)
            mean_dists = distances[:, 1:].mean(axis=1)

            global_mean = mean_dists.mean()
            global_std  = mean_dists.std()
            threshold   = global_mean + std_ratio * global_std

            inlier_mask = mean_dists <= threshold
            filtered_vertex_data = filtered_vertex_data[inlier_mask]

        # 4. Save filtered PLY back to bytes
        n_filtered = len(filtered_vertex_data)
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n_filtered}",
        ]
        for p_type, p_name in raw_props:
            header_lines.append(f"property {p_type} {p_name}")
        header_lines.append("end_header")
        header_enc = "\n".join(header_lines) + "\n"

        filtered_ply_bytes = header_enc.encode("ascii") + filtered_vertex_data.tobytes()

        return {
            "P1": P1_aligned,
            "P2": P2_aligned,
            "filtered_ply": filtered_ply_bytes
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.function(image=image)
def convert_ply_on_modal(ply_bytes: bytes) -> bytes:
    import subprocess
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_in:
        temp_in.write(ply_bytes)
        temp_in_path = temp_in.name
        
    temp_out_path = temp_in_path.replace(".ply", ".ksplat")
    
    cmd = [
        "node", "/workspace/GaussianSplats3D/util/create-ksplat.js",
        temp_in_path, temp_out_path,
        "1", "1"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        try:
            os.remove(temp_in_path)
        except:
            pass
        raise RuntimeError(f"KSplat conversion failed: {res.stderr}")
        
    with open(temp_out_path, "rb") as f:
        ksplat_bytes = f.read()
        
    try:
        os.remove(temp_in_path)
        os.remove(temp_out_path)
    except:
        pass
        
    return ksplat_bytes


@app.local_entrypoint()
def main():
    import os
    image_dir = "./test_images"
    images_bytes = []
    if os.path.exists(image_dir):
        for f in os.listdir(image_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                with open(os.path.join(image_dir, f), "rb") as file:
                    images_bytes.append(file.read())
                    
    if not images_bytes:
        print("No images found in ./test_images")
        return
        
    print(f"Calling run_reconstruction with {len(images_bytes)} images...")
    res = run_reconstruction.remote(images_bytes)
    print(f"Received result of size {len(res)} bytes")
