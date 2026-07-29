import os
import modal

app = modal.App("instantsplat-app")

# GPU Configuration - easy to swap to "a100" or "h100" if "a10g" triggers CUDA Out Of Memory (OOM) errors
GPU_CONFIG = "a10g"

# Clone the repository recursively and download the MASt3R checkpoint into it during image build
image = (
    modal.Image.from_registry("dockerzhiwen/instantsplat_public:2.0")
    .run_commands(
        "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git libgl1-mesa-glx libglib2.0-0",
        "git clone --recursive https://github.com/NVlabs/InstantSplat.git /workspace/InstantSplat",
        "/opt/conda/bin/pip install 'numpy==1.26.0' open3d plyfile icecream pyquaternion configargparse",
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
        "urllib.request.urlretrieve(url, t) if not os.path.exists(t) else print('exists');\""
    )
)

@app.function(
    gpu=GPU_CONFIG,
    timeout=1800,  # 30 minutes
    image=image
)
def run_reconstruction(images_bytes: list[bytes]) -> dict:
    import os
    import time
    import shutil
    import subprocess
    import glob
    
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
    
    # Write incoming images to the input directory
    for i, img_bytes in enumerate(images_bytes):
        img_path = os.path.join(input_dir, f"{i:03d}.jpg")
        with open(img_path, "wb") as f:
            f.write(img_bytes)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved {len(images_bytes)} images to {input_dir}")
    
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
            bufsize=1
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
    # Run WITHOUT --n_views — let init_geo pick its default number of views.
    # It will create a sparse_{N}/ folder inside source_path which we detect next.
    init_cmd = [
        "python3", "init_geo.py",
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

    # 4. Stage 2: Fast 3D-Gaussian Optimization (train.py)
    train_cmd = [
        "python3", "train.py",
        "--source_path", source_path,
        "--model_path", output_dir,
        "--iterations", "7000",
        "--n_views", str(detected_n_views),
        "--optim_pose",
        "--test_iterations", "7000"
    ]
    run_command(train_cmd, "Fast 3D-Gaussian Optimization (train.py)")
    
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

        # Pass 1: Spatial statistical outlier removal (KNN-based)
        NB_NEIGHBORS = 20
        STD_RATIO    = 2.0
        xyz = np.column_stack((vertex_data["x"], vertex_data["y"], vertex_data["z"])).astype(np.float64)
        tree = KDTree(xyz)
        distances, _ = tree.query(xyz, k=NB_NEIGHBORS + 1, workers=-1)
        mean_dists = distances[:, 1:].mean(axis=1)
        threshold_spatial = mean_dists.mean() + STD_RATIO * mean_dists.std()
        spatial_mask = mean_dists <= threshold_spatial
        combined_mask &= spatial_mask
        n_spatial = int((~spatial_mask).sum())
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pass 1 (spatial): removed {n_spatial:,} outliers | remaining {combined_mask.sum():,}")

        # Pass 2: Low-opacity removal (logit threshold — keep sigmoid >= ~0.018)
        MIN_OPACITY_LOGIT = -4.0
        if "opacity" in vertex_data.dtype.names:
            opacity_mask = vertex_data["opacity"] >= MIN_OPACITY_LOGIT
            combined_mask &= opacity_mask
            n_opacity = int((~opacity_mask).sum())
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pass 2 (opacity): removed {n_opacity:,} | remaining {combined_mask.sum():,}")

        # Pass 3: Oversized Gaussian removal (log-scale filter)
        MAX_LOG_SCALE = -1.0   # exp(-1) ~0.37 units; anything larger is almost certainly a floater
        scale_names = [n for n in vertex_data.dtype.names if n.startswith("scale_")]
        if scale_names:
            scales = np.column_stack([vertex_data[n] for n in scale_names])
            max_scales = scales.max(axis=1)
            scale_mask = max_scales <= MAX_LOG_SCALE
            combined_mask &= scale_mask
            n_scale = int((~scale_mask).sum())
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pass 3 (scale):   removed {n_scale:,} | remaining {combined_mask.sum():,}")

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

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Export completed successfully ---")
    return {
        "splat": splat_data,
        "points3d": points3d_data,  # None if not found
    }


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
