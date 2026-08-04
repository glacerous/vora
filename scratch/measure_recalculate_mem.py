import os
import sys
import gc
import time
import subprocess
import requests
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()

from storage.d1_client import execute_d1_query
from carbon.dbh_extractor import parse_ply_points, register_pointmap_to_world

def get_mem_usage_mb():
    pid = os.getpid()
    if sys.platform == 'win32':
        try:
            cmd = f'powershell -NoProfile -Command "(Get-Process -Id {pid}).WorkingSet / 1MB"'
            output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
            return float(output)
        except Exception:
            return 0.0
    return 0.0

def main():
    print(f"1. Initial memory (idle): {get_mem_usage_mb():.2f} MB")
    
    # Fetch scan ID 46 from D1
    scan_id = 46
    sql = "SELECT * FROM tree_scans WHERE id = ?"
    scans = execute_d1_query(sql, [scan_id])
    if not scans:
        print("Scan not found!")
        return
        
    target_scan = scans[0]
    tree_code = target_scan.get("tree_code")
    splat_file_url = target_scan.get("splat_file_url")
    print(f"Target Scan Tree Code: {tree_code}")
    print(f"Splat File URL: {splat_file_url}")
    
    # Derive pointmap and points3d URLs
    if "_result.ply" in splat_file_url:
        pointmap_url = splat_file_url.replace("_result.ply", "_points3D_all.npy")
        points3d_url = splat_file_url.replace("_result.ply", "_points3d.ply")
    elif "result.ply" in splat_file_url:
        pointmap_url = splat_file_url.replace("result.ply", "points3D_all.npy")
        points3d_url = splat_file_url.replace("result.ply", "points3d.ply")
    else:
        pointmap_url = splat_file_url.replace(".ply", "_points3D_all.npy")
        points3d_url = splat_file_url.replace(".ply", "_points3d.ply")

    # Download NPY in memory
    print(f"\n2. Downloading NPY from {pointmap_url}...")
    mem_before_dl_npy = get_mem_usage_mb()
    res_npy = requests.get(pointmap_url, timeout=30)
    npy_content = res_npy.content
    mem_after_dl_npy = get_mem_usage_mb()
    print(f"Downloaded NPY size: {len(npy_content) / (1024*1024):.2f} MB")
    print(f"Memory after NPY download: {mem_after_dl_npy:.2f} MB (increase: {mem_after_dl_npy - mem_before_dl_npy:.2f} MB)")
    
    # Download PLY in memory
    print(f"\n3. Downloading PLY from {points3d_url}...")
    mem_before_dl_ply = get_mem_usage_mb()
    res_ply = requests.get(points3d_url, timeout=30)
    ply_content = res_ply.content
    mem_after_dl_ply = get_mem_usage_mb()
    print(f"Downloaded PLY size: {len(ply_content) / (1024*1024):.2f} MB")
    print(f"Memory after PLY download: {mem_after_dl_ply:.2f} MB (increase: {mem_after_dl_ply - mem_before_dl_ply:.2f} MB)")
    
    # Write temp files
    local_npy_path = "temp_points3D_all.npy"
    local_ply_path = "temp_points3d.ply"
    with open(local_npy_path, "wb") as f:
        f.write(npy_content)
    with open(local_ply_path, "wb") as f:
        f.write(ply_content)
        
    # Free download content buffers
    del npy_content
    del ply_content
    gc.collect()
    print(f"Memory after freeing raw download content buffers: {get_mem_usage_mb():.2f} MB")
    
    # Load NPY array
    print("\n4. Loading NPY array via np.load...")
    mem_before_load_npy = get_mem_usage_mb()
    pts3d = np.load(local_npy_path)
    mem_after_load_npy = get_mem_usage_mb()
    print(f"NPY shape: {pts3d.shape}")
    print(f"Memory after loading NPY: {mem_after_load_npy:.2f} MB (increase: {mem_after_load_npy - mem_before_load_npy:.2f} MB)")
    
    # Select representative pointmap frame
    N, H_crop, W_crop, _ = pts3d.shape
    valid_counts = np.array([
        np.sum(~np.all(pts3d[i] == 0, axis=-1) & ~np.any(np.isnan(pts3d[i]), axis=-1))
        for i in range(N)
    ])
    repr_idx = int(np.argmax(valid_counts))
    pointmap = pts3d[repr_idx]
    
    # Load PLY points
    print("\n5. Parsing PLY points via parse_ply_points...")
    mem_before_load_ply = get_mem_usage_mb()
    pts_world = parse_ply_points(local_ply_path)
    mem_after_load_ply = get_mem_usage_mb()
    print(f"PLY points shape: {pts_world.shape}")
    print(f"Memory after loading PLY: {mem_after_load_ply:.2f} MB (increase: {mem_after_load_ply - mem_before_load_ply:.2f} MB)")
    
    # Run ICP Registration
    print("\n6. Running register_pointmap_to_world (ICP + KDTree)...")
    mem_before_icp = get_mem_usage_mb()
    R, t = register_pointmap_to_world(pointmap, pts_world)
    mem_after_icp = get_mem_usage_mb()
    print(f"Memory after ICP registration: {mem_after_icp:.2f} MB (increase: {mem_after_icp - mem_before_icp:.2f} MB)")
    
    # Perform clean up
    print("\n7. Performing manual deletion (del + gc.collect)...")
    del pts3d
    del pointmap
    del pts_world
    gc.collect()
    print(f"Memory after explicit garbage collection: {get_mem_usage_mb():.2f} MB")
    
    # Clean up temp files
    try:
        os.remove(local_npy_path)
        os.remove(local_ply_path)
    except Exception:
        pass

if __name__ == "__main__":
    main()
