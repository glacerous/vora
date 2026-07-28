import os
import sys
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DBH_Extractor")

def load_scale_factor(manual_scale=1.0):
    import json
    paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_images", "calibration.json"),
        os.path.join(os.getcwd(), "calibration.json")
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, 'r') as fh:
                    data = json.load(fh)
                    if "scale_factor" in data:
                        scale = float(data["scale_factor"])
                        logger.info(f"Loaded scale_factor {scale} from calibration file: {p}")
                        return scale
            except Exception as e:
                logger.warning(f"Failed to read calibration file {p}: {e}")
    logger.info(f"No calibration file found. Using default/manual scale_factor: {manual_scale}")
    return manual_scale

def parse_ply_points(ply_path):
    logger.info(f"Parsing PLY file: {ply_path}")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

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
                
        logger.info(f"Header parsed. Vertices: {num_vertices}, Format: {'Binary' if is_binary else 'ASCII'}, Properties count: {len(properties)}")
        
        if num_vertices <= 0:
            raise ValueError("No vertices found in PLY file header.")

        if is_binary:
            dtype_map = []
            total_bytes = 0
            for p_type, p_name in properties:
                if p_type in ("float", "float32"):
                    dtype_map.append((p_name, "<f4"))
                    total_bytes += 4
                elif p_type in ("int", "int32", "uint"):
                    dtype_map.append((p_name, "<i4"))
                    total_bytes += 4
                elif p_type in ("uchar", "uint8"):
                    dtype_map.append((p_name, "u1"))
                    total_bytes += 1
                else:
                    dtype_map.append((p_name, "<f4"))
                    total_bytes += 4
                    
            vertex_data = np.fromfile(f, dtype=np.dtype(dtype_map), count=num_vertices)
            x = vertex_data['x']
            y = vertex_data['y']
            z = vertex_data['z']
            points = np.column_stack((x, y, z))
        else:
            points = []
            for _ in range(num_vertices):
                line = f.readline().decode("ascii", errors="ignore").strip()
                if not line:
                    break
                parts = line.split()
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            points = np.array(points, dtype=np.float32)

        return points

def fit_circle_2d(points_2d):
    if len(points_2d) < 3:
        return None, None, None, "Too few points (< 3) for circle fitting"
        
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    
    M = np.column_stack((x, y, np.ones_like(x)))
    b = x**2 + y**2
    
    try:
        sol, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
        A, B, C = sol
        xc = A / 2.0
        yc = B / 2.0
        R_sq = C + xc**2 + yc**2
        if R_sq < 0:
            return None, None, None, "Invalid negative radius squared"
        return xc, yc, np.sqrt(R_sq), None
    except Exception as e:
        return None, None, None, f"Least squares solver error: {e}"

def fit_circle_robust(points_2d, max_iters=5, outlier_threshold_ratio=0.15):
    inliers = points_2d
    xc, yc, R = None, None, None
    for i in range(max_iters):
        if len(inliers) < 3:
            break
        xc, yc, R, err = fit_circle_2d(inliers)
        if err is not None:
            break
        dists = np.sqrt((inliers[:, 0] - xc)**2 + (inliers[:, 1] - yc)**2)
        radial_errors = np.abs(dists - R)
        thresh = max(0.01, outlier_threshold_ratio * R)
        mask = radial_errors < thresh
        num_inliers = np.sum(mask)
        if num_inliers == len(inliers) or num_inliers < 3:
            break
        inliers = inliers[mask]
        
    if R is not None:
        dists = np.sqrt((inliers[:, 0] - xc)**2 + (inliers[:, 1] - yc)**2)
        mean_err = np.mean(np.abs(dists - R))
        return xc, yc, R, mean_err
    return None, None, None, None

def extract_dbh(ply_path, scale_factor=1.0, vertical_axis='z', breast_height=1.3, tolerance=0.05):
    scale = load_scale_factor(scale_factor)
    try:
        points = parse_ply_points(ply_path)
    except Exception as e:
        return {"error": f"Failed to load point cloud: {e}"}
        
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[vertical_axis.lower()]
    z_coords = points[:, axis_idx]
    
    z_min = np.percentile(z_coords, 1)
    z_max = np.percentile(z_coords, 99)
    estimated_height_m = float((z_max - z_min) * scale)
    
    z_target = z_min + (breast_height / scale)
    tol = tolerance / scale
    slice_mask = np.abs(z_coords - z_target) <= tol
    slice_points = points[slice_mask]
    method_used = "Numpy Point Cloud Slicing"
    
    proj_axes = [i for i in [0, 1, 2] if i != axis_idx]
    points_2d = slice_points[:, proj_axes]
    
    if len(points_2d) < 10:
        return {
            "dbh_cm": 0.0,
            "height_m": estimated_height_m,
            "confidence_note": "Unreliable: Too few points in slice.",
            "method": method_used,
            "slice_points_count": len(points_2d),
            "mean_fit_error_cm": 0.0
        }
        
    xc, yc, R, mean_err = fit_circle_robust(points_2d)
    if R is None:
        return {
            "dbh_cm": 0.0,
            "height_m": estimated_height_m,
            "confidence_note": "Unreliable: Circle fitting failed.",
            "method": method_used,
            "slice_points_count": len(points_2d),
            "mean_fit_error_cm": 0.0
        }
        
    dbh_m = R * 2.0 * scale
    dbh_cm = dbh_m * 100.0
    
    confidence = "High" if (mean_err * scale <= 0.02) else "Medium"
    
    return {
        "dbh_cm": float(round(dbh_cm, 2)),
        "height_m": float(round(estimated_height_m, 2)),
        "confidence_note": confidence,
        "method": method_used,
        "slice_points_count": len(points_2d),
        "mean_fit_error_cm": float(round(mean_err * scale * 100, 2)),
        "geometry_3d": {
            "center_x": float(round(xc, 4)),
            "center_y": float(round(yc, 4)),
            "radius_units": float(round(R, 4)),
            "z_min": float(round(z_min, 4)),
            "z_max": float(round(z_max, 4)),
            "z_target": float(round(z_target, 4)),
            "axis_name": vertical_axis.lower(),
            "scale_factor": scale,
        }
    }


def extract_dbh_from_mast3r(ply_path: str, scale_factor: float = 1.0,
                             breast_height: float = 1.3) -> dict:
    logger.info("[MAST3R DBH] Starting DBH extraction from MASt3R point cloud...")
    scale = load_scale_factor(scale_factor)

    try:
        points = parse_ply_points(ply_path)
    except Exception as exc:
        logger.error(f"[MAST3R DBH] Failed to load point cloud: {exc}")
        return {"error": f"Failed to load MASt3R point cloud: {exc}"}

    if len(points) < 30:
        return {"error": f"MASt3R cloud too sparse for DBH extraction ({len(points)} pts)"}

    ranges = points.max(axis=0) - points.min(axis=0)
    axis_idx = int(np.argmax(ranges))
    axis_name = ["x", "y", "z"][axis_idx]
    proj_axes = [i for i in [0, 1, 2] if i != axis_idx]
    logger.info(f"[MAST3R DBH] Vertical axis={axis_name} (range={ranges[axis_idx]:.4f}), total points={len(points)}")

    z_coords  = points[:, axis_idx]
    z_min     = float(np.percentile(z_coords, 2))
    z_max     = float(np.percentile(z_coords, 98))
    total_h   = z_max - z_min
    estimated_height_m = float(total_h * scale)

    logger.info(f"[MAST3R DBH] Height range: {z_min:.4f} – {z_max:.4f} units | estimated real height: {estimated_height_m:.2f} m")

    bh_units  = breast_height / scale
    z_target  = z_min + bh_units

    if z_target >= z_max * 0.95:
        z_target = z_min + 0.30 * total_h
        logger.warning("[MAST3R DBH] Breast height exceeds cloud bounds. Using 30 % position.")

    base_tol  = 0.10 / scale
    density_factor = max(1.0, (5000 / max(len(points), 1)) ** 0.5)
    tol       = base_tol * density_factor
    tol       = min(tol, total_h * 0.15)
    tol       = max(tol, total_h * 0.02)
    logger.info(f"[MAST3R DBH] Slice tolerance: {tol:.4f} units (base={base_tol:.4f}, density_factor={density_factor:.2f})")

    offsets = [-tol * 0.4, 0.0, tol * 0.4]
    radii   = []

    for off in offsets:
        z_t  = z_target + off
        mask = np.abs(z_coords - z_t) <= tol
        pts_slice = points[mask]
        if len(pts_slice) < 5:
            continue
        pts_2d = pts_slice[:, proj_axes]
        xc, yc, R, err = fit_circle_2d(pts_2d)
        if R is not None and R > 0:
            radii.append(R)

    method_used = "MASt3R multi-slice median"
    if not radii:
        logger.warning("[MAST3R DBH] Multi-slice failed, trying lower-trunk fallback...")
        mask = (z_coords >= z_min) & (z_coords <= z_min + total_h * 0.5)
        pts_trunk = points[mask]
        if len(pts_trunk) < 5:
            return {"error": "No points in trunk region — cloud may be too sparse or misoriented"}
        pts_2d = pts_trunk[:, proj_axes]
        _, _, R, _ = fit_circle_robust(pts_2d)
        if R is None:
            return {"error": "Circle fitting failed on MASt3R trunk region"}
        radii = [R]
        method_used = "MASt3R lower-trunk fallback"

    R_final = float(np.median(radii))
    dbh_m   = R_final * 2.0 * scale
    dbh_cm  = dbh_m * 100.0

    mask_best  = np.abs(z_coords - z_target) <= tol
    pts_best   = points[mask_best]
    slice_count = len(pts_best)
    mean_err_cm = 0.0
    xc, yc = 0.0, 0.0
    if slice_count >= 5:
        pts_2d = pts_best[:, proj_axes]
        xc_fit, yc_fit, _, mean_err = fit_circle_robust(pts_2d)
        if xc_fit is not None:
            xc, yc = float(xc_fit), float(yc_fit)
        if mean_err is not None:
            mean_err_cm = float(round(mean_err * scale * 100, 2))

    if slice_count < 20:
        confidence = "Low (sparse MASt3R cloud at breast height)"
    elif mean_err_cm > 5.0:
        confidence = "Low (high trunk fitting noise)"
    elif mean_err_cm > 2.0:
        confidence = "Medium (some trunk surface noise)"
    else:
        confidence = "High"

    logger.info(f"[MAST3R DBH] Result: DBH={dbh_cm:.2f} cm, height={estimated_height_m:.2f} m, confidence={confidence}")

    return {
        "dbh_cm":             float(round(dbh_cm, 2)),
        "height_m":           float(round(estimated_height_m, 2)),
        "confidence_note":    confidence,
        "method":             method_used,
        "slice_points_count": slice_count,
        "mean_fit_error_cm":  mean_err_cm,
        "geometry_3d": {
            "center_x":       float(round(xc, 4)),
            "center_y":       float(round(yc, 4)),
            "radius_units":   float(round(R_final, 4)),
            "z_min":          float(round(z_min, 4)),
            "z_max":          float(round(z_max, 4)),
            "z_target":       float(round(z_target, 4)),
            "axis_name":      axis_name,
            "scale_factor":   scale,
        }
    }
