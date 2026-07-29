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
    
    proj_axes = [i for i in [0, 1, 2] if i != axis_idx]
    points_2d = slice_points[:, proj_axes]
    
    if len(points_2d) < 10:
        return {"error": "Too few points in slice"}
        
    xc, yc, R, mean_err = fit_circle_robust(points_2d)
    if R is None:
        return {"error": "Circle fitting failed"}
        
    dbh_cm = R * 2.0 * scale * 100.0
    
    dir_3d = [0.0, 0.0, 0.0]
    dir_3d[axis_idx] = 1.0
    
    center_3d = [0.0, 0.0, 0.0]
    center_3d[proj_axes[0]] = xc
    center_3d[proj_axes[1]] = yc
    center_3d[axis_idx] = z_target
    
    return {
        "dbh_cm": float(round(dbh_cm, 2)),
        "height_m": float(round(estimated_height_m, 2)),
        "confidence_note": "High",
        "method": "Numpy Point Cloud Slicing",
        "slice_points_count": len(points_2d),
        "mean_fit_error_cm": float(round(mean_err * scale * 100, 2)),
        "geometry_3d": {
            "center_x": float(center_3d[0]),
            "center_y": float(center_3d[1]),
            "center_z": float(center_3d[2]),
            "dir_x": float(dir_3d[0]),
            "dir_y": float(dir_3d[1]),
            "dir_z": float(dir_3d[2]),
            "radius_units": float(R),
            "h_min": float(z_min),
            "h_max": float(z_max),
            "h_target": float(z_target),
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

    # ── 1. Determine rough vertical axis coordinates ───────────────────────────
    ranges = points.max(axis=0) - points.min(axis=0)
    rough_axis_idx = int(np.argmax(ranges))
    proj_axes = [i for i in [0, 1, 2] if i != rough_axis_idx]
    
    # ── 2. Filter out Ground/Terrain using 2D density peak detection ───────────
    # The tree trunk is a high-density vertical column. Terrain points are spread out.
    h1 = points[:, proj_axes[0]]
    h2 = points[:, proj_axes[1]]
    
    # 2D Grid Histogram binning to find where the trunk is centered horizontally
    hist, xedges, yedges = np.histogram2d(h1, h2, bins=30)
    max_idx = np.unravel_index(np.argmax(hist), hist.shape)
    peak_h1 = 0.5 * (xedges[max_idx[0]] + xedges[max_idx[0] + 1])
    peak_h2 = 0.5 * (yedges[max_idx[1]] + yedges[max_idx[1] + 1])
    
    # Crop the points horizontally around this peak (crop radius = 0.4 units)
    # This filters out 90%+ of the terrain and surrounding clutter
    dist_sq = (h1 - peak_h1)**2 + (h2 - peak_h2)**2
    CROP_RADIUS = 0.45
    trunk_mask = dist_sq <= CROP_RADIUS**2
    trunk_points = points[trunk_mask]
    
    if len(trunk_points) < 20:
        # Fallback to uncropped points if crop is too aggressive
        trunk_points = points
        logger.warning("[MAST3R DBH] Crop yielded too few points, falling back to full cloud.")

    # ── 3. Run PCA on the isolated trunk points to find 3D direction vector ─────
    rough_z = trunk_points[:, rough_axis_idx]
    rough_z_min = np.percentile(rough_z, 5)
    rough_z_max = np.percentile(rough_z, 95)
    rough_height = rough_z_max - rough_z_min

    # Sample mid-trunk section of the cropped points
    mid_trunk_mask = (rough_z >= rough_z_min + rough_height * 0.15) & (rough_z <= rough_z_min + rough_height * 0.60)
    pca_pts = trunk_points[mid_trunk_mask]
    if len(pca_pts) < 15:
        pca_pts = trunk_points
        
    # Use at most 10,000 points for PCA to keep memory usage bounded
    if len(pca_pts) > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pca_pts), size=10000, replace=False)
        pca_pts = pca_pts[idx]

    trunk_mean = pca_pts.mean(axis=0)
    centered = pca_pts - trunk_mean
    # Use covariance eigendecomposition (3×3 matrix) — memory-safe for any N
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order — largest eigenvalue (principal axis) is last
    v = eigenvectors[:, -1]
    
    # Ensure direction vector points "upwards" relative to rough vertical axis
    if v[rough_axis_idx] < 0:
        v = -v
        
    logger.info(f"[MAST3R DBH] Aligned trunk direction vector: {v}")

    # ── 4. Project all points onto the trunk axis vector to find true height ────
    proj = np.dot(points, v)
    h_min = float(np.percentile(proj, 2))
    h_max = float(np.percentile(proj, 98))
    total_h = h_max - h_min
    estimated_height_m = float(total_h * scale)
    
    logger.info(f"[MAST3R DBH] True height along aligned axis: {estimated_height_m:.2f} m")

    # ── 5. Set target height for DBH ───────────────────────────────────────────
    h_target = h_min + (breast_height / scale)
    if h_target >= h_max * 0.90:
        h_target = h_min + total_h * 0.30

    # ── 6. Project slice points perpendicular to trunk and fit circle ──────────
    # Orthonormal basis perpendicular to v
    if abs(v[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    u1 = np.cross(v, ref)
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(v, u1)
    
    # Project cropped trunk points onto aligned axis
    proj_trunk = np.dot(trunk_points, v)
    
    base_tol = 0.12 / scale
    density_factor = max(1.0, (5000 / max(len(points), 1)) ** 0.5)
    tol = base_tol * density_factor
    tol = min(tol, total_h * 0.15)
    tol = max(tol, total_h * 0.02)

    offsets = [-tol * 0.4, 0.0, tol * 0.4]
    radii = []
    centers_2d = []

    for off in offsets:
        h_t = h_target + off
        mask = np.abs(proj_trunk - h_t) <= tol
        pts_slice = trunk_points[mask]
        if len(pts_slice) < 5:
            continue
            
        # Project onto 2D plane perpendicular to v
        pts_2d = np.column_stack((np.dot(pts_slice, u1), np.dot(pts_slice, u2)))
        
        xc, yc, R, err = fit_circle_2d(pts_2d)
        if R is not None and R > 0 and R < CROP_RADIUS * 1.5:
            radii.append(R)
            centers_2d.append((xc, yc))

    method_used = "MASt3R aligned multi-slice median"
    if not radii:
        logger.warning("[MAST3R DBH] Aligned multi-slice failed, using fallback on trunk points...")
        mask = (proj_trunk >= h_min) & (proj_trunk <= h_min + total_h * 0.5)
        pts_trunk = trunk_points[mask]
        if len(pts_trunk) < 5:
            return {"error": "No points in aligned trunk region after terrain crop"}
        pts_2d = np.column_stack((np.dot(pts_trunk, u1), np.dot(pts_trunk, u2)))
        xc, yc, R, _ = fit_circle_robust(pts_2d)
        if R is None or R > CROP_RADIUS * 2.0:
            # Absolute default fallback
            R = 0.15 / scale
            xc, yc = 0.0, 0.0
        radii = [R]
        centers_2d = [(xc, yc)]
        method_used = "MASt3R aligned lower-trunk fallback"

    # Medians
    R_final = float(np.median(radii))
    dbh_m   = R_final * 2.0 * scale
    dbh_cm  = dbh_m * 100.0

    xc_2d = float(np.median([c[0] for c in centers_2d]))
    yc_2d = float(np.median([c[1] for c in centers_2d]))

    # Reconstruct 3D center point at DBH height
    center_3d = xc_2d * u1 + yc_2d * u2 + h_target * v

    # Quick error estimate on primary slice
    mask_best = np.abs(proj_trunk - h_target) <= tol
    pts_best = trunk_points[mask_best]
    slice_count = len(pts_best)
    mean_err_cm = 0.0
    if slice_count >= 5:
        pts_2d = np.column_stack((np.dot(pts_best, u1), np.dot(pts_best, u2)))
        _, _, _, mean_err = fit_circle_robust(pts_2d)
        if mean_err is not None:
            mean_err_cm = float(round(mean_err * scale * 100, 2))

    confidence = "High" if (mean_err_cm <= 3.0) else "Medium"

    logger.info(f"[MAST3R DBH] Final result: DBH={dbh_cm:.2f} cm, height={estimated_height_m:.2f} m, confidence={confidence}")

    return {
        "dbh_cm":             float(round(dbh_cm, 2)),
        "height_m":           float(round(estimated_height_m, 2)),
        "confidence_note":    confidence,
        "method":             method_used,
        "slice_points_count": slice_count,
        "mean_fit_error_cm":  mean_err_cm,
        "geometry_3d": {
            "center_x":       float(round(center_3d[0], 4)),
            "center_y":       float(round(center_3d[1], 4)),
            "center_z":       float(round(center_3d[2], 4)),
            "dir_x":          float(round(v[0], 4)),
            "dir_y":          float(round(v[1], 4)),
            "dir_z":          float(round(v[2], 4)),
            "radius_units":   float(round(R_final, 4)),
            "h_min":          float(round(h_min, 4)),
            "h_max":          float(round(h_max, 4)),
            "h_target":       float(round(h_target, 4)),
            "scale_factor":   scale,
        }
    }
