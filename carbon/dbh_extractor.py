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
    
    conf_note = "High"
    if estimated_height_m < 1.0:
        conf_note = f"WARNING: Trunk segment captured is only {estimated_height_m:.2f}m tall, insufficient to reach standard breast height (1.3m). DBH measurement may not represent true breast-height diameter - recommend recapturing with more trunk visible in frame"

    return {
        "dbh_cm": float(round(dbh_cm, 2)),
        "height_m": float(round(estimated_height_m, 2)),
        "confidence_note": conf_note,
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
    
    # ── 2. Stage 1 (Coarse Crop) ──────────────────────────────────────────────
    # Density peak 2D + crop radius 0.45 units for initial isolation
    h1 = points[:, proj_axes[0]]
    h2 = points[:, proj_axes[1]]
    
    # 2D Grid Histogram binning to find trunk horizontal peak center
    hist, xedges, yedges = np.histogram2d(h1, h2, bins=30)
    max_idx = np.unravel_index(np.argmax(hist), hist.shape)
    peak_h1 = 0.5 * (xedges[max_idx[0]] + xedges[max_idx[0] + 1])
    peak_h2 = 0.5 * (yedges[max_idx[1]] + yedges[max_idx[1] + 1])
    
    CROP_RADIUS = 0.45
    dist_sq = (h1 - peak_h1)**2 + (h2 - peak_h2)**2
    coarse_trunk_mask = dist_sq <= CROP_RADIUS**2
    coarse_trunk_points = points[coarse_trunk_mask]
    
    if len(coarse_trunk_points) < 20:
        coarse_trunk_points = points
        logger.warning("[MAST3R DBH] Coarse crop yielded too few points, using full cloud.")

    # ── 3. PCA + Circle Fitting (Pass 1 - Coarse) ─────────────────────────────
    rough_z = coarse_trunk_points[:, rough_axis_idx]
    rough_z_min = np.percentile(rough_z, 5)
    rough_z_max = np.percentile(rough_z, 95)
    rough_height = rough_z_max - rough_z_min

    # Sample mid-trunk section of coarse cropped points
    mid_trunk_mask = (rough_z >= rough_z_min + rough_height * 0.15) & (rough_z <= rough_z_min + rough_height * 0.60)
    pca_pts_coarse = coarse_trunk_points[mid_trunk_mask]
    if len(pca_pts_coarse) < 15:
        pca_pts_coarse = coarse_trunk_points
        
    if len(pca_pts_coarse) > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pca_pts_coarse), size=10000, replace=False)
        pca_pts_coarse = pca_pts_coarse[idx]

    trunk_mean_pass1 = pca_pts_coarse.mean(axis=0)
    centered_pass1 = pca_pts_coarse - trunk_mean_pass1
    cov_pass1 = (centered_pass1.T @ centered_pass1) / max(len(centered_pass1) - 1, 1)
    eigenvalues_pass1, eigenvectors_pass1 = np.linalg.eigh(cov_pass1)
    v_pass1 = eigenvectors_pass1[:, -1]
    
    if v_pass1[rough_axis_idx] < 0:
        v_pass1 = -v_pass1

    proj_pass1 = np.dot(points, v_pass1)
    h_min_pass1 = float(np.percentile(proj_pass1, 2))
    h_max_pass1 = float(np.percentile(proj_pass1, 98))
    total_h_pass1 = h_max_pass1 - h_min_pass1
    
    h_target_pass1 = h_min_pass1 + (breast_height / scale)
    if h_target_pass1 >= h_max_pass1 * 0.90:
        h_target_pass1 = h_min_pass1 + total_h_pass1 * 0.30

    if abs(v_pass1[0]) < 0.9:
        ref_pass1 = np.array([1.0, 0.0, 0.0])
    else:
        ref_pass1 = np.array([0.0, 1.0, 0.0])
    u1_pass1 = np.cross(v_pass1, ref_pass1)
    u1_pass1 = u1_pass1 / np.linalg.norm(u1_pass1)
    u2_pass1 = np.cross(v_pass1, u1_pass1)

    proj_trunk_pass1 = np.dot(coarse_trunk_points, v_pass1)
    base_tol = 0.12 / scale
    density_factor = max(1.0, (5000 / max(len(points), 1)) ** 0.5)
    tol = base_tol * density_factor
    tol = min(tol, total_h_pass1 * 0.15)
    tol = max(tol, total_h_pass1 * 0.02)

    offsets = [-tol * 0.4, 0.0, tol * 0.4]
    radii_pass1 = []
    centers_2d_pass1 = []

    for off in offsets:
        h_t = h_target_pass1 + off
        mask = np.abs(proj_trunk_pass1 - h_t) <= tol
        pts_slice = coarse_trunk_points[mask]
        if len(pts_slice) < 5:
            continue
        pts_2d = np.column_stack((np.dot(pts_slice, u1_pass1), np.dot(pts_slice, u2_pass1)))
        xc, yc, R, _ = fit_circle_2d(pts_2d)
        if R is not None and R > 0 and R < CROP_RADIUS * 1.5:
            radii_pass1.append(R)
            centers_2d_pass1.append((xc, yc))

    if not radii_pass1:
        mask = (proj_trunk_pass1 >= h_min_pass1) & (proj_trunk_pass1 <= h_min_pass1 + total_h_pass1 * 0.5)
        pts_trunk = coarse_trunk_points[mask]
        if len(pts_trunk) >= 5:
            pts_2d = np.column_stack((np.dot(pts_trunk, u1_pass1), np.dot(pts_trunk, u2_pass1)))
            xc, yc, R, _ = fit_circle_robust(pts_2d)
            if R is None or R > CROP_RADIUS * 2.0:
                R = 0.15 / scale
                xc, yc = 0.0, 0.0
            radii_pass1 = [R]
            centers_2d_pass1 = [(xc, yc)]
        else:
            radii_pass1 = [0.15 / scale]
            centers_2d_pass1 = [(0.0, 0.0)]

    R_pass1 = float(np.median(radii_pass1))
    xc_pass1 = float(np.median([c[0] for c in centers_2d_pass1]))
    yc_pass1 = float(np.median([c[1] for c in centers_2d_pass1]))
    center_3d_pass1 = xc_pass1 * u1_pass1 + yc_pass1 * u2_pass1 + h_target_pass1 * v_pass1

    # ── 4. Stage 2 (Fine Crop centered on PCA axis) ───────────────────────────
    # fine_crop_radius = radius_pass1 * tolerance_factor (e.g. 1.4)
    TOLERANCE_FACTOR = 1.4
    fine_crop_radius = R_pass1 * TOLERANCE_FACTOR

    # Perpendicular distance of all raw points to the 1st pass axis line
    w = points - center_3d_pass1
    h_proj = np.dot(w, v_pass1)
    perp = w - h_proj[:, np.newaxis] * v_pass1[np.newaxis, :]
    perp_dist = np.linalg.norm(perp, axis=1)

    fine_trunk_mask = perp_dist <= fine_crop_radius
    fine_trunk_points = points[fine_trunk_mask]

    if len(fine_trunk_points) < 20:
        fine_trunk_points = coarse_trunk_points
        logger.warning("[MAST3R DBH] Fine crop yielded too few points, using coarse crop.")

    # ── 5. PCA + Circle Fitting (Pass 2 - Fine/Final) ─────────────────────────
    rough_z_pass2 = fine_trunk_points[:, rough_axis_idx]
    rough_z_min_pass2 = np.percentile(rough_z_pass2, 5)
    rough_z_max_pass2 = np.percentile(rough_z_pass2, 95)
    rough_height_pass2 = rough_z_max_pass2 - rough_z_min_pass2

    mid_trunk_mask_pass2 = (rough_z_pass2 >= rough_z_min_pass2 + rough_height_pass2 * 0.15) & (rough_z_pass2 <= rough_z_min_pass2 + rough_height_pass2 * 0.60)
    pca_pts_pass2 = fine_trunk_points[mid_trunk_mask_pass2]
    if len(pca_pts_pass2) < 15:
        pca_pts_pass2 = fine_trunk_points

    if len(pca_pts_pass2) > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pca_pts_pass2), size=10000, replace=False)
        pca_pts_pass2 = pca_pts_pass2[idx]

    trunk_mean_pass2 = pca_pts_pass2.mean(axis=0)
    centered_pass2 = pca_pts_pass2 - trunk_mean_pass2
    cov_pass2 = (centered_pass2.T @ centered_pass2) / max(len(centered_pass2) - 1, 1)
    eigenvalues_pass2, eigenvectors_pass2 = np.linalg.eigh(cov_pass2)
    v_pass2 = eigenvectors_pass2[:, -1]

    if v_pass2[rough_axis_idx] < 0:
        v_pass2 = -v_pass2

    proj_pass2 = np.dot(points, v_pass2)
    h_min_pass2 = float(np.percentile(proj_pass2, 2))
    h_max_pass2 = float(np.percentile(proj_pass2, 98))
    total_h_pass2 = h_max_pass2 - h_min_pass2
    estimated_height_m = float(total_h_pass2 * scale)

    h_target_pass2 = h_min_pass2 + (breast_height / scale)
    if h_target_pass2 >= h_max_pass2 * 0.90:
        h_target_pass2 = h_min_pass2 + total_h_pass2 * 0.30

    if abs(v_pass2[0]) < 0.9:
        ref_pass2 = np.array([1.0, 0.0, 0.0])
    else:
        ref_pass2 = np.array([0.0, 1.0, 0.0])
    u1_pass2 = np.cross(v_pass2, ref_pass2)
    u1_pass2 = u1_pass2 / np.linalg.norm(u1_pass2)
    u2_pass2 = np.cross(v_pass2, u1_pass2)

    proj_trunk_pass2 = np.dot(fine_trunk_points, v_pass2)
    tol_pass2 = base_tol * density_factor
    tol_pass2 = min(tol_pass2, total_h_pass2 * 0.15)
    tol_pass2 = max(tol_pass2, total_h_pass2 * 0.02)

    offsets = [-tol_pass2 * 0.4, 0.0, tol_pass2 * 0.4]
    radii_pass2 = []
    centers_2d_pass2 = []
    slice_points_list_pass2 = []

    for off in offsets:
        h_t = h_target_pass2 + off
        mask = np.abs(proj_trunk_pass2 - h_t) <= tol_pass2
        pts_slice = fine_trunk_points[mask]
        if len(pts_slice) < 5:
            continue
        pts_2d = np.column_stack((np.dot(pts_slice, u1_pass2), np.dot(pts_slice, u2_pass2)))
        xc, yc, R, _ = fit_circle_2d(pts_2d)
        if R is not None and R > 0 and R < fine_crop_radius * 1.5:
            radii_pass2.append(R)
            centers_2d_pass2.append((xc, yc))
            slice_points_list_pass2.append(pts_slice)

    method_used = "MASt3R aligned multi-slice median (refined)"
    if not radii_pass2:
        logger.warning("[MAST3R DBH] Aligned refined multi-slice failed, using fallback on refined trunk points...")
        mask = (proj_trunk_pass2 >= h_min_pass2) & (proj_trunk_pass2 <= h_min_pass2 + total_h_pass2 * 0.5)
        pts_trunk = fine_trunk_points[mask]
        if len(pts_trunk) < 5:
            return {"error": "No points in aligned trunk region after refined crop"}
        pts_2d = np.column_stack((np.dot(pts_trunk, u1_pass2), np.dot(pts_trunk, u2_pass2)))
        xc, yc, R, _ = fit_circle_robust(pts_2d)
        if R is None or R > fine_crop_radius * 2.0:
            R = R_pass1
            xc, yc = 0.0, 0.0
        radii_pass2 = [R]
        centers_2d_pass2 = [(xc, yc)]
        method_used = "MASt3R aligned refined fallback"
        slice_points_all = pts_trunk
    else:
        slice_points_all = np.concatenate(slice_points_list_pass2, axis=0)

    # Subsample if too dense (max 500 points for efficient storage/rendering)
    if len(slice_points_all) > 500:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(slice_points_all), size=500, replace=False)
        slice_points_all = slice_points_all[idx]

    # Medians
    R_final = float(np.median(radii_pass2))
    dbh_m   = R_final * 2.0 * scale
    dbh_cm  = dbh_m * 100.0

    xc_2d = float(np.median([c[0] for c in centers_2d_pass2]))
    yc_2d = float(np.median([c[1] for c in centers_2d_pass2]))
    center_3d = xc_2d * u1_pass2 + yc_2d * u2_pass2 + h_target_pass2 * v_pass2

    # ── 6. Log both passes for comparison ─────────────────────────────────────
    logger.info(f"[MAST3R DBH] Iterative refinement completed:")
    logger.info(f"             Pass 1 Radius: {R_pass1 * scale * 100:.2f} cm (units: {R_pass1:.4f})")
    logger.info(f"             Pass 2 Radius: {R_final * scale * 100:.2f} cm (units: {R_final:.4f})")
    logger.info(f"             Difference: {(R_pass1 - R_final) * scale * 100:.2f} cm")
    logger.info(f"             Pass 1 Direction: {v_pass1}")
    logger.info(f"             Pass 2 Direction: {v_pass2}")

    slice_count = int(slice_points_all.shape[0])
    mean_err_cm = 0.0
    if slice_count >= 5:
        pts_2d = np.column_stack((np.dot(slice_points_all, u1_pass2), np.dot(slice_points_all, u2_pass2)))
        _, _, _, mean_err = fit_circle_robust(pts_2d)
        if mean_err is not None:
            mean_err_cm = float(round(mean_err * scale * 100, 2))

    confidence = "High" if (mean_err_cm <= 3.0) else "Medium"
    if estimated_height_m < 1.0:
        confidence = f"WARNING: Trunk segment captured is only {estimated_height_m:.2f}m tall, insufficient to reach standard breast height (1.3m). DBH measurement may not represent true breast-height diameter - recommend recapturing with more trunk visible in frame"

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
            "dir_x":          float(round(v_pass2[0], 4)),
            "dir_y":          float(round(v_pass2[1], 4)),
            "dir_z":          float(round(v_pass2[2], 4)),
            "radius_units":   float(round(R_final, 4)),
            "h_min":          float(round(h_min_pass2, 4)),
            "h_max":          float(round(h_max_pass2, 4)),
            "h_target":       float(round(h_target_pass2, 4)),
            "scale_factor":   scale,
            "slice_points_3d": [[float(round(p[0], 4)), float(round(p[1], 4)), float(round(p[2], 4))] for p in slice_points_all],
        }
    }

def extract_dbh_with_manual_override(ply_path: str, cx: float, cy: float, cz: float, radius: float,
                                     scale_factor: float = 1.0, breast_height: float = 1.3) -> dict:
    logger.info(f"[MANUAL DBH] Starting manual DBH extraction from {ply_path} with center=({cx}, {cy}, {cz}), radius={radius}")
    scale = load_scale_factor(scale_factor)

    try:
        points = parse_ply_points(ply_path)
    except Exception as exc:
        return {"error": f"Failed to load point cloud: {exc}"}

    if len(points) < 10:
        return {"error": "Point cloud too sparse"}

    # 1. Determine rough vertical axis
    ranges = points.max(axis=0) - points.min(axis=0)
    rough_axis_idx = int(np.argmax(ranges))

    # 2. Filter point cloud ONLY within selection sphere radius around the clicked center
    dist_sq = (points[:, 0] - cx)**2 + (points[:, 1] - cy)**2 + (points[:, 2] - cz)**2
    trunk_mask = dist_sq <= radius**2
    trunk_points = points[trunk_mask]

    if len(trunk_points) < 10:
        return {"error": f"Too few points within the selection radius ({len(trunk_points)} points). Please select a larger radius or a different point."}

    # 3. PCA on manual trunk points to find direction
    rough_z = trunk_points[:, rough_axis_idx]
    rough_z_min = np.percentile(rough_z, 5)
    rough_z_max = np.percentile(rough_z, 95)
    rough_height = rough_z_max - rough_z_min

    mid_trunk_mask = (rough_z >= rough_z_min + rough_height * 0.1) & (rough_z <= rough_z_min + rough_height * 0.9)
    pca_pts = trunk_points[mid_trunk_mask]
    if len(pca_pts) < 10:
        pca_pts = trunk_points

    trunk_mean = pca_pts.mean(axis=0)
    centered = pca_pts - trunk_mean
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    v = eigenvectors[:, -1]

    if v[rough_axis_idx] < 0:
        v = -v

    logger.info(f"[MANUAL DBH] Aligned trunk direction vector: {v}")

    # 4. Project all points to find true height
    proj = np.dot(points, v)
    h_min = float(np.percentile(proj, 2))
    h_max = float(np.percentile(proj, 98))
    total_h = h_max - h_min
    estimated_height_m = float(total_h * scale)

    # 5. Set target height to the projection of clicked center along v
    h_target = float(np.dot(np.array([cx, cy, cz]), v))

    # 6. Fit circle at slices around h_target
    if abs(v[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    u1 = np.cross(v, ref)
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(v, u1)

    proj_trunk = np.dot(trunk_points, v)
    
    base_tol = 0.08 / scale
    tol = min(base_tol, total_h * 0.1)
    tol = max(tol, 0.02)

    offsets = [-tol * 0.3, 0.0, tol * 0.3]
    radii = []
    centers_2d = []
    slice_points_list = []

    for off in offsets:
        h_t = h_target + off
        mask = np.abs(proj_trunk - h_t) <= tol
        pts_slice = trunk_points[mask]
        if len(pts_slice) < 4:
            continue
        pts_2d = np.column_stack((np.dot(pts_slice, u1), np.dot(pts_slice, u2)))
        xc_slice, yc_slice, R, err = fit_circle_2d(pts_2d)
        if R is not None and R > 0 and R < radius * 1.5:
            radii.append(R)
            centers_2d.append((xc_slice, yc_slice))
            slice_points_list.append(pts_slice)

    method_used = "Manual override trunk select"
    if not radii:
        logger.warning("[MANUAL DBH] Aligned slice failed, using fallback on selected trunk points...")
        pts_2d = np.column_stack((np.dot(trunk_points, u1), np.dot(trunk_points, u2)))
        xc_slice, yc_slice, R, _ = fit_circle_robust(pts_2d)
        if R is None or R > radius * 2.0:
            R = radius
            xc_slice, yc_slice = 0.0, 0.0
        radii = [R]
        centers_2d = [(xc_slice, yc_slice)]
        slice_points_all = trunk_points
        method_used = "Manual override fallback"
    else:
        slice_points_all = np.concatenate(slice_points_list, axis=0)

    if len(slice_points_all) > 500:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(slice_points_all), size=500, replace=False)
        slice_points_all = slice_points_all[idx]

    R_final = float(np.median(radii))
    dbh_m = R_final * 2.0 * scale
    dbh_cm = dbh_m * 100.0

    xc_2d = float(np.median([c[0] for c in centers_2d]))
    yc_2d = float(np.median([c[1] for c in centers_2d]))

    center_3d = xc_2d * u1 + yc_2d * u2 + h_target * v

    slice_count = int(slice_points_all.shape[0])
    mean_err_cm = 0.0
    if slice_count >= 5:
        pts_2d = np.column_stack((np.dot(slice_points_all, u1), np.dot(slice_points_all, u2)))
        _, _, _, mean_err = fit_circle_robust(pts_2d)
        if mean_err is not None:
            mean_err_cm = float(round(mean_err * scale * 100, 2))

    confidence_note = "Manually verified trunk selection"
    if estimated_height_m < 1.0:
        confidence_note = f"WARNING: Trunk segment captured is only {estimated_height_m:.2f}m tall, insufficient to reach standard breast height (1.3m). DBH measurement may not represent true breast-height diameter - recommend recapturing with more trunk visible in frame"

    return {
        "dbh_cm":             float(round(dbh_cm, 2)),
        "height_m":           float(round(estimated_height_m, 2)),
        "confidence_note":    confidence_note,
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
            "slice_points_3d": [[float(round(p[0], 4)), float(round(p[1], 4)), float(round(p[2], 4))] for p in slice_points_all],
        }
    }

def extract_dbh_with_2d_clicks(ply_path: str, P1: np.ndarray, P2: np.ndarray, scale: float, crop_radius_m: float = 0.25, breast_height_m: float = 1.3) -> dict:
    try:
        points = parse_ply_points(ply_path)
    except Exception as exc:
        return {"error": f"Failed to load point cloud: {exc}"}

    if len(points) < 10:
        return {"error": "Point cloud too sparse"}

    # 1. Determine direction vector v
    v = P2 - P1
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-6:
        return {"error": "P1 and P2 are identical or too close"}
    v = v / v_norm

    # 2. Filter point cloud within cylinder around the axis v
    # w is the vector from P1 to each point Q
    w = points - P1
    # h is the projection length along v
    h_proj = np.dot(w, v)
    # d is the perpendicular distance to the axis v
    d_proj = np.linalg.norm(w - h_proj[:, np.newaxis] * v[np.newaxis, :], axis=-1)

    crop_radius = crop_radius_m / scale
    # We crop points that are within crop_radius of the axis and within the span of P1 and P2 (with some margin)
    margin = 0.2 / scale
    trunk_mask = (d_proj <= crop_radius) & (h_proj >= -margin) & (h_proj <= v_norm + margin)
    trunk_points = points[trunk_mask]

    if len(trunk_points) < 10:
        # Fallback: ignore height constraints, just keep all points within crop_radius of axis
        trunk_mask = d_proj <= crop_radius
        trunk_points = points[trunk_mask]

    if len(trunk_points) < 10:
        return {"error": f"Too few points within the crop cylinder ({len(trunk_points)} points)."}

    # 3. Project all points in points3d.ply to find true height (overall tree height)
    proj_all = np.dot(points, v)
    h_min = float(np.percentile(proj_all, 2))
    h_max = float(np.percentile(proj_all, 98))
    total_h = h_max - h_min
    estimated_height_m = float(total_h * scale)

    # 4. Set target breast height target
    # P1 is the ground, so breast height target is 1.3 meters above P1
    h_target = float(np.dot(P1, v) + breast_height_m / scale)

    # 5. Fit circle at slices around h_target
    if abs(v[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    u1 = np.cross(v, ref)
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(v, u1)

    proj_trunk = np.dot(trunk_points, v)
    
    base_tol = 0.08 / scale
    tol = min(base_tol, total_h * 0.1)
    tol = max(tol, 0.02)

    offsets = [-tol * 0.3, 0.0, tol * 0.3]
    radii = []
    centers_2d = []
    slice_points_list = []

    for off in offsets:
        h_t = h_target + off
        mask = np.abs(proj_trunk - h_t) <= tol
        pts_slice = trunk_points[mask]
        if len(pts_slice) < 4:
            continue
        pts_2d = np.column_stack((np.dot(pts_slice, u1), np.dot(pts_slice, u2)))
        xc_slice, yc_slice, R, err = fit_circle_2d(pts_2d)
        if R is not None and R > 0 and R < crop_radius * 1.5:
            radii.append(R)
            centers_2d.append((xc_slice, yc_slice))
            slice_points_list.append(pts_slice)

    method_used = "Manual override 2D clicks"
    if not radii:
        logger.warning("[MANUAL DBH] Aligned slice failed, using fallback on selected cylinder points...")
        pts_2d = np.column_stack((np.dot(trunk_points, u1), np.dot(trunk_points, u2)))
        xc_slice, yc_slice, R, _ = fit_circle_robust(pts_2d)
        if R is None or R > crop_radius * 2.0:
            R = crop_radius / 2
            xc_slice, yc_slice = 0.0, 0.0
        radii = [R]
        centers_2d = [(xc_slice, yc_slice)]
        slice_points_all = trunk_points
        method_used = "Manual override 2D fallback"
    else:
        slice_points_all = np.concatenate(slice_points_list, axis=0)

    if len(slice_points_all) > 500:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(slice_points_all), size=500, replace=False)
        slice_points_all = slice_points_all[idx]

    R_final = float(np.median(radii))
    dbh_m = R_final * 2.0 * scale
    dbh_cm = dbh_m * 100.0

    xc_2d = float(np.median([c[0] for c in centers_2d]))
    yc_2d = float(np.median([c[1] for c in centers_2d]))

    center_3d = xc_2d * u1 + yc_2d * u2 + h_target * v

    slice_count = int(slice_points_all.shape[0])
    mean_err_cm = 0.0
    if slice_count >= 5:
        pts_2d = np.column_stack((np.dot(slice_points_all, u1), np.dot(slice_points_all, u2)))
        _, _, _, mean_err = fit_circle_robust(pts_2d)
        if mean_err is not None:
            mean_err_cm = float(round(mean_err * scale * 100, 2))

    confidence_note = "Manually corrected"
    if estimated_height_m < 1.0:
        confidence_note = f"WARNING: Trunk segment captured is only {estimated_height_m:.2f}m tall, insufficient to reach standard breast height (1.3m). DBH measurement may not represent true breast-height diameter - recommend recapturing with more trunk visible in frame"

    return {
        "dbh_cm":             float(round(dbh_cm, 2)),
        "height_m":           float(round(estimated_height_m, 2)),
        "confidence_note":    confidence_note,
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
            "slice_points_3d": [[float(round(p[0], 4)), float(round(p[1], 4)), float(round(p[2], 4))] for p in slice_points_all],
        }
    }
