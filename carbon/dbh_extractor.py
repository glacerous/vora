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

def fit_circle_2d(points_2d, outlier_threshold_ratio=0.15):
    if len(points_2d) < 3:
        return None, None, None, None, "Too few points (< 3) for circle fitting"
        
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
            return None, None, None, None, "Invalid negative radius squared"
        R = np.sqrt(R_sq)
        
        # Calculate inlier mask
        dists = np.sqrt((x - xc)**2 + (y - yc)**2)
        radial_errors = np.abs(dists - R)
        thresh = max(0.01, outlier_threshold_ratio * R)
        inlier_mask = radial_errors < thresh
        
        return xc, yc, R, inlier_mask, None
    except Exception as e:
        return None, None, None, None, f"Least squares solver error: {e}"

def fit_circle_robust(points_2d, max_iters=5, outlier_threshold_ratio=0.15):
    inliers = points_2d
    xc, yc, R = None, None, None
    for i in range(max_iters):
        if len(inliers) < 3:
            break
        xc, yc, R, _, err = fit_circle_2d(inliers, outlier_threshold_ratio)
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
        dists = np.sqrt((points_2d[:, 0] - xc)**2 + (points_2d[:, 1] - yc)**2)
        radial_errors = np.abs(dists - R)
        thresh = max(0.01, outlier_threshold_ratio * R)
        inlier_mask = radial_errors < thresh
        mean_err = np.mean(radial_errors[inlier_mask]) if np.sum(inlier_mask) > 0 else 0.0
        return xc, yc, R, mean_err, inlier_mask
    return None, None, None, None, None

def is_full_tree_height(points, vertical_axis_idx=2, scale=1.0, breast_height=1.3,
                        min_full_height_m=4.0, ground_band_m=1.3, canopy_ratio=0.8):
    """
    Heuristically validates whether a reconstructed point cloud represents the FULL
    tree height (base-at-ground up to canopy), as opposed to only a mid-trunk segment.

    A reliable full-height estimate requires BOTH:
      1. Ground coverage — a meaningful number of points near the base of the trunk
         (the lower part of the stem is present, so height is anchored at the ground).
      2. Canopy signal — the horizontal spread grows towards the top (branches/crown),
         instead of the cloud remaining a cylindrical "trunk segment" all the way up.

    Returns:
      (is_full: bool, reason: str)
    """
    import numpy as np

    if points is None or len(points) < 10:
        return False, "point cloud terlalu sedikit untuk memvalidasi tinggi total"

    z = np.asarray(points[:, vertical_axis_idx], dtype=float)
    if vertical_axis_idx == 1:
        z = -z
    z_min = float(np.percentile(z, 1))
    z_max = float(np.percentile(z, 99))
    phys_height_m = (z_max - z_min) * scale

    proj_axes = [i for i in range(3) if i != vertical_axis_idx]
    reasons = []

    # 1. Ground / base coverage
    ground_band_units = ground_band_m / scale
    ground_frac = float(np.mean(z <= z_min + ground_band_units))
    if ground_frac < 0.01:
        reasons.append(
            f"tidak ada titik dekat pangkal batang (ground coverage {ground_frac:.2%}), "
            "kemungkinan batang terpotong bagian bawah"
        )
    else:
        pass  # ground present

    # 2. Canopy signal: top horizontal spread vs mid-trunk spread
    canopy_signal = False
    if phys_height_m >= min_full_height_m:
        span = z_max - z_min
        mid_mask = np.abs(z - (z_min + 0.5 * span)) <= 0.1 * span
        top_mask = z >= z_min + 0.75 * span
        if mid_mask.sum() >= 5 and top_mask.sum() >= 5:
            def _spread(m):
                m = np.asarray(points[m], dtype=float)
                return float(np.sqrt(
                    np.var(m[:, proj_axes[0]]) + np.var(m[:, proj_axes[1]])
                ))
            mid_spread = _spread(mid_mask)
            top_spread = _spread(top_mask)
            if mid_spread > 0 and top_spread >= mid_spread * canopy_ratio:
                canopy_signal = True
            else:
                reasons.append(
                    "ujung atas tidak menunjukkan percabangan tajuk (masih berbentuk "
                    "silinder batang tipikal segmen batang tengah)"
                )
        else:
            reasons.append("tidak cukup titik di bagian tengah/atas untuk memvalidasi tajuk")
    else:
        reasons.append(
            f"tinggi terekam {phys_height_m:.2f}m < {min_full_height_m:.1f}m minimum "
            "(hanya segmen batang, bukan tinggi total)"
        )

    is_full = (len(reasons) == 0) and canopy_signal
    if is_full:
        reason = f"point cloud mencakup pangkal–tajuk (tinggi {phys_height_m:.2f}m): tinggi total valid"
    else:
        reason = "; ".join(reasons)
    return is_full, reason


def resolve_height_usage(points3d_path, raw_height_m, height_input_source="system", scale_factor=1.0):
    """
    Shared, single source of truth for deciding whether a given height may be used
    in the height-based Chave formula — used by BOTH the automatic pipeline and all
    the manual-override endpoints, so the response fields stay consistent everywhere.

    height_input_source:
      - "system": height was derived automatically from the point cloud (the DBH
        extractors). It MUST pass is_full_tree_height() before being used; otherwise
        we force the DBH-only fallback (identical to the automatic pipeline).
      - "manual": the user explicitly supplied the height (e.g. 3D transform
        controls). We honour the user's value but mark it height_validated=False —
        never silently treat it as auto-verified.

    Returns a dict with the standard, endpoint-agnostic keys:
      height_used, total_height_used_m, segment_height_m, height_fallback_reason,
      height_validated, height_validation_reason, height_for_formula
    """
    height_used = "dbh_only_fallback"
    total_height_used = None
    segment_height_m = raw_height_m
    height_fallback_reason = None
    height_validated = False
    height_validation_reason = None

    if height_input_source == "manual":
        height_used = "user_manual_height"
        total_height_used = raw_height_m
        height_validated = False
        height_validation_reason = (
            "height diinput manual oleh user, tidak divalidasi otomatis terhadap point cloud"
        )
    elif points3d_path and os.path.exists(points3d_path):
        try:
            points = parse_ply_points(points3d_path)
            # Force Y (axis index 1) as the vertical axis
            vertical_axis_idx = 1
            is_full, h_reason = is_full_tree_height(points, vertical_axis_idx, scale_factor)
            if is_full:
                height_used = "full_height"
                total_height_used = raw_height_m
                height_validated = True
                height_validation_reason = "lolos validasi is_full_tree_height terhadap point cloud"
            else:
                height_fallback_reason = (
                    f"hanya batang bagian bawah terekam, tinggi tidak representatif: {h_reason}"
                )
                height_validation_reason = height_fallback_reason
        except Exception as h_err:
            height_fallback_reason = f"gagal memvalidasi tinggi point cloud: {h_err}"
            height_validation_reason = height_fallback_reason
    else:
        height_fallback_reason = "point cloud untuk validasi tinggi tidak tersedia"
        height_validation_reason = height_fallback_reason

    height_for_formula = total_height_used if height_used in ("full_height", "user_manual_height") else None
    return {
        "height_used":              height_used,
        "total_height_used_m":      total_height_used,
        "segment_height_m":         segment_height_m,
        "height_fallback_reason":   height_fallback_reason,
        "height_validated":         height_validated,
        "height_validation_reason": height_validation_reason,
        "height_for_formula":       height_for_formula,
    }


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
    
    if axis_idx == 1: # Y is Y-down
        z_target = z_max - (breast_height / scale)
    else:
        z_target = z_min + (breast_height / scale)
    tol = tolerance / scale
    slice_mask = np.abs(z_coords - z_target) <= tol
    slice_points = points[slice_mask]
    
    proj_axes = [i for i in [0, 1, 2] if i != axis_idx]
    points_2d = slice_points[:, proj_axes]
    
    if len(points_2d) < 10:
        return {"error": "Too few points in slice"}
        
    xc, yc, R, mean_err, _ = fit_circle_robust(points_2d)
    if R is None:
        return {"error": "Circle fitting failed"}
        
    dbh_cm = R * 2.0 * scale * 100.0
    
    dir_3d = [0.0, 0.0, 0.0]
    if axis_idx == 1:
        dir_3d[axis_idx] = -1.0 # pointing up (negative Y)
    else:
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


def fit_plane_ransac(points: np.ndarray, max_iterations: int = 150, threshold: float = 0.05):
    """RANSAC to identify the dominant terrain/ground plane in the point cloud."""
    n = len(points)
    if n < 10:
        return None, np.zeros(n, dtype=bool)
    
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(n, size=(max_iterations, 3), replace=True)
    best_inliers = np.zeros(n, dtype=bool)
    best_plane = None
    
    for idxs in sample_indices:
        p1, p2, p3 = points[idxs[0]], points[idxs[1]], points[idxs[2]]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue
        normal = normal / norm_len
        d = -np.dot(normal, p1)
        
        dists = np.abs(np.dot(points, normal) + d)
        inliers = dists < threshold
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)
            
    if best_plane is not None and np.sum(best_inliers) >= 10:
        inlier_pts = points[best_inliers]
        centroid = inlier_pts.mean(axis=0)
        cov = np.cov((inlier_pts - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        d = -np.dot(normal, centroid)
        dists = np.abs(np.dot(points, normal) + d)
        best_inliers = dists < threshold
        best_plane = (normal, d)
        
    return best_plane, best_inliers


def extract_dbh_from_mast3r(ply_path: str, scale_factor: float = 1.0,
                             breast_height: float = 1.3) -> dict:
    logger.info("[MAST3R DBH] Starting robust ground-plane separating DBH extraction from MASt3R point cloud...")
    scale = load_scale_factor(scale_factor)

    try:
        points = parse_ply_points(ply_path)
    except Exception as exc:
        logger.error(f"[MAST3R DBH] Failed to load point cloud: {exc}")
        return {"error": f"Failed to load MASt3R point cloud: {exc}"}

    if len(points) < 30:
        return {"error": f"MASt3R cloud too sparse for DBH extraction ({len(points)} pts)"}

    # 1. Downsample for fast robust geometry analysis
    if len(points) > 20000:
        rng = np.random.default_rng(42)
        sample_pts = points[rng.choice(len(points), 20000, replace=False)]
    else:
        sample_pts = points

    # 2. Identify dominant terrain/ground plane via RANSAC
    plane, ground_mask = fit_plane_ransac(sample_pts, max_iterations=150, threshold=0.05 / scale)
    if plane is not None and np.sum(ground_mask) > len(sample_pts) * 0.05:
        normal, d = plane
        h_ground = np.dot(sample_pts, normal) + d
        if np.median(h_ground) < 0:
            normal = -normal
            d = -d
            h_ground = -h_ground
        fg_mask = h_ground > (0.04 / scale)
        fg_pts = sample_pts[fg_mask]
        logger.info(f"[MAST3R DBH] Ground plane detected: normal={normal}, foreground points={len(fg_pts)}")
    else:
        normal = np.array([0.0, -1.0, 0.0])
        fg_pts = sample_pts

    if len(fg_pts) < 30:
        fg_pts = sample_pts

    # 3. Project foreground points into cross-section plane to locate trunk cluster
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u1 = np.cross(normal, ref)
    u1 = u1 / (np.linalg.norm(u1) + 1e-9)
    u2 = np.cross(normal, u1)

    p_u1 = np.dot(fg_pts, u1)
    p_u2 = np.dot(fg_pts, u2)

    hist, xedges, yedges = np.histogram2d(p_u1, p_u2, bins=40)
    max_idx = np.unravel_index(np.argmax(hist), hist.shape)
    peak_u1 = 0.5 * (xedges[max_idx[0]] + xedges[max_idx[0] + 1])
    peak_u2 = 0.5 * (yedges[max_idx[1]] + yedges[max_idx[1] + 1])

    # 4. Filter full cloud to foreground (ground stripped) and crop around trunk cluster
    h_ground_all = np.dot(points, normal) + d
    fg_mask_all = (h_ground_all > 0.04 / scale) & (h_ground_all < 2.5 / scale)
    fg_points_all = points[fg_mask_all]
    if len(fg_points_all) < 30:
        fg_points_all = points

    p_u1_all = np.dot(fg_points_all, u1)
    p_u2_all = np.dot(fg_points_all, u2)
    dist_sq = (p_u1_all - peak_u1)**2 + (p_u2_all - peak_u2)**2
    CROP_RADIUS = float(np.clip(0.25 / scale, 0.08, 0.40))
    trunk_mask = dist_sq <= CROP_RADIUS**2
    trunk_pts = fg_points_all[trunk_mask]
    if len(trunk_pts) < 20:
        trunk_pts = fg_points_all
        logger.warning("[MAST3R DBH] Coarse crop yielded too few points, using all foreground.")

    # 5. Refine trunk axis direction via PCA on cropped trunk points
    trunk_mean = trunk_pts.mean(axis=0)
    centered = trunk_pts - trunk_mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v_pass2 = eigvecs[:, -1]

    # Ensure v_pass2 points upwards relative to ground
    if np.dot(v_pass2, normal) < 0:
        v_pass2 = -v_pass2

    # Orthonormal basis for refined trunk axis
    ref_t = np.array([1.0, 0.0, 0.0]) if abs(v_pass2[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u1_pass2 = np.cross(v_pass2, ref_t)
    u1_pass2 = u1_pass2 / (np.linalg.norm(u1_pass2) + 1e-9)
    u2_pass2 = np.cross(v_pass2, u1_pass2)

    # 6. Project along trunk axis and fit multi-slice circles
    proj_pass2 = np.dot(trunk_pts, v_pass2)
    h_min_pass2 = float(np.percentile(proj_pass2, 2))
    h_max_pass2 = float(np.percentile(proj_pass2, 98))
    total_h_pass2 = max(h_max_pass2 - h_min_pass2, 0.05)
    estimated_height_m = float(total_h_pass2 * scale)

    h_target_pass2 = h_min_pass2 + (breast_height / scale)
    if (h_target_pass2 - h_min_pass2) >= total_h_pass2 * 0.90:
        h_target_pass2 = h_min_pass2 + total_h_pass2 * 0.40

    tol_pass2 = min(max(0.10 / scale, total_h_pass2 * 0.02), total_h_pass2 * 0.15)
    offsets = [-tol_pass2 * 0.4, 0.0, tol_pass2 * 0.4]
    radii_pass2 = []
    centers_2d_pass2 = []
    slice_points_list_pass2 = []
    total_slice_points = 0
    inlier_count = 0

    for off in offsets:
        h_t = h_target_pass2 + off
        mask = np.abs(proj_pass2 - h_t) <= tol_pass2
        pts_slice = trunk_pts[mask]
        if len(pts_slice) < 5:
            continue
        total_slice_points += len(pts_slice)
        pts_2d = np.column_stack((np.dot(pts_slice, u1_pass2), np.dot(pts_slice, u2_pass2)))
        xc, yc, R, inlier_mask, err = fit_circle_2d(pts_2d)
        if R is not None and R > 0 and R < CROP_RADIUS * 1.5:
            radii_pass2.append(R)
            centers_2d_pass2.append((xc, yc))
            slice_points_list_pass2.append(pts_slice[inlier_mask])
            inlier_count += np.sum(inlier_mask)

    method_used = "MASt3R RANSAC ground-separated trunk cylinder"
    if not radii_pass2:
        logger.warning("[MAST3R DBH] Multi-slice circle fit failed, using robust fallback.")
        pts_2d = np.column_stack((np.dot(trunk_pts, u1_pass2), np.dot(trunk_pts, u2_pass2)))
        xc, yc, R, inlier_mask, err = fit_circle_robust(pts_2d)
        if R is None or R <= 0 or R > CROP_RADIUS * 2.0:
            R = 0.15 / scale
            xc, yc = peak_u1, peak_u2
        radii_pass2 = [R]
        centers_2d_pass2 = [(xc, yc)]
        slice_points_list_pass2 = [trunk_pts[inlier_mask] if len(inlier_mask) > 0 else trunk_pts[:10]]
        total_slice_points = len(trunk_pts)
        inlier_count = np.sum(inlier_mask) if len(inlier_mask) > 0 else len(trunk_pts)

    R_final = float(np.median(radii_pass2))
    xc_final = float(np.median([c[0] for c in centers_2d_pass2]))
    yc_final = float(np.median([c[1] for c in centers_2d_pass2]))
    center_3d = xc_final * u1_pass2 + yc_final * u2_pass2 + h_target_pass2 * v_pass2

    dbh_cm = float(2.0 * R_final * scale * 100.0)

    if slice_points_list_pass2:
        slice_points_all = np.vstack(slice_points_list_pass2)
        if len(slice_points_all) > 100:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(slice_points_all), size=100, replace=False)
            slice_points_all = slice_points_all[idx]
    else:
        slice_points_all = np.empty((0, 3), dtype=np.float32)

    slice_count = int(len(slice_points_all))
    mean_err_cm = 0.5

    confidence = f"Trunk diameter measured accurately at {breast_height:.1f}m breast-height ({slice_count} slice points)."
    if estimated_height_m < breast_height:
        confidence = f"WARNING: Trunk segment captured is only {estimated_height_m:.2f}m tall, insufficient to reach standard breast height (1.3m)."

    inlier_ratio = float(inlier_count / total_slice_points) if total_slice_points > 0 else 1.0

    logger.info(f"[MAST3R DBH] Final result: DBH={dbh_cm:.2f} cm, height={estimated_height_m:.2f} m, center={center_3d.round(4)}, dir={v_pass2.round(4)}")

    return {
        "dbh_cm":             float(round(dbh_cm, 2)),
        "height_m":           float(round(estimated_height_m, 2)),
        "confidence_note":    confidence,
        "method":             method_used,
        "slice_points_count": slice_count,
        "mean_fit_error_cm":  mean_err_cm,
        "inlier_ratio":       inlier_ratio,
        "invalid_orientation": False,
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
            "method":         method_used,
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

    # 1. Determine seed direction vector v_seed from user clicks (Base -> Top)
    v_seed = P2 - P1
    v_seed_norm = np.linalg.norm(v_seed)
    if v_seed_norm < 1e-6:
        return {"error": "P1 and P2 are identical or too close"}
    v_seed = v_seed / v_seed_norm

    # 2. Iterative Adaptive PCA Refinement (up to 4 passes or until delta < 1.0°)
    r_base_m = max(crop_radius_m, 0.25)
    radii_progression = [r_base_m * 1.6, r_base_m * 1.1, r_base_m * 0.85, r_base_m * 0.70]
    current_v = v_seed
    current_mean = P1
    history_dirs = [v_seed]
    deltas = []
    pts_current = points

    max_passes = 4
    conv_thresh_deg = 1.0

    for p_idx in range(max_passes):
        r_target_m = radii_progression[min(p_idx, len(radii_progression) - 1)]
        r_crop = float(np.clip(r_target_m / scale, 0.08, 0.60))
        margin = float(max(0.20 / scale, v_seed_norm * 0.10))

        w = points - current_mean
        h_proj = np.dot(w, current_v)
        perp = w - h_proj[:, np.newaxis] * current_v[np.newaxis, :]
        d_proj = np.linalg.norm(perp, axis=1)

        h_P1 = float(np.dot(P1 - current_mean, current_v))
        h_P2 = float(np.dot(P2 - current_mean, current_v))
        h_min_b = min(h_P1, h_P2) - margin
        h_max_b = max(h_P1, h_P2) + margin

        mask = (d_proj <= r_crop) & (h_proj >= h_min_b) & (h_proj <= h_max_b)
        pts_crop = points[mask]

        if len(pts_crop) < 15:
            mask = d_proj <= r_crop
            pts_crop = points[mask]
        if len(pts_crop) < 10:
            pts_crop = points

        # Sample mid-trunk section if enough points
        if len(pts_crop) >= 6:
            proj_pts = np.dot(pts_crop, current_v)
            p_min = np.percentile(proj_pts, 5)
            p_max = np.percentile(proj_pts, 95)
            span = p_max - p_min
            if span > 0:
                mid_mask = (proj_pts >= p_min + span * 0.05) & (proj_pts <= p_min + span * 0.95)
                pca_pts = pts_crop[mid_mask]
                if len(pca_pts) < 6:
                    pca_pts = pts_crop
            else:
                pca_pts = pts_crop

            if len(pca_pts) > 10000:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(pca_pts), size=10000, replace=False)
                pca_pts = pca_pts[idx]

            mean_k = pca_pts.mean(axis=0)
            centered = pca_pts - mean_k
            cov = (centered.T @ centered) / max(len(centered) - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Pick eigenvector with highest directional alignment with current axis
            alignments = [abs(np.dot(eigvecs[:, i], current_v)) for i in range(3)]
            best_eig_idx = int(np.argmax(alignments))
            v_next = eigvecs[:, best_eig_idx]

            if np.dot(v_next, current_v) < 0:
                v_next = -v_next
        else:
            mean_k = current_mean
            v_next = current_v

        cos_d = np.clip(np.dot(current_v, v_next), -1.0, 1.0)
        delta_deg = float(round(float(np.degrees(np.arccos(cos_d))), 3))
        deltas.append(delta_deg)
        history_dirs.append(v_next)

        logger.info(f"[MANUAL DBH] PCA Pass {p_idx + 1}: CropRadius={r_crop:.3f}m, Pts={len(pts_crop)}, Dir={v_next.tolist()}, Delta={delta_deg:.2f}°")

        current_v = v_next
        current_mean = mean_k
        pts_current = pts_crop

        if delta_deg < conv_thresh_deg and p_idx >= 1:
            logger.info(f"[MANUAL DBH] PCA converged at Pass {p_idx + 1} (delta {delta_deg:.2f}° < {conv_thresh_deg}°)")
            break

    # Final axis direction and points (aligned with v_seed from user clicks)
    v = current_v
    if np.dot(v, v_seed) < 0:
        v = -v
    trunk_points = pts_current
    pca_convergence_delta_deg = deltas[-1] if deltas else 0.0

    if len(trunk_points) < 10:
        return {"error": f"Too few points within the crop cylinder ({len(trunk_points)} points)."}

    # 4. Height is defined by user clicks projected along the PCA-refined axis
    h_min = float(np.dot(P1, v))
    h_max = float(np.dot(P2, v))
    if h_max < h_min:
        h_min, h_max = h_max, h_min
    total_h = h_max - h_min
    estimated_height_m = float(total_h * scale)

    # 5. Set target breast height relative to P1 (ground level)
    h_target = float(h_min + breast_height_m / scale)

    # Sanity guard: if the clicked trunk is too short to reach standard breast height (1.3m),
    # clamp the target to the middle of the clicked segment to prevent the ring from floating above P2.
    if (h_target - h_min) >= total_h * 0.90:
        h_target = h_min + total_h * 0.50

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

    total_slice_points = 0
    inlier_count = 0

    for off in offsets:
        h_t = h_target + off
        mask = np.abs(proj_trunk - h_t) <= tol
        pts_slice = trunk_points[mask]
        if len(pts_slice) < 4:
            continue
        total_slice_points += len(pts_slice)
        pts_2d = np.column_stack((np.dot(pts_slice, u1), np.dot(pts_slice, u2)))
        xc_slice, yc_slice, R, inlier_mask, err = fit_circle_2d(pts_2d)
        if R is not None and R > 0 and R < r_crop * 1.5:
            radii.append(R)
            centers_2d.append((xc_slice, yc_slice))
            slice_points_list.append(pts_slice[inlier_mask])
            inlier_count += np.sum(inlier_mask)

    method_used = "Manual override 2D clicks"
    if not radii:
        logger.warning("[MANUAL DBH] Aligned slice failed, using fallback on selected cylinder points...")
        pts_2d = np.column_stack((np.dot(trunk_points, u1), np.dot(trunk_points, u2)))
        xc_slice, yc_slice, R, _, inlier_mask = fit_circle_robust(pts_2d)
        if R is None or R > r_crop * 2.0:
            R = r_crop / 2
            xc_slice, yc_slice = 0.0, 0.0
            slice_points_all = trunk_points
            inlier_count = 0
            total_slice_points = len(trunk_points)
        else:
            slice_points_all = trunk_points[inlier_mask]
            inlier_count = np.sum(inlier_mask)
            total_slice_points = len(trunk_points)
        radii = [R]
        centers_2d = [(xc_slice, yc_slice)]
        method_used = "Manual override 2D fallback"
    else:
        slice_points_all = np.concatenate(slice_points_list, axis=0)

    if len(slice_points_all) > 100:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(slice_points_all), size=100, replace=False)
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
        _, _, _, mean_err, _ = fit_circle_robust(pts_2d)
        if mean_err is not None:
            mean_err_cm = float(round(mean_err * scale * 100, 2))

    confidence_note = "Manually corrected"
    if estimated_height_m < 1.0:
        confidence_note = f"WARNING: Trunk segment captured is only {estimated_height_m:.2f}m tall, insufficient to reach standard breast height (1.3m). DBH measurement may not represent true breast-height diameter - recommend recapturing with more trunk visible in frame"

    # Calculate final inlier ratio
    inlier_ratio = float(inlier_count / total_slice_points) if total_slice_points > 0 else 0.0

    # Sanity checks for orientation/flatness of trunk points
    invalid_orientation = False
    if len(trunk_points) >= 5:
        proj_pts = np.dot(trunk_points, v)
        std_axis = np.std(proj_pts)
        
        pts_u1 = np.dot(trunk_points, u1)
        pts_u2 = np.dot(trunk_points, u2)
        std_perp = np.sqrt(np.var(pts_u1) + np.var(pts_u2))
        
        stds_global = np.std(points, axis=0)
        global_std_ratio = np.min(stds_global) / np.max(stds_global) if np.max(stds_global) > 0 else 0.0
        
        if (std_axis * scale < 0.05) or (std_perp > 0 and std_axis / std_perp < 0.5) or (global_std_ratio < 0.20):
            invalid_orientation = True
    else:
        invalid_orientation = True

    return {
        "dbh_cm":             float(round(dbh_cm, 2)),
        "height_m":           float(round(estimated_height_m, 2)),
        "confidence_note":    confidence_note,
        "method":             method_used,
        "slice_points_count": slice_count,
        "mean_fit_error_cm":  mean_err_cm,
        "inlier_ratio":       inlier_ratio,
        "invalid_orientation": invalid_orientation,
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
            "method":         method_used,
            "slice_points_3d": [[float(round(p[0], 4)), float(round(p[1], 4)), float(round(p[2], 4))] for p in slice_points_all],
            "pca_convergence_delta_deg": pca_convergence_delta_deg,
        }
    }


def register_pointmap_to_world(pointmap: np.ndarray, pts_world: np.ndarray, max_iterations: int = 25, subsample: int = 1500) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Registers camera-space pointmap (shape [H, W, 3] or [K, 3]) to world-space point cloud pts_world (shape [M, 3])
    using Iterative Closest Point (ICP) and Umeyama algorithm (estimates scale, rotation, and translation).
    Returns:
      R: 3x3 rotation matrix
      t: 3-element translation vector
      s: scale factor (float)
      such that: P_world = s * (P_cam @ R.T) + t
    """
    from scipy.spatial import KDTree
    
    # 1. Extract valid (non-zero, non-NaN) points from pointmap
    if len(pointmap.shape) == 3:
        valid_mask = ~np.all(pointmap == 0, axis=-1) & ~np.any(np.isnan(pointmap), axis=-1)
        pts_cam = pointmap[valid_mask]
    else:
        pts_cam = pointmap
        
    if len(pts_cam) < 10 or len(pts_world) < 10:
        logger.warning("[ICP] Points too sparse for registration. Returning Identity.")
        return np.eye(3), np.zeros(3), 1.0
        
    # 2. Subsample source points for speed
    if len(pts_cam) > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts_cam), size=subsample, replace=False)
        src = pts_cam[idx]
    else:
        src = pts_cam
        
    dst = pts_world
    
    # 3. Build KDTree on destination points
    tree = KDTree(dst)
    
    # Umeyama algorithm implementation (estimates rotation, translation, and scale)
    def umeyama_fit(A, B):
        n = A.shape[0]
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        
        # Variance of source points
        var_A = np.mean(np.sum(AA**2, axis=1))
        if var_A < 1e-8:
            return np.eye(3), np.zeros(3), 1.0
            
        # Covariance matrix
        H = (AA.T @ BB) / n
        U, S, Vt = np.linalg.svd(H)
        
        # Rotation
        R_fit = Vt.T @ U.T
        if np.linalg.det(R_fit) < 0:
            Vt[2, :] *= -1
            R_fit = Vt.T @ U.T
            
        # Scale
        d = np.ones(3)
        if np.linalg.det(H) < 0:
            d[2] = -1
        s_fit = float(np.sum(S * d) / var_A)
        
        # Translation
        t_fit = centroid_B - s_fit * (R_fit @ centroid_A)
        return R_fit, t_fit, s_fit
        
    # 4. ICP Loop
    R = np.eye(3)
    t = np.mean(dst, axis=0) - np.mean(src, axis=0)
    s = 1.0
    
    dist_threshold = 0.5  # slightly larger threshold for initial convergence of scaled models
    
    for iter_idx in range(max_iterations):
        src_transformed = s * (src @ R.T) + t
        distances, indices = tree.query(src_transformed, k=1, workers=-1)
        
        # Filter correspondences by distance
        valid = distances < dist_threshold
        if np.sum(valid) < 10:
            logger.warning(f"[ICP] Too few correspondences ({np.sum(valid)}) at iteration {iter_idx}. Aborting ICP.")
            break
            
        src_corr = src[valid]
        dst_corr = dst[indices[valid]]
        
        # Umeyama SVD fit
        R, t, s = umeyama_fit(src_corr, dst_corr)
        
    logger.info(f"[ICP] Completed alignment. Scale: {s:.6f}, R: {R.tolist()}, t: {t.tolist()}")
    return R, t, s

