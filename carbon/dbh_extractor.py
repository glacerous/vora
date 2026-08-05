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
    # ── 1. Force Y as the vertical axis (axis index 1) ───────────────────────────
    rough_axis_idx = 1
    proj_axes = [0, 2]
    
    # ── 2. Stage 1: Rough Horizontal Peak using entire cloud ──────────────────
    h1_all = points[:, proj_axes[0]]
    h2_all = points[:, proj_axes[1]]
    hist, xedges, yedges = np.histogram2d(h1_all, h2_all, bins=30)
    max_idx = np.unravel_index(np.argmax(hist), hist.shape)
    rough_peak_h1 = 0.5 * (xedges[max_idx[0]] + xedges[max_idx[0] + 1])
    rough_peak_h2 = 0.5 * (yedges[max_idx[1]] + yedges[max_idx[1] + 1])
    
    # Rough crop to remove background (2.2 meters radius in real world)
    ROUGH_CROP_RADIUS = 2.2 / scale
    dist_sq_rough = (points[:, proj_axes[0]] - rough_peak_h1)**2 + (points[:, proj_axes[1]] - rough_peak_h2)**2
    rough_cropped = points[dist_sq_rough <= ROUGH_CROP_RADIUS**2]
    if len(rough_cropped) < 30:
        rough_cropped = points
        
    # ── 3. Stage 2: Refined Peak using lower 35% of rough cropped points ─────
    rough_y = rough_cropped[:, rough_axis_idx]
    y_min = np.percentile(rough_y, 1)
    y_max = np.percentile(rough_y, 99)
    y_height = y_max - y_min
    
    # Lower 35% height in Y-down convention: Y is close to y_max.
    lower_mask = rough_y >= (y_max - y_height * 0.35)
    lower_points = rough_cropped[lower_mask]
    if len(lower_points) < 20:
        lower_points = rough_cropped
        
    h1_lower = lower_points[:, proj_axes[0]]
    h2_lower = lower_points[:, proj_axes[1]]
    hist_refined, xedges_ref, yedges_ref = np.histogram2d(h1_lower, h2_lower, bins=30)
    max_idx_ref = np.unravel_index(np.argmax(hist_refined), hist_refined.shape)
    peak_h1 = 0.5 * (xedges_ref[max_idx_ref[0]] + xedges_ref[max_idx_ref[0] + 1])
    peak_h2 = 0.5 * (yedges_ref[max_idx_ref[1]] + yedges_ref[max_idx_ref[1] + 1])

    # Scale-adaptive coarse crop: ~0.40 m in real-world units, capped at 0.60 PLY units
    CROP_RADIUS_TARGET_M = 0.40          # generous initial capture (one-sided)
    CROP_RADIUS = float(np.clip(CROP_RADIUS_TARGET_M / scale, 0.10, 0.60))
    logger.info(f"[MAST3R DBH] Adaptive CROP_RADIUS = {CROP_RADIUS:.4f} PLY units "
                f"({CROP_RADIUS * scale * 100:.1f} cm real-world, scale={scale})")

    # Use the full points list for the crop distance calculation
    dist_sq = (points[:, proj_axes[0]] - peak_h1)**2 + (points[:, proj_axes[1]] - peak_h2)**2
    coarse_trunk_mask = dist_sq <= CROP_RADIUS**2
    coarse_trunk_points = points[coarse_trunk_mask]
    
    if len(coarse_trunk_points) < 20:
        coarse_trunk_points = points
        logger.warning("[MAST3R DBH] Coarse crop yielded too few points, using full cloud.")

    # ── 2b. Secondary radial rejection ────────────────────────────────────────
    # Remove points whose horizontal distance from the peak exceeds the expected
    # maximum trunk radius (~0.35 m = 70 cm DBH, an extremely large tree).
    # This strips soil / root / debris that slipped into the coarse crop.
    MAX_TRUNK_RADIUS_M = 0.35           # one-sided; DBH ≤ 70 cm catches 99%+ of trees
    max_trunk_radius_ply = float(np.clip(MAX_TRUNK_RADIUS_M / scale, 0.05, CROP_RADIUS))
    horiz_dist = np.sqrt((coarse_trunk_points[:, proj_axes[0]] - peak_h1)**2 +
                         (coarse_trunk_points[:, proj_axes[1]] - peak_h2)**2)
    inlier_radial_mask = horiz_dist <= max_trunk_radius_ply
    if inlier_radial_mask.sum() >= 20:
        coarse_trunk_points = coarse_trunk_points[inlier_radial_mask]
        logger.info(f"[MAST3R DBH] Radial rejection kept {inlier_radial_mask.sum()} / "
                    f"{len(inlier_radial_mask)} coarse points "
                    f"(max_r={max_trunk_radius_ply:.4f} PLY = {MAX_TRUNK_RADIUS_M*100:.0f} cm)")
    else:
        logger.warning("[MAST3R DBH] Radial rejection too aggressive, keeping original coarse crop.")


    # ── 3. PCA + Circle Fitting (Pass 1 - Coarse) ───────────────────────    # Sample mid-trunk section of coarse cropped points (Y-down vs other convention)
    rough_z = coarse_trunk_points[:, rough_axis_idx]
    rough_z_min = np.percentile(rough_z, 5)
    rough_z_max = np.percentile(rough_z, 95)
    rough_height = rough_z_max - rough_z_min
    if rough_axis_idx == 1:
        mid_trunk_mask = (rough_z <= rough_z_max - rough_height * 0.15) & (rough_z >= rough_z_max - rough_height * 0.60)
    else:
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
    
    # Ensure vertical component points upwards (negative Y for Y-down, positive for Z-up)
    if rough_axis_idx == 1:
        if v_pass1[1] > 0:
            v_pass1 = -v_pass1
    else:
        if v_pass1[rough_axis_idx] < 0:
            v_pass1 = -v_pass1

    proj_pass1 = np.dot(coarse_trunk_points, v_pass1)
    h_min_pass1 = float(np.percentile(proj_pass1, 2))
    h_max_pass1 = float(np.percentile(proj_pass1, 98))
    total_h_pass1 = h_max_pass1 - h_min_pass1
    
    h_target_pass1 = h_min_pass1 + (breast_height / scale)
    if (h_target_pass1 - h_min_pass1) >= total_h_pass1 * 0.90:
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
        xc, yc, R, _, _ = fit_circle_2d(pts_2d)
        if R is not None and R > 0 and R < max_trunk_radius_ply * 1.5:
            radii_pass1.append(R)
            centers_2d_pass1.append((xc, yc))

    if not radii_pass1:
        mask = (proj_trunk_pass1 >= h_min_pass1) & (proj_trunk_pass1 <= h_min_pass1 + total_h_pass1 * 0.5)
        pts_trunk = coarse_trunk_points[mask]
        if len(pts_trunk) >= 5:
            pts_2d = np.column_stack((np.dot(pts_trunk, u1_pass1), np.dot(pts_trunk, u2_pass1)))
            xc, yc, R, _, _ = fit_circle_robust(pts_2d)
            if R is None or R > max_trunk_radius_ply * 2.0:
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

    if rough_axis_idx == 1:
        mid_trunk_mask_pass2 = (rough_z_pass2 <= rough_z_max_pass2 - rough_height_pass2 * 0.15) & (rough_z_pass2 >= rough_z_max_pass2 - rough_height_pass2 * 0.60)
    else:
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

    # Ensure vertical component points upwards (negative Y for Y-down, positive for Z-up)
    if rough_axis_idx == 1:
        if v_pass2[1] > 0:
            v_pass2 = -v_pass2
    else:
        if v_pass2[rough_axis_idx] < 0:
            v_pass2 = -v_pass2

    proj_pass2 = np.dot(fine_trunk_points, v_pass2)
    h_min_pass2 = float(np.percentile(proj_pass2, 2))
    h_max_pass2 = float(np.percentile(proj_pass2, 98))
    total_h_pass2 = h_max_pass2 - h_min_pass2
    estimated_height_m = float(total_h_pass2 * scale)

    h_target_pass2 = h_min_pass2 + (breast_height / scale)
    if (h_target_pass2 - h_min_pass2) >= total_h_pass2 * 0.90:
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

    total_slice_points = 0
    inlier_count = 0

    for off in offsets:
        h_t = h_target_pass2 + off
        mask = np.abs(proj_trunk_pass2 - h_t) <= tol_pass2
        pts_slice = fine_trunk_points[mask]
        if len(pts_slice) < 5:
            continue
        total_slice_points += len(pts_slice)
        pts_2d = np.column_stack((np.dot(pts_slice, u1_pass2), np.dot(pts_slice, u2_pass2)))
        xc, yc, R, inlier_mask, err = fit_circle_2d(pts_2d)
        if R is not None and R > 0 and R < fine_crop_radius * 1.5:
            radii_pass2.append(R)
            centers_2d_pass2.append((xc, yc))
            slice_points_list_pass2.append(pts_slice[inlier_mask])
            inlier_count += np.sum(inlier_mask)

    method_used = "MASt3R aligned multi-slice median (refined)"
    if not radii_pass2:
        logger.warning("[MAST3R DBH] Aligned refined multi-slice failed, using fallback on refined trunk points...")
        mask = (proj_trunk_pass2 >= h_min_pass2) & (proj_trunk_pass2 <= h_min_pass2 + total_h_pass2 * 0.5)
        pts_trunk = fine_trunk_points[mask]
        if len(pts_trunk) < 5:
            return {"error": "No points in aligned trunk region after refined crop"}
        pts_2d = np.column_stack((np.dot(pts_trunk, u1_pass2), np.dot(pts_trunk, u2_pass2)))
        xc, yc, R, _, inlier_mask = fit_circle_robust(pts_2d)
        if R is None or R > fine_crop_radius * 2.0:
            R = R_pass1
            xc, yc = 0.0, 0.0
            slice_points_all = pts_trunk
            inlier_count = 0
            total_slice_points = len(pts_trunk)
        else:
            slice_points_all = pts_trunk[inlier_mask]
            inlier_count = np.sum(inlier_mask)
            total_slice_points = len(pts_trunk)
        radii_pass2 = [R]
        centers_2d_pass2 = [(xc, yc)]
        method_used = "MASt3R aligned refined fallback"
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
        _, _, _, mean_err, _ = fit_circle_robust(pts_2d)
        if mean_err is not None:
            mean_err_cm = float(round(mean_err * scale * 100, 2))

    confidence = "High" if (mean_err_cm <= 3.0) else "Medium"
    if estimated_height_m < 1.0:
        confidence = f"WARNING: Trunk segment captured is only {estimated_height_m:.2f}m tall, insufficient to reach standard breast height (1.3m). DBH measurement may not represent true breast-height diameter - recommend recapturing with more trunk visible in frame"

    # Calculate final inlier ratio
    inlier_ratio = float(inlier_count / total_slice_points) if total_slice_points > 0 else 0.0

    # Sanity checks for flat point cloud (e.g. grass/ground plane) and cylinder orientation
    ranges = points.max(axis=0) - points.min(axis=0)
    vertical_axis_idx = int(np.argmax(ranges))
    
    stds = np.std(points, axis=0)
    std_ratio = np.min(stds) / np.max(stds) if np.max(stds) > 0 else 0.0
    
    cos_angle = abs(v_pass2[vertical_axis_idx]) / np.linalg.norm(v_pass2)
    dev_angle = np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
    
    invalid_orientation = bool(std_ratio < 0.20 or dev_angle > 30.0)

    logger.info(f"[MAST3R DBH] Final result: DBH={dbh_cm:.2f} cm, height={estimated_height_m:.2f} m, confidence={confidence}, inlier_ratio={inlier_ratio:.2%}, invalid_orientation={invalid_orientation}")

    return {
        "dbh_cm":             float(round(dbh_cm, 2)),
        "height_m":           float(round(estimated_height_m, 2)),
        "confidence_note":    confidence,
        "method":             method_used,
        "slice_points_count": slice_count,
        "mean_fit_error_cm":  mean_err_cm,
        "inlier_ratio":       inlier_ratio,
        "invalid_orientation": invalid_orientation,
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

    # 4. Project all points in trunk cylinder column to find height range
    w_cyl = points - np.array([cx, cy, cz])
    h_proj_cyl = np.dot(w_cyl, v)
    perp_dist_cyl = np.linalg.norm(w_cyl - h_proj_cyl[:, np.newaxis] * v[np.newaxis, :], axis=1)
    crop_radius_cyl = max(radius * 1.5, 0.40 / scale)
    trunk_column_points = points[perp_dist_cyl <= crop_radius_cyl]
    if len(trunk_column_points) < 10:
        trunk_column_points = points
    proj = np.dot(trunk_column_points, v)
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
        xc_slice, yc_slice, R, inlier_mask, err = fit_circle_2d(pts_2d)
        if R is not None and R > 0 and R < radius * 1.5:
            radii.append(R)
            centers_2d.append((xc_slice, yc_slice))
            slice_points_list.append(pts_slice[inlier_mask])

    method_used = "Manual override trunk select"
    if not radii:
        logger.warning("[MANUAL DBH] Aligned slice failed, using fallback on selected trunk points...")
        pts_2d = np.column_stack((np.dot(trunk_points, u1), np.dot(trunk_points, u2)))
        xc_slice, yc_slice, R, _, inlier_mask = fit_circle_robust(pts_2d)
        if R is None or R > radius * 2.0:
            R = radius
            xc_slice, yc_slice = 0.0, 0.0
            slice_points_all = trunk_points
        else:
            slice_points_all = trunk_points[inlier_mask]
        radii = [R]
        centers_2d = [(xc_slice, yc_slice)]
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
        _, _, _, mean_err, _ = fit_circle_robust(pts_2d)
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

    # 1. Determine direction vector v
    v = P2 - P1
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-6:
        return {"error": "P1 and P2 are identical or too close"}
    v = v / v_norm

    # Force direction to point upwards (negative Y axis)
    if v[1] > 0:
        v = -v

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

    # 3. Height is defined directly by the user's manual clicks (P1 is bottom/ground, P2 is top)
    h_min = float(np.dot(P1, v))
    h_max = float(np.dot(P2, v))
    total_h = h_max - h_min
    estimated_height_m = float(total_h * scale)

    # 4. Set target breast height target relative to P1 (ground level)
    h_target = float(h_min + breast_height_m / scale)

    # Sanity guard: if the clicked trunk is too short to reach standard breast height (1.3m),
    # clamp the target to the middle of the clicked segment to prevent the ring from floating above P2.
    if (h_target - h_min) >= total_h * 0.90:
        h_target = h_min + total_h * 0.50

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
        if R is not None and R > 0 and R < crop_radius * 1.5:
            radii.append(R)
            centers_2d.append((xc_slice, yc_slice))
            slice_points_list.append(pts_slice[inlier_mask])
            inlier_count += np.sum(inlier_mask)

    method_used = "Manual override 2D clicks"
    if not radii:
        logger.warning("[MANUAL DBH] Aligned slice failed, using fallback on selected cylinder points...")
        pts_2d = np.column_stack((np.dot(trunk_points, u1), np.dot(trunk_points, u2)))
        xc_slice, yc_slice, R, _, inlier_mask = fit_circle_robust(pts_2d)
        if R is None or R > crop_radius * 2.0:
            R = crop_radius / 2
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

