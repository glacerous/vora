import os
import sys
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DBH_Extractor")

def load_scale_factor(manual_scale=1.0):
    """
    Checks for a calibration.json file in the project workspace or test_images
    directory to calibrate the coordinate scale. Defaults to manual_scale if not found.
    """
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
    """
    Parses vertex (x, y, z) coordinates from both binary little-endian and ASCII .ply files.
    This custom parser avoids heavy library dependencies and runs natively on any Python version.
    """
    logger.info(f"Parsing PLY file: {ply_path}")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    with open(ply_path, "rb") as f:
        # Parse header
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
                # Track properties of the vertex element
                parts = line.split()
                if len(parts) >= 3:
                    properties.append((parts[1], parts[2])) # (type, name)
            elif line == "end_header":
                break
                
        logger.info(f"Header parsed. Vertices: {num_vertices}, Format: {'Binary' if is_binary else 'ASCII'}, Properties count: {len(properties)}")
        
        if num_vertices <= 0:
            raise ValueError("No vertices found in PLY file header.")

        if is_binary:
            # Read binary float32 coordinates
            # Count size of each vertex in bytes (assuming standard float properties)
            # Most properties in Gaussian Splat PLY files are float32 (4 bytes)
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
                    # Fallback to float32
                    dtype_map.append((p_name, "<f4"))
                    total_bytes += 4
                    
            logger.info(f"Vertex stride size: {total_bytes} bytes")
            # Read structured array
            vertex_data = np.fromfile(f, dtype=np.dtype(dtype_map), count=num_vertices)
            
            # Extract x, y, z
            x = vertex_data['x']
            y = vertex_data['y']
            z = vertex_data['z']
            points = np.column_stack((x, y, z))
        else:
            # Parse ASCII format
            points = []
            for _ in range(num_vertices):
                line = f.readline().decode("ascii", errors="ignore").strip()
                if not line:
                    break
                parts = line.split()
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            points = np.array(points, dtype=np.float32)

        logger.info(f"Successfully loaded {len(points)} points. Spatial bounds: Min={points.min(axis=0)}, Max={points.max(axis=0)}")
        return points

def fit_circle_2d(points_2d):
    """
    Fits a circle to 2D coordinates using the linearized least-squares Kasa method.
    Returns: (xc, yc, R, error_message)
    """
    if len(points_2d) < 3:
        return None, None, None, "Too few points (< 3) for circle fitting"
        
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    
    # Solve linear system: [x, y, 1] * [A, B, C]^T = x^2 + y^2
    # where A = 2*xc, B = 2*yc, C = R^2 - xc^2 - yc^2
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
    """
    Fits a circle iteratively, rejecting outliers further than threshold * R at each step.
    Helps isolate tree trunk contour from leaves, ground, or reconstruction noise.
    """
    inliers = points_2d
    xc, yc, R = None, None, None
    logger.info(f"[DBH_Extractor] Starting robust circle fitting on {len(points_2d)} coordinates.")
    
    for i in range(max_iters):
        if len(inliers) < 3:
            logger.warning(f"[DBH_Extractor] Iteration {i}: Too few inliers ({len(inliers)}) remaining.")
            break
            
        xc, yc, R, err = fit_circle_2d(inliers)
        if err is not None:
            logger.warning(f"[DBH_Extractor] Iteration {i}: Fit failed: {err}")
            break
            
        logger.info(f"[DBH_Extractor] Iteration {i}: Fitted center=({xc:.4f}, {yc:.4f}), R={R:.4f} units")
            
        # Calculate radial distance for each point to check error
        dists = np.sqrt((inliers[:, 0] - xc)**2 + (inliers[:, 1] - yc)**2)
        radial_errors = np.abs(dists - R)
        
        # Determine error threshold (at least 1 cm)
        thresh = max(0.01, outlier_threshold_ratio * R)
        mask = radial_errors < thresh
        
        num_inliers = np.sum(mask)
        logger.info(f"[DBH_Extractor] Iteration {i}: Inliers count: {num_inliers}/{len(inliers)} using threshold: {thresh:.4f}")
        
        if num_inliers == len(inliers) or num_inliers < 3:
            logger.info(f"[DBH_Extractor] Iteration {i}: Converged or reached minimal subset.")
            break # Converged or can't filter further
            
        inliers = inliers[mask]
        
    # Final fit validation
    if R is not None:
        dists = np.sqrt((inliers[:, 0] - xc)**2 + (inliers[:, 1] - yc)**2)
        mean_err = np.mean(np.abs(dists - R))
        logger.info(f"[DBH_Extractor] Robust fit complete. Final center=({xc:.4f}, {yc:.4f}), R={R:.4f}, Inliers={len(inliers)}/{len(points_2d)}, Mean Error={mean_err:.4f}")
        return xc, yc, R, mean_err
    return None, None, None, None

def extract_dbh(ply_path, scale_factor=1.0, vertical_axis='z', breast_height=1.3, tolerance=0.05):
    """
    Extracts tree DBH (Diameter at Breast Height) from a point cloud (.ply).
    Tries to use Open3D Poisson surface reconstruction if available.
    Otherwise, falls back to a pure-numpy point cloud slicing and iterative circle fitting.
    
    Parameters:
      - ply_path: path to the output .ply file
      - scale_factor: scale multiplier (units to meters conversion factor)
      - vertical_axis: axis corresponding to the height coordinate ('x', 'y', or 'z')
      - breast_height: standard height above base for DBH measurement (1.3 meters)
      - tolerance: height slice thickness parameter in meters
      
    Returns:
      dict with keys: {dbh_cm, height_m, confidence_note, method}
    """
    logger.info("Initializing DBH extraction...")
    scale = load_scale_factor(scale_factor)
    
    # 1. Load coordinates
    try:
        points = parse_ply_points(ply_path)
    except Exception as e:
        logger.error(f"Failed to load point cloud: {e}")
        return {"error": f"Failed to load point cloud: {e}"}
        
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[vertical_axis.lower()]
    z_coords = points[:, axis_idx]
    
    # Filter vertical bounds (1st & 99th percentiles to avoid extreme noise)
    z_min = np.percentile(z_coords, 1)
    z_max = np.percentile(z_coords, 99)
    estimated_height_m = float((z_max - z_min) * scale)
    
    logger.info(f"[DBH_Extractor] Ground alignment & bounds check:")
    logger.info(f"[DBH_Extractor] Estimated base ground Z level (1st percentile): {z_min:.4f} units")
    logger.info(f"[DBH_Extractor] Estimated tree top Z level (99th percentile): {z_max:.4f} units")
    logger.info(f"[DBH_Extractor] Calculated coordinate height delta: {z_max - z_min:.4f} units")
    logger.info(f"[DBH_Extractor] Tree height (meters, scaled): {estimated_height_m:.2f} m")
    
    # 2. Try Open3D Poisson Reconstruction first
    try:
        import open3d as o3d
        logger.info("Open3D is available. Running Poisson Surface Reconstruction...")
        
        # Load Open3D PointCloud
        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.is_empty():
            raise ValueError("Empty point cloud loaded in Open3D")
            
        # Normal estimation (required for Poisson)
        logger.info("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Poisson surface reconstruction
        logger.info("Running TriangleMesh.create_from_point_cloud_poisson (depth=9)...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        
        # Crop low density vertices to remove reconstruction halos
        vertices_to_remove = densities < np.percentile(densities, 10)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Extract vertices
        mesh_points = np.asarray(mesh.vertices)
        z_mesh = mesh_points[:, axis_idx]
        
        # Slicing reconstruction mesh at breast height (convert meters to units)
        z_target = z_min + (breast_height / scale)
        tol = tolerance / scale
        
        logger.info(f"[DBH_Extractor] Poisson slice variables: target Z={z_target:.4f}, tolerance={tol:.4f} units")
        
        slice_mask = np.abs(z_mesh - z_target) <= tol
        slice_points = mesh_points[slice_mask]
        method_used = "Open3D Poisson Mesh"
        logger.info(f"[DBH_Extractor] Sliced Poisson Mesh. Extracted {len(slice_points)} contour vertices.")
        
    except ImportError:
        logger.info("[DBH_Extractor] Open3D is not installed/compatible. Falling back to robust Pure-Numpy Point Cloud Slicing...")
        # Slicing the raw point cloud
        z_target = z_min + (breast_height / scale)
        tol = tolerance / scale
        
        logger.info(f"[DBH_Extractor] Slicing variables: target Z={z_target:.4f}, tolerance={tol:.4f} units")
        
        slice_mask = np.abs(z_coords - z_target) <= tol
        slice_points = points[slice_mask]
        method_used = "Numpy Point Cloud Slicing"
        logger.info(f"[DBH_Extractor] Sliced raw point cloud. Extracted {len(slice_points)} points in slice.")
        
    except Exception as e:
        logger.warning(f"[DBH_Extractor] Open3D Poisson pipeline failed: {e}. Falling back to Numpy Point Cloud Slicing...")
        z_target = z_min + (breast_height / scale)
        tol = tolerance / scale
        
        logger.info(f"[DBH_Extractor] Slicing variables: target Z={z_target:.4f}, tolerance={tol:.4f} units")
        
        slice_mask = np.abs(z_coords - z_target) <= tol
        slice_points = points[slice_mask]
        method_used = "Numpy Point Cloud Slicing"
        logger.info(f"[DBH_Extractor] Sliced raw point cloud. Extracted {len(slice_points)} points in slice.")
        
    # 3. Project to 2D and Fit Circle
    proj_axes = [i for i in [0, 1, 2] if i != axis_idx]
    points_2d = slice_points[:, proj_axes]
    
    if len(points_2d) < 10:
        logger.error("Too few points in the cross-section slice to fit a trunk circle.")
        return {
            "dbh_cm": 0.0,
            "height_m": estimated_height_m,
            "confidence_note": "Unreliable: Too few points in breast-height slice. Splat might be too sparse or noisy.",
            "method": method_used,
            "slice_points_count": len(points_2d),
            "mean_fit_error_cm": 0.0
        }
        
    # Fit circle robustly
    xc, yc, R, mean_err = fit_circle_robust(points_2d)
    
    if R is None:
        return {
            "dbh_cm": 0.0,
            "height_m": estimated_height_m,
            "confidence_note": "Unreliable: Circle fitting failed on sliced trunk points.",
            "method": method_used,
            "slice_points_count": len(points_2d),
            "mean_fit_error_cm": 0.0
        }
        
    # Convert Radius in units to Diameter in Centimeters (1 unit * scale = 1 meter -> * 100 = cm)
    dbh_m = R * 2.0 * scale
    dbh_cm = dbh_m * 100.0
    
    # Assess confidence based on noise and points count
    if len(points_2d) < 50:
        confidence = "Low (sparse points in slice)"
    elif mean_err * scale > 0.05: # mean fitting error > 5cm
        confidence = "Low (high trunk fitting noise)"
    elif mean_err * scale > 0.02: # mean fitting error > 2cm
        confidence = "Medium (some trunk surface noise)"
    else:
        confidence = "High"
        
    logger.info(f"Extracted DBH: {dbh_cm:.2f} cm (Height: {estimated_height_m:.2f} m, Confidence: {confidence})")
    
    return {
        "dbh_cm": float(round(dbh_cm, 2)),
        "height_m": float(round(estimated_height_m, 2)),
        "confidence_note": confidence,
        "method": method_used,
        "slice_points_count": len(points_2d),
        "mean_fit_error_cm": float(round(mean_err * scale * 100, 2))
    }
