#!/usr/bin/env python3
"""
3D Reconstruction Pipeline Server — FastAPI implementation
Run alongside Flask for parallel testing:
    uvicorn server_fastapi:app --host 0.0.0.0 --port 8001 --reload

Production start command (after Flask removal):
    uvicorn server:app --host 0.0.0.0 --port $PORT
"""
import asyncio
import glob
import os
import time
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import boto3
from botocore.config import Config

load_dotenv()

# ── Directory setup ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRAMES_DIR = os.path.join(BASE_DIR, "test_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

for _d in (UPLOAD_DIR, FRAMES_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Global pipeline state ────────────────────────────────────────────────────
state: dict = {
    "stage":             "idle",   # idle | extracting | extracted | reconstructing | done | error
    "message":           "Ready.",
    "frame_count":       0,
    "error":             None,
    "carbon_estimation": None,
    "overlap_warning":   None,
    "cancel_requested":  False,
    "calibration_frame":  None,
    "tree_code":         None,
    "started_at":        None,
}

def upd(stage: str, msg: str, **kw: Any) -> None:
    state.update({"stage": stage, "message": msg, **kw})

# ── Pydantic request / response models ───────────────────────────────────────

class ReconstructRequest(BaseModel):
    """Optional JSON body for POST /reconstruct."""
    tree_code: Optional[str] = None
    remove_background: Optional[bool] = False
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    p1: Optional[List[float]] = None
    p2: Optional[List[float]] = None
    width: Optional[int] = None
    height: Optional[int] = None
    iterations: Optional[int] = 2000

class StatusResponse(BaseModel):
    stage: str
    message: str
    frame_count: int
    error: Optional[str]
    carbon_estimation: Optional[Any]
    overlap_warning: Optional[str]
    frames: List[str]
    has_result: bool
    calibration_frame: Optional[Any] = None
    tree_code: Optional[str] = None
    started_at: Optional[float] = None

class HistoryResponse(BaseModel):
    success: bool
    tree_code: str
    history: List[Any]

class ScansResponse(BaseModel):
    success: bool
    scans: List[Any]



class Recalculate2DRequest(BaseModel):
    p1: list[float]
    p2: list[float]
    width: int
    height: int

# ── Helper: load scale_factor from calibration.json (scan-id-aware) ─────────
import json as _json
import logging as _logging

_calib_logger = _logging.getLogger("calibration")

def _load_scale_factor_for_scan(scan_id: str = None):
    """
    Looks up scale_factor from calibration.json (scan-id-aware).
    Returns: (scale_factor, is_calibrated: bool, calibration_source: str).

    Priority: scan_id entry → 'default' entry → uncalibrated 1.0 fallback.
    When no calibration is configured the fallback is (1.0, False, "uncalibrated_default"):
    results are flagged uncalibrated so downstream consumers show the warning badge.
    """
    calib_path = os.path.join(BASE_DIR, "calibration.json")
    if os.path.exists(calib_path):
        try:
            with open(calib_path, "r") as fh:
                registry = _json.load(fh)
            # 1. Try scan-specific entry first
            if scan_id and scan_id in registry:
                sf = float(registry[scan_id]["scale_factor"])
                _calib_logger.info(
                    f"[CALIBRATION] Loaded scan-specific scale_factor={sf:.8f} "
                    f"for scan_id='{scan_id}' from {calib_path}"
                )
                return sf, True, "manual_scan_specific"
            # 2. Fall back to global 'default' entry if present
            if "default" in registry:
                sf = float(registry["default"]["scale_factor"])
                _calib_logger.info(
                    f"[CALIBRATION] No scan-specific calibration for '{scan_id}'. "
                    f"Using global 'default' scale_factor={sf:.8f} from {calib_path}"
                )
                return sf, True, "manual_default"
            _calib_logger.info(
                f"[CALIBRATION] calibration.json exists at {calib_path} but contains "
                f"no entry for scan_id='{scan_id}' and no 'default' key. "
                f"Falling back to uncalibrated scale_factor=1.0 (uncalibrated_default)."
            )
        except Exception as e:
            _calib_logger.warning(
                f"[CALIBRATION] Failed to read {calib_path}: {e}. "
                f"Falling back to uncalibrated scale_factor=1.0 (uncalibrated_default)."
            )
    else:
        _calib_logger.info(
            "[CALIBRATION] calibration.json NOT FOUND. "
            "Falling back to uncalibrated scale_factor=1.0 (uncalibrated_default)."
        )
    return 1.0, False, "uncalibrated_default"


def filter_points3d_ply(ply_path: str, center_x: float = None, center_z: float = None) -> None:
    """
    Applies the same filtering logic (horizontal crop around trunk cluster peak + statistical outlier removal)
    to points3d.ply before saving/uploading it, ensuring the point cloud shown in Laser Scan mode is clean.
    """
    import numpy as np
    from scipy.spatial import KDTree
    
    if not ply_path or not os.path.exists(ply_path):
        return

    try:
        # 1. Read PLY using a structured reader
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

            if num_vertices <= 0 or not is_binary:
                print(f"[RECONSTRUCT-FILTER] Empty or non-binary PLY: {ply_path}")
                return

            dtype_map = []
            for p_type, p_name in raw_props:
                if p_type in ("float", "float32"):
                    dtype_map.append((p_name, "<f4"))
                elif p_type in ("int", "int32", "uint"):
                    dtype_map.append((p_name, "<i4"))
                elif p_type in ("uchar", "uint8"):
                    dtype_map.append((p_name, "u1"))
                else:
                    dtype_map.append((p_name, "<f4"))

            vertex_data = np.fromfile(f, dtype=np.dtype(dtype_map), count=num_vertices)

        # 2. Extract XYZ
        x = vertex_data['x']
        y = vertex_data['y']
        z = vertex_data['z']
        xyz = np.column_stack((x, y, z))

        if len(xyz) < 30:
            print(f"[RECONSTRUCT-FILTER] Too few points ({len(xyz)}) to filter: {ply_path}")
            return

        # 3. Force Y as the vertical axis (axis index 1)
        rough_axis_idx = 1
        proj_axes = [0, 2]

        # 4. Crop horizontally around peak (trunk cluster)
        if center_x is not None and center_z is not None:
            peak_h1 = center_x
            peak_h2 = center_z
            print(f"[RECONSTRUCT-FILTER] Using custom crop center: ({peak_h1:.3f}, {peak_h2:.3f})")
        else:
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
        CROP_RADIUS = 1.0
        crop_mask = dist_sq <= CROP_RADIUS**2
        
        # If crop yields too few points, fallback to all points
        if np.sum(crop_mask) < 20:
            crop_mask = np.ones(len(xyz), dtype=bool)

        filtered_vertex_data = vertex_data[crop_mask]
        filtered_xyz = xyz[crop_mask]

        # 5. Statistical Outlier Removal using KDTree on cropped points
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

        # 6. Save back to the PLY path
        n = len(filtered_vertex_data)
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
        ]
        for p_type, p_name in raw_props:
            header_lines.append(f"property {p_type} {p_name}")
        header_lines.append("end_header")
        header = "\n".join(header_lines) + "\n"

        with open(ply_path, "wb") as f:
            f.write(header.encode("ascii"))
            filtered_vertex_data.tofile(f)

        print(f"[RECONSTRUCT-FILTER] Successfully cleaned point cloud: {num_vertices} -> {n} points")

    except Exception as exc:
        print(f"[RECONSTRUCT-FILTER] Error filtering points3d.ply: {exc}")


def decimate_ply_file(input_ply_path, output_ply_path, target_count=150000):
    """
    Decimates a PLY file to the target count using random sampling,
    preserving binary little endian format and all attributes.
    """
    import shutil
    import random
    if not input_ply_path or not os.path.exists(input_ply_path):
        return False
    try:
        with open(input_ply_path, "rb") as f:
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

            if num_vertices <= 0 or not is_binary:
                print(f"[DECIMATE] Empty or non-binary PLY: {input_ply_path}")
                return False

            dtype_map = []
            for p_type, p_name in raw_props:
                if p_type in ("float", "float32"):
                    dtype_map.append((p_name, "<f4"))
                elif p_type in ("int", "int32", "uint"):
                    dtype_map.append((p_name, "<i4"))
                elif p_type in ("uchar", "uint8"):
                    dtype_map.append((p_name, "u1"))
                else:
                    dtype_map.append((p_name, "<f4"))

            vertex_data = np.fromfile(f, dtype=np.dtype(dtype_map), count=num_vertices)

        if num_vertices <= target_count:
            if input_ply_path != output_ply_path:
                shutil.copy(input_ply_path, output_ply_path)
            return True

        indices = np.random.choice(num_vertices, size=target_count, replace=False)
        decimated_data = vertex_data[indices]

        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {target_count}",
        ]
        for p_type, p_name in raw_props:
            header_lines.append(f"property {p_type} {p_name}")
        header_lines.append("end_header")
        header = "\n".join(header_lines) + "\n"

        with open(output_ply_path, "wb") as f:
            f.write(header.encode("ascii"))
            decimated_data.tofile(f)
        print(f"[DECIMATE] Decimated {input_ply_path} to {target_count} points: {num_vertices} -> {target_count}")
        return True
    except Exception as e:
        print(f"[DECIMATE ERROR] Failed to decimate PLY {input_ply_path}: {e}")
        if input_ply_path != output_ply_path:
            try:
                shutil.copy(input_ply_path, output_ply_path)
            except:
                pass
        return False


def run_carbon_analysis(
    ply_path: str,
    points3d_path: str = None,
    scan_id: str = None,
    wood_density: float = 0.6,
    forest_type: str = "moist",
    wood_density_source: str = "generic-default",
    climate_zone: str = "Unknown",
    P1: list[float] = None,
    P2: list[float] = None,
    scale_calibration: dict = None,
) -> dict:
    try:
        from carbon.allometric import estimate_carbon, get_root_to_shoot_ratio
        from carbon.dbh_extractor import (extract_dbh, extract_dbh_from_mast3r,
                                          extract_dbh_with_2d_clicks, resolve_height_usage)

        # Load scale_factor from calibration.json (scan-id-aware).
        # Returns (scale_factor, is_calibrated, calibration_source).
        scale_factor, is_calibrated, calibration_source = _load_scale_factor_for_scan(scan_id)

        # ── Implicit auto-pose calibration (only when no manual calibration exists) ──
        # If a full-body person is detected in the frames AND localised inside the
        # reconstructed point cloud, use that as the scale factor instead of the raw
        # uncalibrated default. Manual calibration (calibration.json) always wins.
        auto_calib_note = None
        if not is_calibrated:
            if scale_calibration and scale_calibration.get("is_calibrated"):
                scale_factor = float(scale_calibration["scale_factor"])
                is_calibrated = True
                calibration_source = scale_calibration["source"]
                auto_calib_note = f"auto-kalibrasi pose diterapkan (offloaded ke Modal): {scale_calibration['reason']}"
                print(f"[CARBON-AUTOCALIB] {auto_calib_note} scale_factor={scale_factor:.6f}")

        scale_status = "calibrated" if is_calibrated else "uncalibrated"

        # ── Primary: extract from MASt3R geometric point cloud ────────────────
        dbh_result = None
        err_msg = None
        if P1 is not None and P2 is not None and points3d_path and os.path.exists(points3d_path):
            print(f"[CARBON] Running manual 2D clicks DBH extraction using P1={P1}, P2={P2}")
            try:
                dbh_result = extract_dbh_with_2d_clicks(
                    ply_path=points3d_path,
                    P1=np.array(P1),
                    P2=np.array(P2),
                    scale=scale_factor
                )
                if "error" in dbh_result:
                    print(f"[CARBON] Manual extraction failed: {dbh_result['error']}")
                    err_msg = dbh_result['error']
                    dbh_result = None
                else:
                    print(f"[CARBON] Manual extraction succeeded: DBH={dbh_result['dbh_cm']} cm")
            except Exception as manual_err:
                print(f"[CARBON] Manual extraction exception: {manual_err}")
                err_msg = str(manual_err)
                dbh_result = None
        elif points3d_path and os.path.exists(points3d_path):
            print(f"[CARBON] Trying MASt3R point cloud for measurement: {points3d_path}")
            try:
                dbh_result = extract_dbh_from_mast3r(
                    ply_path=points3d_path, scale_factor=scale_factor
                )
                if "error" in dbh_result:
                    print(f"[CARBON] MASt3R extraction failed: {dbh_result['error']}")
                    err_msg = dbh_result['error']
                    dbh_result = None
                else:
                    print(f"[CARBON] MASt3R extraction succeeded: DBH={dbh_result['dbh_cm']} cm")
            except Exception as mast3r_err:
                print(f"[CARBON] MASt3R extraction exception: {mast3r_err}")
                err_msg = str(mast3r_err)
                dbh_result = None
        else:
            print("[CARBON] points3d.ply not found or not provided. Extraction cannot proceed.")
            err_msg = "points3d.ply tidak ditemukan"

        # If extraction failed, return null metrics with FAILED confidence note
        if dbh_result is None:
            return {
                "dbh_cm":                  None,
                "height_m":                None,
                "confidence":              f"FAILED - points3d.ply tidak tersedia, hasil DBH tidak valid ({err_msg})",
                "method":                  "None",
                "slice_points_count":      0,
                "mean_fit_error_cm":       0.0,
                "scale_factor_used":       scale_factor,
                "calibrated":              is_calibrated,
                "calibration_source":      calibration_source,
                "scale_status":            scale_status,
                "height_used":             "dbh_only_fallback",
                "total_height_used_m":     None,
                "segment_height_m":        None,
                "height_fallback_reason":  "DBH extraction gagal",
                "height_validated":        False,
                "height_validation_reason": "DBH extraction gagal",
                "quality_status":          "failed",
                "biomass_kg":              None,
                "above_ground_biomass_kg": None,
                "below_ground_biomass_kg": None,
                "carbon_kg":               None,
                "co2e_kg":                 None,
                "co2e_low_kg":             None,
                "co2e_high_kg":            None,
                "co2e_uncertainty_pct":    None,
                "root_to_shoot_ratio":     get_root_to_shoot_ratio(forest_type),
                "root_to_shoot_source":    "IPCC 2006 Tier 1 (Table 4.4)",
                "wood_density_used":       wood_density,
                "wood_density_source":     wood_density_source,
                "climate_zone_detected":   climate_zone,
                "formula_used":            "None",
                "disclaimer":              "Reconstruction failed: points3d.ply not available.",
                "geometry_3d":             None,
            }

        # ── Height validity (full-tree vs trunk-segment) ─────────────────────
        # Only use the height-based Chave formula when there is evidence that the
        # extracted height represents TOTAL tree height, not just a trunk segment.
        hinfo = resolve_height_usage(points3d_path, dbh_result.get("height_m"),
                                      height_input_source="system", scale_factor=scale_factor)
        height_used = hinfo["height_used"]
        total_height_used = hinfo["total_height_used_m"]
        segment_height_m = hinfo["segment_height_m"]
        height_fallback_reason = hinfo["height_fallback_reason"]
        height_validated = hinfo["height_validated"]
        height_validation_reason = hinfo["height_validation_reason"]

        # ── Quality gate on the circle fit ───────────────────────────────────
        quality_status = "ok"
        inlier_ratio = dbh_result.get("inlier_ratio", 1.0)
        invalid_orientation = dbh_result.get("invalid_orientation", False)
        
        if dbh_result.get("slice_points_count", 0) < 10:
            quality_status = "low_points"
        elif invalid_orientation:
            quality_status = "invalid_orientation"
        elif dbh_result.get("mean_fit_error_cm", 0.0) > 10.0:
            quality_status = "high_fit_error"
        elif inlier_ratio < 0.15:
            quality_status = "low_inlier_ratio"

        carbon_result = estimate_carbon(
            dbh_cm=dbh_result["dbh_cm"],
            height_m=hinfo["height_for_formula"],
            wood_density=wood_density,
            forest_type=forest_type,
        )
        # Build the confidence note, explicitly flagging uncalibrated scale.
        confidence_note = dbh_result["confidence_note"]
        if not is_calibrated:
            confidence_note += (
                " | UNKALIBRASI: skala default (PLY unit) dipakai — hasil TIDAK dapat "
                "diandalkan tanpa kalibrasi skala (auto-pose atau calibrate_scale.py)"
            )
        if auto_calib_note:
            confidence_note += f" | {auto_calib_note}"
        if height_used == "dbh_only_fallback" and height_fallback_reason:
            confidence_note += f" | Fallback DBH-only ({height_fallback_reason})"

        return {
            "dbh_cm":                  dbh_result["dbh_cm"],
            "height_m":                dbh_result["height_m"],
            "confidence":              confidence_note,
            "method":                  dbh_result["method"],
            "slice_points_count":      dbh_result["slice_points_count"],
            "mean_fit_error_cm":       dbh_result["mean_fit_error_cm"],
            "scale_factor_used":       scale_factor,
            "calibrated":              is_calibrated,
            "calibration_source":      calibration_source,
            "scale_status":            scale_status,
            "height_used":             height_used,
            "total_height_used_m":     total_height_used,
            "segment_height_m":        segment_height_m,
            "height_fallback_reason":  height_fallback_reason,
            "height_validated":        height_validated,
            "height_validation_reason": height_validation_reason,
            "quality_status":          quality_status,
            "inlier_ratio":            inlier_ratio,
            "biomass_kg":              carbon_result["total_biomass_kg"],
            "above_ground_biomass_kg": carbon_result["above_ground_biomass_kg"],
            "below_ground_biomass_kg": carbon_result["below_ground_biomass_kg"],
            "carbon_kg":               carbon_result["carbon_kg"],
            "co2e_kg":                 carbon_result["co2e_kg"],
            "co2e_low_kg":             carbon_result["co2e_low_kg"],
            "co2e_high_kg":            carbon_result["co2e_high_kg"],
            "co2e_uncertainty_pct":    carbon_result["co2e_uncertainty_pct"],
            "root_to_shoot_ratio":     carbon_result["root_to_shoot_ratio"],
            "root_to_shoot_source":    carbon_result["root_to_shoot_source"],
            "wood_density_used":       wood_density,
            "wood_density_source":     wood_density_source,
            "climate_zone_detected":   climate_zone,
            "formula_used":            carbon_result["formula_used"],
            "disclaimer":              carbon_result["disclaimer"],
            "geometry_3d":             dbh_result.get("geometry_3d"),
        }
    except Exception as exc:
        return {"error": f"Failed to compute carbon metrics: {exc}"}

# ── Helper: check overlap between selected frames and dynamically resample ───
def _check_overlap_and_resample(candidates: list, initial_idxs: list, threshold: float = 0.15) -> list:
    """
    Checks overlap between consecutive frames in initial_idxs.
    If overlap is less than threshold (e.g. 15%), dynamically inserts the middle
    candidate frame from the candidates list to heal the coverage gap.
    """
    import cv2
    import numpy as np

    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    current_idxs = list(initial_idxs)
    added_count = 0
    max_added = 10  # Cap the number of additional frames
    i = 0
    gaps_detected = []

    print(f"[OVERLAP] Starting overlap validation for {len(current_idxs)} frames...")

    while i < len(current_idxs) - 1 and added_count < max_added:
        idx_a = current_idxs[i]
        idx_b = current_idxs[i+1]

        # Cannot insert if they are already adjacent candidates
        if idx_b - idx_a <= 1:
            i += 1
            continue

        # Get grayscale frame matrices directly
        raw_a = candidates[idx_a][2]
        raw_b = candidates[idx_b][2]

        if isinstance(raw_a, np.ndarray) and raw_a.ndim > 1:
            gray_a = raw_a if (len(raw_a.shape) == 2 or raw_a.shape[2] == 1) else cv2.cvtColor(raw_a, cv2.COLOR_BGR2GRAY)
        elif isinstance(raw_a, str) and os.path.exists(raw_a):
            gray_a = cv2.imread(raw_a, cv2.IMREAD_GRAYSCALE)
        else:
            gray_a = cv2.imdecode(raw_a, cv2.IMREAD_GRAYSCALE)

        if isinstance(raw_b, np.ndarray) and raw_b.ndim > 1:
            gray_b = raw_b if (len(raw_b.shape) == 2 or raw_b.shape[2] == 1) else cv2.cvtColor(raw_b, cv2.COLOR_BGR2GRAY)
        elif isinstance(raw_b, str) and os.path.exists(raw_b):
            gray_b = cv2.imread(raw_b, cv2.IMREAD_GRAYSCALE)
        else:
            gray_b = cv2.imdecode(raw_b, cv2.IMREAD_GRAYSCALE)

        # Downscale for performance during overlap matching
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

        print(f"[OVERLAP] Pair {i:02d} (candidate {idx_a} -> {idx_b}): overlap = {ratio*100:.1f}%")

        if ratio < threshold:
            idx_mid = (idx_a + idx_b) // 2
            print(f"[OVERLAP] [WARNING] Low overlap ({ratio*100:.1f}% < {threshold*100:.1f}%). Inserting candidate {idx_mid} between {idx_a} and {idx_b}")
            current_idxs.insert(i + 1, idx_mid)
            added_count += 1
            gaps_detected.append(f"Gap between frames {i} and {i+1} ({ratio*100:.1f}% overlap)")
            i += 2  # Skip testing the newly inserted frame in this pass to prevent loops
        else:
            i += 1

    if gaps_detected:
        warning_msg = f"[WARNING] Low overlap warning: {len(gaps_detected)} gaps detected. Resampled +{added_count} frames. Try slower/steadier capture next time."
        state["overlap_warning"] = warning_msg
        print(f"[OVERLAP] {warning_msg}")
    else:
        state["overlap_warning"] = None
        print(f"[OVERLAP] All pairs satisfy overlap threshold of {threshold*100:.0f}%")

    return current_idxs


# ── Background thread: frame extraction (sync, CPU+IO heavy) ─────────────────
def _extract_thread(video_path: str, target: int, blur_thresh: int) -> None:
    try:
        upd("extracting", "Reading video file...")
        t_read_start = time.time()
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        t_read_end = time.time()
        print(f"[TIMING] Read video file into memory: {t_read_end - t_read_start:.4f}s for {len(video_bytes)/1024/1024:.2f} MB")

        upd("extracting", "Offloading frame extraction to Modal...")
        t_modal_start = time.time()
        import modal
        fn = modal.Function.from_name("instantsplat-app", "extract_video_frames_modal")
        res = fn.remote(video_bytes, target, blur_thresh)
        t_modal_end = time.time()
        print(f"[TIMING] Modal remote frame extraction call: {t_modal_end - t_modal_start:.4f}s")

        frames = res.get("frames", [])
        overlap_warning = res.get("overlap_warning")

        n = len(frames)
        if n == 0:
            raise ValueError(f"No sharp frames found (blur_thresh={blur_thresh}). Try a slower, steadier recording.")

        upd("extracting", f"Saving {n} frames from Modal to disk...")
        t_save_start = time.time()
        for f in glob.glob(os.path.join(FRAMES_DIR, "*")):
            try:
                os.remove(f)
            except Exception:
                pass

        for j, frame_bytes in enumerate(frames):
            img_path = os.path.join(FRAMES_DIR, f"{j:04d}.jpg")
            with open(img_path, "wb") as f_out:
                f_out.write(frame_bytes)

        t_save_end = time.time()
        print(f"[TIMING] Saved {n} frames to disk: {t_save_end - t_save_start:.4f}s")

        state["frame_count"] = n
        state["overlap_warning"] = overlap_warning
        state["calibration_frame"] = None

        if overlap_warning:
            print(overlap_warning)

        upd("extracted", f"✓ {n} sharp frames ready")
        print(f"[EXTRACT] Completed. {n} frames written to {FRAMES_DIR}")

    except Exception as exc:
        print(f"[EXTRACT ERROR] During frame processing: {exc}")
        upd("error", str(exc), error=str(exc))

# ── Background thread: GPU reconstruction + R2/D1 persistence (sync, IO heavy) ─
def _reconstruct_thread(
    tree_code: str,
    remove_background: bool = False,
    gps_lat: float = None,
    gps_lon: float = None,
    p1: list[float] = None,
    p2: list[float] = None,
    width: int = None,
    height: int = None,
    plot_id: int = None,
    claimed_by_user_id: int = None,
    iterations: int = 2000,
) -> None:
    import modal
    progress_dict = modal.Dict.from_name("instantsplat-progress-dict", create_if_missing=True)
    try:
        progress_dict[tree_code] = "Uploading images"
        if state.get("cancel_requested", False):
            raise RuntimeError("Job cancelled by user")
        upd("reconstructing", "Connecting to Modal…")

        t_disk_start = time.time()
        files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
        if not files:
            raise ValueError("No frames found in test_images/")

        imgs = []
        for f in files:
            with open(f, "rb") as fh:
                imgs.append(fh.read())
        t_disk_end = time.time()
        print(f"[TIMING] Read {len(imgs)} frames from local disk: {t_disk_end - t_disk_start:.4f}s")

        upd("reconstructing", f"Sending {len(imgs)} frames to Modal A10G GPU…")
        
        t0 = time.time()
        print(f"[RECONSTRUCT] Connecting to Modal pipeline for tree_code '{tree_code}' at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}")
        print(f"[RECONSTRUCT] Uploading {len(imgs)} frames to GPU cloud (background removed on Modal: {remove_background})...")
        
        if state.get("cancel_requested", False):
            raise RuntimeError("Job cancelled by user")
        fn     = modal.Function.from_name("instantsplat-app", "run_reconstruction")
        
        # Resolve R2 config for direct upload from Modal (prevents OOM on Render)
        r2_config = {
            "CLOUDFLARE_ACCOUNT_ID": os.environ.get("CLOUDFLARE_ACCOUNT_ID"),
            "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID"),
            "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY"),
            "R2_BUCKET_NAME": os.environ.get("R2_BUCKET_NAME"),
            "R2_PUBLIC_URL_PREFIX": os.environ.get("R2_PUBLIC_URL_PREFIX"),
        }

        t_remote_start = time.time()
        try:
            result = fn.remote(imgs, tree_code, remove_background, r2_config, iterations)
        except Exception as remote_exc:
            # fn.remote() failed — this typically happens when the Render server
            # restarted while the Modal job was still running (connection reset).
            # The Modal job may have finished and written its completion marker to
            # the shared Dict. Poll for it for up to 20 minutes before giving up.
            print(f"[RECONSTRUCT] fn.remote() raised: {remote_exc}")
            print(f"[RECONSTRUCT] Checking Modal Dict for crash-recovery completion marker…")
            result = None
            import modal as _modal
            _prog_dict = _modal.Dict.from_name("instantsplat-progress-dict", create_if_missing=True)
            complete_key = f"{tree_code}_complete"
            for _attempt in range(120):  # poll up to 20 min (10s intervals)
                try:
                    if complete_key in _prog_dict:
                        result = {**_prog_dict[complete_key], "uploaded": True}
                        del _prog_dict[complete_key]  # consume it
                        print(f"[RECONSTRUCT] Crash-recovery: found completion marker after {_attempt * 10}s. URLs restored.")
                        break
                except Exception:
                    pass
                upd("reconstructing", f"Server restarted mid-job — waiting for Modal to finish… ({_attempt * 10}s)")
                time.sleep(10)
            if result is None:
                raise RuntimeError(f"fn.remote() failed and Modal did not complete within 20 min: {remote_exc}") from remote_exc
        t_remote_end = time.time()
        
        elapsed_remote = t_remote_end - t_remote_start
        print(f"[RECONSTRUCT] GPU Reconstruction remote call completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_remote_end))}")
        print(f"[RECONSTRUCT] Remote duration: {elapsed_remote:.2f} seconds")

        # ── Unpack result (new dict format: {splat, points3d}) ──────────────
        scale_calibration = None
        splat_bytes = b""
        points3d_bytes = None
        points3d_all_bytes = None
        
        uploaded = False
        splat_url = ""
        points3d_url = ""
        points3d_all_url = ""
        thumbnail_url = ""
        custom_ts = None

        if isinstance(result, dict):
            scale_calibration = result.get("scale_calibration")
            uploaded = result.get("uploaded", False)
            splat_url = result.get("splat_url", "")
            points3d_url = result.get("points3d_url", "")
            points3d_all_url = result.get("points3d_all_url", "")
            thumbnail_url = result.get("thumbnail_url", "")
            custom_ts = result.get("timestamp")
            
            if not uploaded:
                splat_bytes = result.get("splat", b"")
                points3d_bytes = result.get("points3d")
                points3d_all_bytes = result.get("points3d_all")
        else:
            # Backward compat: old Modal version returned raw bytes
            splat_bytes = result

        out = os.path.join(OUTPUT_DIR, "result.ply")
        points3d_path = os.path.join(OUTPUT_DIR, "points3d.ply")
        points3d_highres_path = os.path.join(OUTPUT_DIR, "points3d_highres.ply")
        points3d_all_path = os.path.join(OUTPUT_DIR, "points3D_all.npy")

        # Clean old files to free up disk space
        for p in (out, points3d_path, points3d_highres_path, points3d_all_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

        # Load splat (viewer)
        t_dl_splat_start = time.time()
        if uploaded and splat_url:
            print(f"[RECONSTRUCT] Stream-downloading result.ply from R2 URL: {splat_url}")
            import requests
            res_dl = requests.get(splat_url, stream=True)
            with open(out, "wb") as f:
                for chunk in res_dl.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[RECONSTRUCT] result.ply stream download completed successfully")
        elif splat_bytes:
            with open(out, "wb") as f:
                f.write(splat_bytes)
        else:
            print("[RECONSTRUCT] WARNING: No Gaussian splat PLY received!")
        t_dl_splat_end = time.time()
        print(f"[TIMING] Save/download result.ply: {t_dl_splat_end - t_dl_splat_start:.4f}s")

        mb = 0.0
        if os.path.exists(out):
            mb = os.path.getsize(out) / 1024 / 1024
        print(f"[RECONSTRUCT] Saved splat PLY: {out} ({mb:.2f} MB)")

        # Save points3d.ply (keep raw on local disk for ICP alignment)
        t_dl_pts_start = time.time()
        if uploaded and points3d_url:
            print(f"[RECONSTRUCT] Stream-downloading raw points3d.ply from R2 URL: {points3d_url}")
            import requests
            res_dl = requests.get(points3d_url, stream=True)
            with open(points3d_path, "wb") as f:
                for chunk in res_dl.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[RECONSTRUCT] Raw points3d.ply stream download completed successfully")
        elif points3d_bytes:
            with open(points3d_path, "wb") as f:
                f.write(points3d_bytes)
            print(f"[RECONSTRUCT] Saved raw MASt3R point cloud: {points3d_path} ({len(points3d_bytes)/1024:.1f} KB)")
        else:
            print("[RECONSTRUCT] No MASt3R point cloud returned — measurement will use splat")
            points3d_path = None
        t_dl_pts_end = time.time()
        print(f"[TIMING] Save/download points3d.ply: {t_dl_pts_end - t_dl_pts_start:.4f}s")

        # Save points3d_all.npy
        t_dl_all_start = time.time()
        if uploaded and points3d_all_url:
            print(f"[RECONSTRUCT] Stream-downloading points3d_all.npy from R2 URL: {points3d_all_url}")
            import requests
            res_dl = requests.get(points3d_all_url, stream=True)
            with open(points3d_all_path, "wb") as f:
                for chunk in res_dl.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[RECONSTRUCT] points3d_all.npy stream download completed successfully")
        elif points3d_all_bytes:
            with open(points3d_all_path, "wb") as f:
                f.write(points3d_all_bytes)
            print(f"[RECONSTRUCT] Saved raw MASt3R dense pointmap: {points3d_all_path} ({len(points3d_all_bytes)/1024/1024:.1f} MB)")
        else:
            points3d_all_path = None
        t_dl_all_end = time.time()
        print(f"[TIMING] Save/download points3D_all.npy: {t_dl_all_end - t_dl_all_start:.4f}s")

        t_icp_start = time.time()
        P1_3d = None
        P2_3d = None
        
        if points3d_path and os.path.exists(points3d_path):
            try:
                print(f"[RECONSTRUCT] Reading point cloud files for offloaded Modal alignment/filtering...")
                with open(points3d_path, "rb") as f:
                    ply_bytes = f.read()
                    
                points3d_all_bytes = b""
                if points3d_all_path and os.path.exists(points3d_all_path):
                    with open(points3d_all_path, "rb") as f:
                        points3d_all_bytes = f.read()
                
                print(f"[RECONSTRUCT] Offloading alignment and filtering to Modal...")
                import modal
                fn_align = modal.Function.from_name("instantsplat-app", "align_and_filter_ply_modal")
                res = fn_align.remote(ply_bytes, points3d_all_bytes, p1, p2, width, height)
                
                P1_3d = res.get("P1")
                P2_3d = res.get("P2")
                filtered_ply = res.get("filtered_ply")
                
                if filtered_ply:
                    with open(points3d_highres_path, "wb") as f:
                        f.write(filtered_ply)
                    print(f"[RECONSTRUCT] Saved high-res filtered point cloud from Modal ({len(filtered_ply)/1024:.1f} KB)")
                    # Generate decimated point cloud for display
                    decimate_ply_file(points3d_highres_path, points3d_path, target_count=150000)
                
                if P1_3d and P2_3d:
                    print(f"[RECONSTRUCT] ICP Alignment successful: P1_3d={P1_3d}, P2_3d={P2_3d}")
                else:
                    print(f"[RECONSTRUCT] Auto-filtering complete (no coordinates mapped)")
            except Exception as align_err:
                print(f"[RECONSTRUCT ERROR] Offloaded alignment/filtering failed: {align_err}")
                # Fallback to local crop if it fails
                try:
                    filter_points3d_ply(points3d_path)
                    import shutil
                    shutil.copy(points3d_path, points3d_highres_path)
                    decimate_ply_file(points3d_highres_path, points3d_path, target_count=150000)
                except Exception as fb_err:
                    print(f"[RECONSTRUCT ERROR] Local fallback crop failed: {fb_err}")
        t_icp_end = time.time()
        print(f"[TIMING] Mapping pixel and ICP alignment / filter_points3d_ply: {t_icp_end - t_icp_start:.4f}s")

        elapsed = time.time() - t0
        print(f"[RECONSTRUCT] Total pipeline runtime: {elapsed:.2f} seconds")

        t_meta_start = time.time()
        # 1. Resolve GPS coordinates from EXIF metadata if not provided manually
        if gps_lat is None or gps_lon is None:
            try:
                from carbon.gps_exif import get_exif_gps
                for f in files:
                    coords = get_exif_gps(f)
                    if coords:
                        gps_lat, gps_lon = coords
                        print(f"[RECONSTRUCT-GPS] EXIF GPS detected: ({gps_lat}, {gps_lon})")
                        break
            except Exception as gps_exif_err:
                print(f"[RECONSTRUCT-GPS ERROR] EXIF scan failed: {gps_exif_err}")

        # 2. Resolve Climate Zone from GPS coordinates
        climate_zone = "Unknown"
        forest_type = "moist"
        if gps_lat is not None and gps_lon is not None:
            try:
                from carbon.climate_zone import get_koppen_classification, classify_koppen_to_forest_type
                code = get_koppen_classification(gps_lat, gps_lon)
                if code:
                    climate_zone = code
                    forest_type = classify_koppen_to_forest_type(code)
                    print(f"[RECONSTRUCT-CLIMATE] Koppen Climate: {climate_zone}, Forest type: {forest_type}")
            except Exception as climate_err:
                print(f"[RECONSTRUCT-CLIMATE ERROR] Mapresso query failed: {climate_err}")
        t_meta_end = time.time()
        print(f"[TIMING] EXIF GPS and Koppen Climate Zone classification: {t_meta_end - t_meta_start:.4f}s")

        t_species_start = time.time()
        # 3. Detect Species via Pl@ntNet API
        species_preds = None
        try:
            upd("reconstructing", "Detecting tree species using Pl@ntNet API...")
            from carbon.species_detection import detect_species
            img_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
            if img_files:
                detect_files = []
                if len(img_files) >= 1:
                    detect_files.append(img_files[0])
                if len(img_files) >= 3:
                    detect_files.append(img_files[len(img_files)//2])
                    detect_files.append(img_files[-1])
                elif len(img_files) == 2:
                    detect_files.append(img_files[1])
                species_preds = detect_species(detect_files)
        except Exception as sp_err:
            print(f"[RECONSTRUCT] Pl@ntNet species detection exception: {sp_err}")

        # 4. Determine Wood Density
        wood_density = 0.6
        wood_density_source = "generic-default"
        # Species ID confidence threshold. Below this we do NOT trust top-1 for
        # wood density and fall back to the generic default with an explicit flag
        # instead of silently using whatever Pl@ntNet returned as top-1.
        SPECIES_CONFIDENCE_THRESHOLD = 30.0
        if species_preds and len(species_preds) > 0:
            top_pred = species_preds[0]
            top_conf = top_pred.get("confidence", 0.0)
            if top_conf < SPECIES_CONFIDENCE_THRESHOLD:
                wood_density_source = "generic-default (spesies tidak pasti)"
                print(f"[RECONSTRUCT-WD] Top species confidence {top_conf:.1f}% < {SPECIES_CONFIDENCE_THRESHOLD}% — "
                      f"menggunakan densitas default 0.6 (spesies tidak pasti)")
            else:
                sci_name = top_pred.get("scientific_name")
                try:
                    from carbon.wood_density_lookup import get_wood_density
                    specific_wd = get_wood_density(sci_name)
                    if specific_wd is not None:
                        wood_density = specific_wd
                        wood_density_source = "species-matched"
                        print(f"[RECONSTRUCT-WD] Matched wood density for {sci_name}: {wood_density}")
                except Exception as wd_err:
                    print(f"[RECONSTRUCT-WD ERROR] Wood density lookup exception: {wd_err}")
        t_species_end = time.time()
        print(f"[TIMING] Pl@ntNet species detection & wood density lookup: {t_species_end - t_species_start:.4f}s")

        t_carbon_start = time.time()
        # 5. Run Carbon Analysis using custom parameters
        progress_dict[tree_code] = "Computing DBH & carbon"
        upd("reconstructing", "✓ Reconstruction done. Estimating DBH and Carbon...")
        carbon_est = run_carbon_analysis(
            out, 
            points3d_path=points3d_highres_path, 
            scan_id=tree_code,
            wood_density=wood_density,
            forest_type=forest_type,
            wood_density_source=wood_density_source,
            climate_zone=climate_zone,
            P1=P1_3d,
            P2=P2_3d,
            scale_calibration=scale_calibration,
        )
        
        # Append fallback message to confidence note if GPS not available
        if gps_lat is None or gps_lon is None:
            fallback_msg = " (GPS data not available - fallback to moist forest assumption)"
            if carbon_est.get("confidence"):
                carbon_est["confidence"] += fallback_msg
            else:
                carbon_est["confidence"] = "GPS data not available - fallback to moist forest assumption"
                
        state["carbon_estimation"] = carbon_est
        t_carbon_end = time.time()
        print(f"[TIMING] Local carbon estimation analysis: {t_carbon_end - t_carbon_start:.4f}s")

        if carbon_est and "error" not in carbon_est:
            try:
                t_persistence_start = time.time()
                progress_dict[tree_code] = "Uploading results"
                
                if uploaded:
                    print(f"[RECONSTRUCT] Files were uploaded directly from Modal to R2.")
                    ts = custom_ts or int(time.time())
                    # Upload high-res version to R2
                    if points3d_highres_path and os.path.exists(points3d_highres_path):
                        try:
                            from storage.r2_client import upload_splat
                            upload_splat(points3d_highres_path, tree_code, custom_timestamp=ts)
                            print(f"[RECONSTRUCT] Uploaded high-res points3d_highres.ply to R2")
                        except Exception as upload_err:
                            print(f"Failed to upload points3d_highres.ply to R2: {upload_err}")
                    # Overwrite raw points3d.ply in R2 with the decimated version
                    if points3d_path and os.path.exists(points3d_path):
                        try:
                            from storage.r2_client import upload_splat
                            upload_splat(points3d_path, tree_code, custom_timestamp=ts)
                            print(f"[RECONSTRUCT] Overwrote R2 points3d.ply with decimated version")
                        except Exception as upload_err:
                            print(f"Failed to upload points3d.ply to R2: {upload_err}")
                else:
                    upd("reconstructing", "Uploading reconstruction files to Cloudflare R2...")
                    from storage.r2_client import upload_splat, upload_thumbnail
                    ts = int(time.time())
                    splat_url = upload_splat(out, tree_code, custom_timestamp=ts)
                    
                    # Upload high-res version
                    if points3d_highres_path and os.path.exists(points3d_highres_path):
                        try:
                            upload_splat(points3d_highres_path, tree_code, custom_timestamp=ts)
                            print(f"[RECONSTRUCT] Uploaded high-res points3d_highres.ply to R2")
                        except Exception as upload_err:
                            print(f"Failed to upload points3d_highres.ply to R2: {upload_err}")
                            
                    # Upload decimated points3d.ply
                    if points3d_path and os.path.exists(points3d_path):
                        try:
                            upload_splat(points3d_path, tree_code, custom_timestamp=ts)
                            print(f"[RECONSTRUCT] Uploaded decimated points3d.ply to R2")
                        except Exception as upload_err:
                            print(f"Failed to upload points3d.ply to R2: {upload_err}")

                    # If MASt3R points3D_all.npy was computed, upload it too
                    if points3d_all_path and os.path.exists(points3d_all_path):
                        try:
                            upload_splat(points3d_all_path, tree_code, custom_timestamp=ts)
                            print(f"[RECONSTRUCT] Uploaded MASt3R points3D_all.npy with timestamp {ts}")
                        except Exception as upload_err:
                            print(f"Failed to upload points3D_all.npy to R2: {upload_err}")

                    # Select middle representative frame as thumbnail
                    thumbnail_url = None
                    if files:
                        mid_idx = len(files) // 2
                        representative_frame = files[mid_idx]
                        try:
                            upd("reconstructing", "Uploading representative frame as thumbnail to R2...")
                            thumbnail_url = upload_thumbnail(representative_frame, tree_code)
                        except Exception as thumb_err:
                            print(f"Thumbnail upload error: {thumb_err}")

                upd("reconstructing", "Saving scan results to Cloudflare D1...")
                from storage.d1_client import save_scan_result
                save_scan_result(
                    tree_code=tree_code,
                    dbh_cm=carbon_est.get("dbh_cm"),
                    tinggi_m=carbon_est.get("height_m"),
                    biomassa_kg=carbon_est.get("biomass_kg"),
                    karbon_kg=carbon_est.get("carbon_kg"),
                    co2e_kg=carbon_est.get("co2e_kg"),
                    splat_file_url=splat_url,
                    confidence_note=carbon_est.get("confidence"),
                    thumbnail_url=thumbnail_url,
                    geometry_3d=carbon_est.get("geometry_3d"),
                    species_predictions=species_preds,
                    wood_density_used=carbon_est.get("wood_density_used"),
                    wood_density_source=carbon_est.get("wood_density_source"),
                    climate_zone_detected=carbon_est.get("climate_zone_detected"),
                    formula_used=carbon_est.get("formula_used"),
                    agb_kg=carbon_est.get("above_ground_biomass_kg"),
                    bgb_kg=carbon_est.get("below_ground_biomass_kg"),
                    gps_lat=gps_lat,
                    gps_lon=gps_lon,
                    scale_status=carbon_est.get("scale_status"),
                    scale_factor_used=carbon_est.get("scale_factor_used"),
                    calibration_source=carbon_est.get("calibration_source"),
                    height_used=carbon_est.get("height_used"),
                    total_height_used_m=carbon_est.get("total_height_used_m"),
                    segment_height_m=carbon_est.get("segment_height_m"),
                    height_fallback_reason=carbon_est.get("height_fallback_reason"),
                    quality_status=carbon_est.get("quality_status"),
                    inlier_ratio=carbon_est.get("inlier_ratio"),
                    root_to_shoot_ratio=carbon_est.get("root_to_shoot_ratio"),
                    co2e_uncertainty_pct=carbon_est.get("co2e_uncertainty_pct"),
                    co2e_low_kg=carbon_est.get("co2e_low_kg"),
                    co2e_high_kg=carbon_est.get("co2e_high_kg"),
                    plot_id=plot_id,
                    claimed_by_user_id=claimed_by_user_id,
                )
                t_persistence_end = time.time()
                print(f"[TIMING] R2 upload & D1 database persistence: {t_persistence_end - t_persistence_start:.4f}s")
                upd("done", f"✓ Done in {elapsed:.0f}s — {mb:.1f} MB Gaussian Splat ready! (Tree code: {tree_code})")
            except Exception as exc:
                print(f"Persistence error: {exc}")
                upd("error", f"Reconstruction done, but failed to save: {exc}", error=str(exc))
        else:
            err = (carbon_est or {}).get("error", "Unknown carbon analysis error")
            upd("error", f"Reconstruction done, but carbon analysis failed: {err}", error=err)

    except BaseException as exc:
        if state.get("cancel_requested", False):
            print("[RECONSTRUCT] Cancel requested by user. Aborting...")
            upd("idle", "Ready.")
            state["cancel_requested"] = False
        else:
            print(f"[RECONSTRUCT ERROR] Critical pipeline failure: {exc}")
            upd("error", str(exc), error=str(exc))
    finally:
        try:
            del progress_dict[tree_code]
        except Exception:
            pass

# ── Application lifespan: pre-calculate carbon if result.ply already exists ──
@asynccontextmanager
async def lifespan(application: FastAPI):
    existing_ply = os.path.join(OUTPUT_DIR, "result.ply")
    if os.path.exists(existing_ply):
        print("Found existing result.ply. Pre-calculating carbon metrics (no scan_id context at startup)...")
        # At server startup we don't know which scan_id was last processed,
        # so pass None — _load_scale_factor_for_scan will warn and use default/global.
        state["carbon_estimation"] = await asyncio.to_thread(run_carbon_analysis, existing_ply, None)

    print("=" * 50)
    print("  3D Reconstruction Pipeline (FastAPI)")
    print("=" * 50)
    yield

# ── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI(
    title="3D Tree Reconstruction Pipeline",
    description="Reconstruct 3D Gaussian Splats from video/photos and estimate tree carbon metrics.",
    version="2.0.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

# CORS Configuration
# allow_credentials=False means we can safely use allow_origins=["*"].
# Listed origins are documented for reference; the wildcard covers all of them.
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://vora-frontend-six.vercel.app",  # production Vercel deployment
    "https://vora-frontend.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import Cookie, Depends, HTTPException, status
import secrets
import json

async def get_optional_user(session_token: Optional[str] = Cookie(None)):
    if not session_token:
        return None
    try:
        from storage.d1_client import execute_d1_query
        sql = "SELECT * FROM sessions WHERE token = ?"
        sessions = execute_d1_query(sql, [session_token])
        if not sessions:
            return None
        
        session = sessions[0]
        # Replace Z with +00:00 for backward compatibility in Python fromisoformat
        expires_str = session["expires_at"].replace('Z', '+00:00')
        expires_at = datetime.fromisoformat(expires_str)
        if expires_at < datetime.now(timezone.utc):
            execute_d1_query("DELETE FROM sessions WHERE token = ?", [session_token])
            return None
        
        users = execute_d1_query("SELECT id, username, display_name, is_demo_account, created_at FROM users WHERE id = ?", [session["user_id"]])
        if not users:
            return None
        return users[0]
    except Exception as e:
        print(f"[AUTH ERROR] Failed checking optional user session: {e}")
        return None

async def get_current_user(session_token: Optional[str] = Cookie(None)):
    user = await get_optional_user(session_token)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication credentials were not provided or are invalid."
        )
    return user


@app.get("/metrics", summary="Get server memory usage metrics")
async def get_metrics():
    import os
    max_rss_mb = 0.0
    current_rss_mb = 0.0
    try:
        import resource
        # ru_maxrss is in KB on Linux
        max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        pass
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    current_rss_mb = int(line.split()[1]) / 1024
                    break
    except Exception:
        pass
    return {
        "max_rss_mb": max_rss_mb,
        "current_rss_mb": current_rss_mb,
        "pid": os.getpid()
    }


@app.get("/ping", summary="Liveness / ping check endpoint to wake up server")
async def ping_server():
    return {"status": "ok", "message": "Server is running."}


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/viewer", include_in_schema=False)
@app.get("/viewer.html", include_in_schema=False)
async def viewer():
    return FileResponse(os.path.join(BASE_DIR, "viewer.html"))

@app.get("/gaussian-splats-3d.umd.js", include_in_schema=False)
async def splat_js():
    return FileResponse(os.path.join(BASE_DIR, "gaussian-splats-3d.umd.js"))

@app.get("/output/{fn:path}", include_in_schema=False)
async def output_file(fn: str):
    path = os.path.join(OUTPUT_DIR, fn)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)

@app.get("/frames/{fn}", include_in_schema=False)
async def frame_file(fn: str):
    path = os.path.join(FRAMES_DIR, fn)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(path)

# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/status", response_model=StatusResponse, summary="Poll pipeline state")
async def status():
    """Returns the current pipeline stage and all associated metadata."""
    frames = (
        sorted(f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(".jpg"))
        if os.path.exists(FRAMES_DIR)
        else []
    )
    
    current_msg = state.get("message")
    if state.get("stage") == "reconstructing":
        tc = state.get("tree_code")
        if tc:
            try:
                import modal
                progress_dict = modal.Dict.from_name("instantsplat-progress-dict", create_if_missing=True)
                if tc in progress_dict:
                    current_msg = progress_dict[tc]
            except Exception as e:
                print(f"[STATUS] Failed to read Modal progress: {e}")
                
    return {
        **state,
        "message":    current_msg,
        "frames":     frames,
        "has_result": os.path.exists(os.path.join(OUTPUT_DIR, "result.ply")),
    }

@app.post("/upload_video", summary="Upload a video and start frame extraction")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file (MP4/MOV/AVI/WEBM/MKV, up to 4 GB)"),
    frames: int = Form(default=25, description="Target number of frames to extract"),
    blur_thresh: int = Form(default=80, description="Minimum Laplacian blur score to keep a frame"),
):
    """
    Accepts a video upload, saves it to disk, then starts smart frame extraction
    asynchronously. Poll `/status` for progress.
    """
    allowed_extensions = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    path = os.path.join(UPLOAD_DIR, f"input{ext}")

    t_start = time.time()
    total_bytes = 0
    # Stream to disk in 1 MB chunks — safe for very large (4 GB) files
    with open(path, "wb") as out:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            total_bytes += len(chunk)
    t_end = time.time()
    print(f"[TIMING] Video upload write to disk: {t_end - t_start:.4f}s for {total_bytes / (1024 * 1024):.2f} MB")

    state["error"] = None
    state["cancel_requested"] = False
    upd("extracting", "Video received, starting smart extraction…")
    # BackgroundTasks dispatches sync callables to thread pool automatically
    background_tasks.add_task(_extract_thread, path, frames, blur_thresh)
    return {"queued": True}


@app.post("/reconstruct", summary="Start GPU reconstruction on extracted frames")
async def reconstruct(
    background_tasks: BackgroundTasks,
    # Accept tree_code, remove_background, and GPS coordinates from JSON body OR query string for maximum flexibility
    body: Optional[ReconstructRequest] = Body(default=None),
    tree_code_query: Optional[str] = Query(default=None, alias="tree_code"),
    remove_bg_query: Optional[bool] = Query(default=None, alias="remove_background"),
    gps_lat_query: Optional[float] = Query(default=None, alias="gps_lat"),
    gps_lon_query: Optional[float] = Query(default=None, alias="gps_lon"),
    iterations_query: Optional[int] = Query(default=None, alias="iterations"),
    optional_user: Optional[dict] = Depends(get_optional_user),
):
    """
    Dispatches the GPU reconstruction job (via Modal) as a background task.
    Returns immediately with `tree_code` so the client can track this scan.
    If no `tree_code` is provided, one is auto-generated in format `POHON-XXXX`.
    """
    if state["stage"] not in ("extracted", "done", "error"):
        raise HTTPException(status_code=400, detail="Not ready — extract frames first")

    # Resolve remove_background
    remove_bg = False
    if remove_bg_query is not None:
        remove_bg = remove_bg_query
    elif body and body.remove_background is not None:
        remove_bg = body.remove_background

    # Reset cancellation request
    state["cancel_requested"] = False

    # Generate or resolve tree code
    import random
    final_code = tree_code_query or (body.tree_code if body else None) or f"POHON-{random.randint(1000, 9999)}"
    final_code = final_code.strip().upper()

    # Resolve GPS params
    gps_lat = gps_lat_query if gps_lat_query is not None else (body.gps_lat if body else None)
    gps_lon = gps_lon_query if gps_lon_query is not None else (body.gps_lon if body else None)

    # Resolve manual clicks
    p1 = body.p1 if body else None
    p2 = body.p2 if body else None
    width = body.width if body else None
    height = body.height if body else None

    # Resolve iterations
    iterations = 2000
    if iterations_query is not None:
        iterations = iterations_query
    elif body and body.iterations is not None:
        iterations = body.iterations

    # Resolve active plot session for auto-association has been deprecated and removed.
    plot_id = None
    claimed_by_user_id = optional_user["id"] if optional_user else None

    state["error"] = None
    state["tree_code"] = final_code
    import time
    state["started_at"] = time.time()
    upd("reconstructing", "Queuing reconstruction…")
    background_tasks.add_task(
        _reconstruct_thread,
        final_code,
        remove_bg,
        gps_lat,
        gps_lon,
        p1,
        p2,
        width,
        height,
        plot_id,
        claimed_by_user_id,
        iterations
    )
    return {"started": True, "tree_code": final_code}


@app.post("/cancel", summary="Cancel active pipeline job")
async def cancel_job():
    """Signals cancellation to background threads and resets state to idle."""
    state["cancel_requested"] = True
    upd("idle", "Ready (Previous job cancelled).")
    state["overlap_warning"] = None
    state["error"] = None
    return {"success": True, "message": "Cancellation request registered."}

@app.get(
    "/history/{tree_code}",
    response_model=HistoryResponse,
    summary="Fetch scan history for a tree code",
)
async def history(tree_code: str):
    """
    Returns all historical scan records for `tree_code` from Cloudflare D1,
    ordered by scan_date descending. The underlying `requests` call is wrapped
    in `asyncio.to_thread` so it does not block the event loop.
    """
    try:
        from storage.d1_client import get_scan_history
        records = await asyncio.to_thread(get_scan_history, tree_code)
        return {"success": True, "tree_code": tree_code, "history": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get(
    "/scans",
    response_model=ScansResponse,
    summary="Fetch all scan history with pagination",
)
async def get_scans(limit: int = 20, offset: int = 0, include_invalid: bool = Query(default=False)):
    """
    Returns all scan records from Cloudflare D1 with optional limit and offset,
    ordered by scan_date descending.
    """
    try:
        from storage.d1_client import get_all_scans
        records = await asyncio.to_thread(get_all_scans, limit, offset, include_invalid)
        return {"success": True, "scans": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get(
    "/splat-proxy/{tree_code}/{filename}",
    summary="Proxy and stream splat files from Cloudflare R2",
)
async def splat_proxy(tree_code: str, filename: str):
    """
    Proxies splat/ply files from Cloudflare R2, streaming them back to the client.
    Bypasses DNS/SSL blocks on R2.dev subdomains by using the direct S3 API.
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("R2_BUCKET_NAME")

    if not all([account_id, access_key, secret_key, bucket_name]):
        raise HTTPException(status_code=500, detail="R2 storage credentials not configured")

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    object_key = f"tree_scans/{tree_code}/{filename}"

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name="auto"
        )
        # Fetch object metadata and stream from R2 (boto3 call is blocking, so run in thread pool)
        response = await asyncio.to_thread(
            s3.get_object,
            Bucket=bucket_name,
            Key=object_key
        )
    except Exception as exc:
        err_msg = str(exc)
        if "NoSuchKey" in err_msg:
            raise HTTPException(status_code=404, detail="File not found in storage")
        raise HTTPException(status_code=500, detail=f"Failed to fetch from R2: {err_msg}")

    body = response["Body"]

    def iter_chunks():
        try:
            while True:
                # Read 1 MB chunk (blocking call, but executed safely in thread by StreamingResponse)
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            body.close()

    media_type = "application/octet-stream"
    if filename.endswith(".ply"):
        media_type = "application/x-ply"

    headers = {
        "Cache-Control": "public, max-age=31536000, immutable",
        "Content-Length": str(response.get("ContentLength", ""))
    }

    return StreamingResponse(iter_chunks(), media_type=media_type, headers=headers)

@app.post("/log", include_in_schema=False)
async def client_log(data: dict = Body(...)):
    print(f"[CLIENT LOG] {data.get('level', 'INFO')}: {data.get('message', '')}")
    return {"status": "ok"}





def map_pixel_to_cropped(u_org, v_org, W1, H1, W_crop, H_crop, size=512):
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


def get_robust_3d_point(pointmap, u, v, window=15):
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
        # Spiral search fallback
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
        
    # Sort by depth (Z axis is index 2)
    # The tree trunk is the foreground, so it has smaller Z values.
    # Take the average of the closest 30% of points.
    points = sorted(points, key=lambda p: p[2])
    n_keep = max(1, int(len(points) * 0.3))
    return np.mean(points[:n_keep], axis=0)


@app.patch("/scan/{scan_id}/recalculate", summary="Recalculate DBH and carbon using 2D clicked coordinates mapped to MASt3R pointmap")
async def recalculate_scan(scan_id: int, body: Recalculate2DRequest):
    """
    Accepts 2D click coordinates from representative frame.
    Downloads points3D_all.npy (dense pointmap) and points3d.ply.
    Maps clicked coordinates to 3D point cloud, recalculates DBH/carbon.
    Updates the existing scan record in database by scan_id.
    """
    try:
        import requests
        from storage.d1_client import execute_d1_query, update_scan_result
        from carbon.dbh_extractor import extract_dbh_with_2d_clicks, resolve_height_usage
        from carbon.allometric import estimate_carbon

        # 1. Fetch target scan record by scan_id
        sql = "SELECT * FROM tree_scans WHERE id = ?"
        scans = execute_d1_query(sql, [scan_id])
        if not scans:
            raise HTTPException(status_code=404, detail="Scan record not found")
        
        target_scan = scans[0]
        tree_code = target_scan.get("tree_code")
        splat_file_url = target_scan.get("splat_file_url")
        if not splat_file_url:
            raise HTTPException(status_code=400, detail="Target scan does not have a splat file URL")

        # 2. Derive pointmap (.npy) and points3d (.ply) URLs from splat_file_url
        # 2. Derive pointmap (.npy) and points3d (.ply) URLs from splat_file_url robustly
        base_url, filename = splat_file_url.rsplit("/", 1)
        name_parts = filename.split("_", 1)
        if len(name_parts) == 2:
            timestamp = name_parts[0]
        else:
            timestamp = filename.split(".")[0]
        
        pointmap_url = f"{base_url}/{timestamp}_points3D_all.npy"
        points3d_url = f"{base_url}/{timestamp}_points3d.ply"

        # 3. Create temp local directories
        local_dir = os.path.join(UPLOAD_DIR, "recalculates")
        os.makedirs(local_dir, exist_ok=True)
        local_npy_path = os.path.join(local_dir, f"{tree_code}_{timestamp}_points3D_all.npy")
        local_ply_path = os.path.join(local_dir, f"{tree_code}_{timestamp}_points3d.ply")
        local_ply_highres_path = os.path.join(local_dir, f"{tree_code}_{timestamp}_points3d_highres.ply")

        # 4. Download dense pointmap (NPY) from R2 if not already cached locally
        if os.path.exists(local_npy_path) and os.path.getsize(local_npy_path) > 100000:
            print(f"[RECALCULATE] Found locally cached dense pointmap: {local_npy_path} — skipping download.")
        else:
            print(f"[RECALCULATE] Downloading dense pointmap from {pointmap_url}")
            try:
                res_npy = requests.get(pointmap_url, timeout=30)
                if res_npy.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail="Historical scan does not have dense pointmap data. Please perform a new reconstruction."
                    )
                with open(local_npy_path, "wb") as f:
                    f.write(res_npy.content)
            except HTTPException as he:
                raise he
            except Exception as npy_err:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download dense pointmap: {npy_err}"
                )

        # 5. Download the existing points3d_highres.ply from R2 if not already cached locally
        if os.path.exists(local_ply_highres_path) and os.path.getsize(local_ply_highres_path) > 1000:
            print(f"[RECALCULATE] Found locally cached points3d_highres.ply: {local_ply_highres_path} — skipping download.")
        else:
            points3d_highres_url = points3d_url.replace("points3d.ply", "points3d_highres.ply")
            print(f"[RECALCULATE] Downloading existing points3d_highres.ply from {points3d_highres_url}")
            try:
                res_ply = requests.get(points3d_highres_url, timeout=30)
                if res_ply.status_code == 200:
                    with open(local_ply_highres_path, "wb") as f:
                        f.write(res_ply.content)
                    print(f"[RECALCULATE] Successfully downloaded existing points3d_highres.ply from R2")
                else:
                    # Fallback to standard points3d.ply
                    print(f"[RECALCULATE] points3d_highres.ply not found. Falling back to downloading standard points3d.ply from {points3d_url}")
                    res_ply_std = requests.get(points3d_url, timeout=30)
                    if res_ply_std.status_code != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to download points3d.ply: HTTP {res_ply_std.status_code}"
                        )
                    with open(local_ply_highres_path, "wb") as f:
                        f.write(res_ply_std.content)
                    print(f"[RECALCULATE] Successfully downloaded standard points3d.ply from R2 as fallback")
            except Exception as ply_err:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download points3d: {ply_err}"
                )

        # 6. Load pointmap and perform coordinate mapping
        pts3d = np.load(local_npy_path)
        N, H_crop, W_crop, _ = pts3d.shape
        # Pick the representative frame by valid-pixel coverage (same logic as _reconstruct_thread)
        valid_counts = np.array([
            np.sum(~np.all(pts3d[i] == 0, axis=-1) & ~np.any(np.isnan(pts3d[i]), axis=-1))
            for i in range(N)
        ])
        repr_idx = int(np.argmax(valid_counts))
        print(f"[RECALCULATE] Representative frame: idx={repr_idx}/{N} ({valid_counts[repr_idx]:,} valid pixels)")
        pointmap = pts3d[repr_idx]

        u1_crop, v1_crop = map_pixel_to_cropped(body.p1[0], body.p1[1], body.width, body.height, W_crop, H_crop)
        u2_crop, v2_crop = map_pixel_to_cropped(body.p2[0], body.p2[1], body.width, body.height, W_crop, H_crop)

        P1_cam = get_robust_3d_point(pointmap, u1_crop, v1_crop)
        P2_cam = get_robust_3d_point(pointmap, u2_crop, v2_crop)
        
        # Hybrid depth constraint: clamp depth only if deviation is > 1.5m (background hit)
        z_diff = P2_cam[2] - P1_cam[2]
        if abs(z_diff) > 1.5:
            print(f"[RECALCULATE] Z-depth deviation ({abs(z_diff):.3f}m) > 1.5m. Background hit detected. Clamping Z2 to Z1.")
            P2_cam[2] = P1_cam[2]

        # Align camera space to world space using ICP (raw-to-raw) on Modal
        try:
            print(f"[RECALCULATE] Reading point cloud files for offloaded Modal alignment/filtering...")
            with open(local_ply_highres_path, "rb") as f:
                ply_bytes = f.read()
            with open(local_npy_path, "rb") as f:
                points3d_all_bytes = f.read()

            print(f"[RECALCULATE] Offloading alignment and filtering to Modal...")
            import modal
            fn_align = modal.Function.from_name("instantsplat-app", "align_and_filter_ply_modal")
            res = fn_align.remote(ply_bytes, points3d_all_bytes, body.p1, body.p2, body.width, body.height)
            
            P1 = res.get("P1")
            P2 = res.get("P2")
            filtered_ply = res.get("filtered_ply")
            
            if filtered_ply:
                with open(local_ply_highres_path, "wb") as f:
                    f.write(filtered_ply)
                print(f"[RECALCULATE] Saved high-res filtered point cloud from Modal ({len(filtered_ply)/1024:.1f} KB)")
                # Generate decimated point cloud for display
                decimate_ply_file(local_ply_highres_path, local_ply_path, target_count=150000)
                
                # Upload both the decimated points3d.ply and high-res version to R2 to update them
                try:
                    import shutil
                    temp_upload_dir = os.path.join(local_dir, f"temp_upload_{timestamp}_{tree_code}")
                    os.makedirs(temp_upload_dir, exist_ok=True)
                    
                    target_ply = os.path.join(temp_upload_dir, "points3d.ply")
                    target_ply_highres = os.path.join(temp_upload_dir, "points3d_highres.ply")
                    
                    shutil.copy2(local_ply_path, target_ply)
                    shutil.copy2(local_ply_highres_path, target_ply_highres)
                    
                    from storage.r2_client import upload_splat
                    upload_splat(target_ply, tree_code, custom_timestamp=int(timestamp))
                    upload_splat(target_ply_highres, tree_code, custom_timestamp=int(timestamp))
                    print(f"[RECALCULATE] Uploaded updated points3d.ply and points3d_highres.ply to R2")
                    
                    # Clean up temp upload files
                    shutil.rmtree(temp_upload_dir, ignore_errors=True)
                except Exception as upload_err:
                    print(f"[RECALCULATE] Failed to upload recalculated PLYs to R2: {upload_err}")
                
            if P1 and P2:
                print(f"[RECALCULATE] ICP Alignment successful. P1_world={P1}, P2_world={P2}")
            else:
                raise ValueError("Modal did not return aligned coordinates")
        except Exception as align_err:
            print(f"[RECALCULATE ERROR] Offloaded ICP Alignment failed: {align_err}. Using local camera-space fallback.")
            P1 = P1_cam.tolist()
            P2 = P2_cam.tolist()
            try:
                import shutil
                shutil.copy(local_ply_highres_path, local_ply_path)
                filter_points3d_ply(local_ply_path)
                shutil.copy(local_ply_path, local_ply_highres_path)
                decimate_ply_file(local_ply_highres_path, local_ply_path, target_count=150000)
            except Exception as fb_err:
                print(f"[RECALCULATE ERROR] Local fallback crop failed: {fb_err}")

        # 7. Perform DBH extraction with 2D clicks
        scale_factor, _is_cal2, _src2 = _load_scale_factor_for_scan(tree_code)
        res_override = extract_dbh_with_2d_clicks(
            ply_path=local_ply_highres_path,
            P1=np.array(P1),
            P2=np.array(P2),
            scale=scale_factor
        )

        if "error" in res_override:
            raise HTTPException(status_code=400, detail=res_override["error"])

        # 8. Recalculate biomass & carbon
        # Check species_predictions
        species_preds = None
        raw_sp = target_scan.get("species_predictions")
        if raw_sp:
            try:
                while isinstance(raw_sp, str) and raw_sp.strip():
                    raw_sp = json.loads(raw_sp)
                species_preds = raw_sp
            except Exception as e:
                print(f"[RECALCULATE] Failed to parse species_predictions: {e}")
                species_preds = None

        # Pl@ntNet opportunistic retry if species_predictions is missing/empty
        thumbnail_url = target_scan.get("thumbnail_url")
        if not species_preds and thumbnail_url:
            print(f"[RECALCULATE] species_predictions is empty. Retrying Pl@ntNet with thumbnail {thumbnail_url}...")
            try:
                temp_thumb_path = os.path.join(local_dir, f"{tree_code}_{timestamp}_thumb.jpg")
                res_thumb = requests.get(thumbnail_url, timeout=15)
                if res_thumb.status_code == 200:
                    with open(temp_thumb_path, "wb") as f:
                        f.write(res_thumb.content)
                    from carbon.species_detection import detect_species
                    species_preds = detect_species([temp_thumb_path])
                    print(f"[RECALCULATE] Pl@ntNet retry result: {species_preds}")
                    try:
                        os.remove(temp_thumb_path)
                    except Exception:
                        pass
                else:
                    print(f"[RECALCULATE] Thumbnail download failed (HTTP {res_thumb.status_code})")
            except Exception as plant_err:
                print(f"[RECALCULATE] Pl@ntNet retry failed: {plant_err}")

        # Resolve wood density from species (same threshold as the automatic pipeline)
        wood_density = 0.6
        wood_density_source = "generic-default"
        SPECIES_CONFIDENCE_THRESHOLD = 30.0
        if species_preds and len(species_preds) > 0:
            top_pred = species_preds[0]
            if top_pred.get("confidence", 0.0) >= SPECIES_CONFIDENCE_THRESHOLD:
                sci_name = top_pred.get("scientific_name")
                try:
                    from carbon.wood_density_lookup import get_wood_density
                    specific_wd = get_wood_density(sci_name)
                    if specific_wd is not None:
                        wood_density = specific_wd
                        wood_density_source = "species-matched"
                        print(f"[RECALCULATE] Wood density matched for {sci_name}: {wood_density}")
                except Exception as wd_err:
                    print(f"[RECALCULATE ERROR] Wood density lookup: {wd_err}")

        # Fallback to existing if not matched to specific species
        if wood_density_source == "generic-default" and target_scan.get("wood_density_used"):
            wood_density = target_scan.get("wood_density_used")
            wood_density_source = target_scan.get("wood_density_source") or "generic-default"

        forest_type = "moist"
        climate_zone = target_scan.get("climate_zone_detected") or "Unknown"
        if climate_zone != "Unknown":
            try:
                from carbon.climate_zone import classify_koppen_to_forest_type
                forest_type = classify_koppen_to_forest_type(climate_zone)
            except Exception as e:
                print(f"[RECALCULATE ERROR] Failed to map climate: {e}")

        # Height is derived from the (regenerated) point cloud by extract_dbh_with_2d_clicks,
        # i.e. a system height — validate like the automatic pipeline; fallback to DBH-only
        # if the truncated trunk does not represent total tree height.
        hinfo = resolve_height_usage(local_ply_path, res_override["height_m"],
                                      height_input_source="system", scale_factor=scale_factor)

        carbon_result = estimate_carbon(
            dbh_cm=res_override["dbh_cm"],
            height_m=hinfo["height_for_formula"],
            wood_density=wood_density,
            forest_type=forest_type
        )

        # ── Recalculate Quality Status ────────────────────────────────────────
        quality_status = "ok"
        inlier_ratio = res_override.get("inlier_ratio", 1.0)
        invalid_orientation = res_override.get("invalid_orientation", False)
        
        if res_override.get("slice_points_count", 0) < 10:
            quality_status = "low_points"
        elif invalid_orientation:
            quality_status = "invalid_orientation"
        elif res_override.get("mean_fit_error_cm", 0.0) > 10.0:
            quality_status = "high_fit_error"
        elif inlier_ratio < 0.15:
            quality_status = "low_inlier_ratio"

        # 9. Update target scan record in Cloudflare D1
        recalc_conf_note = res_override["confidence_note"]
        if not hinfo["height_validated"]:
            recalc_conf_note += (
                f" | {hinfo['height_validation_reason'] or hinfo['height_fallback_reason']}"
            )
        if not _is_cal2:
            recalc_conf_note += (
                " | UNKALIBRASI: skala default (PLY unit) dipakai — hasil TIDAK dapat "
                "diandalkan tanpa kalibrasi skala (auto-pose atau calibrate_scale.py)"
            )
        update_scan_result(
            scan_id=scan_id,
            dbh_cm=res_override["dbh_cm"],
            tinggi_m=res_override["height_m"],
            biomassa_kg=carbon_result["total_biomass_kg"],
            karbon_kg=carbon_result["carbon_kg"],
            co2e_kg=carbon_result["co2e_kg"],
            confidence_note=recalc_conf_note,
            geometry_3d=res_override["geometry_3d"],
            wood_density_used=wood_density,
            wood_density_source=wood_density_source,
            climate_zone_detected=climate_zone,
            formula_used=carbon_result["formula_used"],
            agb_kg=carbon_result["above_ground_biomass_kg"],
            bgb_kg=carbon_result["below_ground_biomass_kg"],
            gps_lat=target_scan.get("gps_lat"),
            gps_lon=target_scan.get("gps_lon"),
            species_predictions=species_preds,
            scale_status=("calibrated" if _is_cal2 else "uncalibrated"),
            scale_factor_used=scale_factor,
            calibration_source=_src2,
            height_used=hinfo["height_used"],
            total_height_used_m=hinfo["total_height_used_m"],
            segment_height_m=hinfo["segment_height_m"],
            height_fallback_reason=hinfo["height_fallback_reason"],
            height_validated=hinfo["height_validated"],
            height_validation_reason=hinfo["height_validation_reason"],
            quality_status=quality_status,
            inlier_ratio=inlier_ratio,
            root_to_shoot_ratio=carbon_result["root_to_shoot_ratio"],
            co2e_uncertainty_pct=carbon_result["co2e_uncertainty_pct"],
            co2e_low_kg=carbon_result["co2e_low_kg"],
            co2e_high_kg=carbon_result["co2e_high_kg"],
        )
        print(f"[RECALCULATE] Successfully updated D1 record id {scan_id} for {tree_code} via 2D clicks override.")

        # 10. Clean up temp files
        try:
            if os.path.exists(local_npy_path):
                os.remove(local_npy_path)
            if os.path.exists(local_ply_path):
                os.remove(local_ply_path)
        except Exception as cleanup_err:
            print(f"[RECALCULATE CLEANUP ERROR] {cleanup_err}")

        return {
            "success": True,
            "tree_code": tree_code,
            "dbh_cm": res_override["dbh_cm"],
            "height_m": res_override["height_m"],
            "biomassa_kg": carbon_result["total_biomass_kg"],
            "karbon_kg": carbon_result["carbon_kg"],
            "co2e_kg": carbon_result["co2e_kg"],
            "confidence_note": recalc_conf_note,
            "scale_status": ("calibrated" if _is_cal2 else "uncalibrated"),
            "height_used": hinfo["height_used"],
            "total_height_used_m": hinfo["total_height_used_m"],
            "segment_height_m": hinfo["segment_height_m"],
            "height_fallback_reason": hinfo["height_fallback_reason"],
            "height_validated": hinfo["height_validated"],
            "height_validation_reason": hinfo["height_validation_reason"],
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


class AdjustGeometryRequest(BaseModel):
    center_x: float
    center_y: float
    center_z: float
    dir_x: float
    dir_y: float
    dir_z: float
    radius_units: float
    h_min: float
    h_max: float
    h_target: float

@app.patch("/scan/{scan_id}/adjust-geometry", summary="Manually update CAD cylinder geometry and recalculate metrics")
async def adjust_geometry(scan_id: int, body: AdjustGeometryRequest):
    try:
        from storage.d1_client import execute_d1_query, update_scan_result
        from carbon.allometric import estimate_carbon

        # 1. Fetch target scan record by scan_id
        sql = "SELECT * FROM tree_scans WHERE id = ?"
        scans = execute_d1_query(sql, [scan_id])
        if not scans:
            raise HTTPException(status_code=404, detail="Scan record not found")
        target_scan = scans[0]
        tree_code = target_scan.get("tree_code")
        # Use the same scan-id-aware loader as every other endpoint for consistency.
        scale_factor, _is_cal3, _src3 = _load_scale_factor_for_scan(tree_code)

        # Derive pointmap (.npy) and points3d (.ply) URLs from splat_file_url for re-slicing
        splat_file_url = target_scan.get("splat_file_url")
        local_ply_path = ""
        if splat_file_url:
            base_url, filename = splat_file_url.rsplit("/", 1)
            name_parts = filename.split("_", 1)
            if len(name_parts) == 2:
                timestamp = name_parts[0]
            else:
                timestamp = filename.split(".")[0]
            
            points3d_url = f"{base_url}/{timestamp}_points3d.ply"

            local_dir = os.path.join(UPLOAD_DIR, "recalculates")
            os.makedirs(local_dir, exist_ok=True)
            local_ply_path = os.path.join(local_dir, f"{tree_code}_{timestamp}_points3d.ply")

            if not os.path.exists(local_ply_path) or os.path.getsize(local_ply_path) < 1000:
                import requests
                print(f"[ADJUST GEOMETRY] Downloading points3d.ply from {points3d_url}")
                try:
                    res_ply = requests.get(points3d_url, timeout=15)
                    if res_ply.status_code == 200:
                        with open(local_ply_path, "wb") as f:
                            f.write(res_ply.content)
                except Exception as ply_err:
                    print(f"[ADJUST GEOMETRY ERROR] Failed to download points3d.ply: {ply_err}")

        # Recalculate slice points based on the new cylinder position
        slice_points_3d = []
        if local_ply_path and os.path.exists(local_ply_path):
            try:
                from carbon.dbh_extractor import parse_ply_points
                points = parse_ply_points(local_ply_path)
                if len(points) > 0:
                    v = np.array([body.dir_x, body.dir_y, body.dir_z])
                    v_norm = np.linalg.norm(v)
                    if v_norm > 1e-6:
                        v = v / v_norm
                    else:
                        v = np.array([0.0, -1.0, 0.0])

                    P = np.array([body.center_x, body.center_y, body.center_z])
                    w = points - P
                    h_proj = np.dot(w, v)
                    d_proj = np.linalg.norm(w - h_proj[:, np.newaxis] * v[np.newaxis, :], axis=-1)

                    total_h = body.h_max - body.h_min
                    base_tol = 0.08 / scale_factor
                    tol = min(base_tol, total_h * 0.1)
                    tol = max(tol, 0.02)

                    # Slice at the top/center plane of the new cylinder
                    slice_mask = (np.abs(h_proj) <= tol) & (np.abs(d_proj - body.radius_units) <= tol * 1.5)
                    pts_slice = points[slice_mask]
                    if len(pts_slice) > 500:
                        rng = np.random.default_rng(42)
                        idx = rng.choice(len(pts_slice), size=500, replace=False)
                        pts_slice = pts_slice[idx]
                    
                    slice_points_3d = pts_slice.tolist()
                    print(f"[ADJUST GEOMETRY] Recalculated {len(slice_points_3d)} slice points around new cylinder.")
            except Exception as slice_err:
                import traceback
                with open("scratch/adjust_error.log", "w") as f_err:
                    traceback.print_exc(file=f_err)
                print(f"[ADJUST GEOMETRY ERROR] Failed to recalculate slice points: {slice_err}")

        # 2. Recalculate metrics based on manual coordinates
        dbh_m = body.radius_units * 2.0 * scale_factor
        dbh_cm = dbh_m * 100.0
        
        # Height is height span along the trunk axis, EXPLICITLY provided by the user
        # via the 3D transform controls (h_min/h_max) — i.e. a manual height input.
        height_m = (body.h_max - body.h_min) * scale_factor

        # 3. Resolve wood density from scan record
        wood_density = target_scan.get("wood_density_used") or 0.6
        wood_density_source = target_scan.get("wood_density_source") or "generic-default"

        # 4. Resolve climate and forest type
        forest_type = "moist"
        climate_zone = target_scan.get("climate_zone_detected") or "Unknown"
        if climate_zone != "Unknown":
            try:
                from carbon.climate_zone import classify_koppen_to_forest_type
                forest_type = classify_koppen_to_forest_type(climate_zone)
            except Exception as e:
                print(f"[ADJUST GEOMETRY] Failed to map climate: {e}")

        # User explicitly supplied the height: honour it (no forced DBH-only fallback)
        # but flag it clearly as un-validated — the system did not verify it against
        # the reconstructed point cloud.
        from carbon.dbh_extractor import resolve_height_usage
        hinfo = resolve_height_usage(None, height_m,
                                      height_input_source="manual", scale_factor=scale_factor)

        # 5. Estimate carbon
        carbon_result = estimate_carbon(
            dbh_cm=dbh_cm,
            height_m=hinfo["height_for_formula"],
            wood_density=wood_density,
            forest_type=forest_type
        )


        # 6. Reconstruct updated geometry_3d dictionary
        geometry_3d = {
            "center_x":       float(round(body.center_x, 4)),
            "center_y":       float(round(body.center_y, 4)),
            "center_z":       float(round(body.center_z, 4)),
            "dir_x":          float(round(body.dir_x, 4)),
            "dir_y":          float(round(body.dir_y, 4)),
            "dir_z":          float(round(body.dir_z, 4)),
            "radius_units":   float(round(body.radius_units, 4)),
            "h_min":          float(round(body.h_min, 4)),
            "h_max":          float(round(body.h_max, 4)),
            "h_target":       float(round(body.h_target, 4)),
            "scale_factor":   scale_factor,
            "method":         "Manual transform controls adjustment"
        }
        if slice_points_3d:
            geometry_3d["slice_points_3d"] = slice_points_3d
        else:
            geometry_3d["slice_points_3d"] = []

        # 8. Safely parse species_predictions to avoid double serialization in update_scan_result
        species_preds = target_scan.get("species_predictions")
        if isinstance(species_preds, str):
            try:
                species_preds = json.loads(species_preds)
            except Exception:
                pass

        # 9. Update target scan record in Cloudflare D1
        adj_conf_note = "Manually adjusted via 3D Transform Controls"
        if not hinfo["height_validated"]:
            adj_conf_note += f" | {hinfo['height_validation_reason']}"
        if not _is_cal3:
            adj_conf_note += (
                " | UNKALIBRASI: skala default (PLY unit) dipakai — hasil TIDAK dapat "
                "diandalkan tanpa kalibrasi skala (auto-pose atau calibrate_scale.py)"
            )
        update_scan_result(
            scan_id=scan_id,
            dbh_cm=float(round(dbh_cm, 2)),
            tinggi_m=float(round(height_m, 2)),
            biomassa_kg=carbon_result["total_biomass_kg"],
            karbon_kg=carbon_result["carbon_kg"],
            co2e_kg=carbon_result["co2e_kg"],
            confidence_note=adj_conf_note,
            geometry_3d=geometry_3d,
            wood_density_used=wood_density,
            wood_density_source=wood_density_source,
            climate_zone_detected=climate_zone,
            formula_used=carbon_result["formula_used"],
            agb_kg=carbon_result["above_ground_biomass_kg"],
            bgb_kg=carbon_result["below_ground_biomass_kg"],
            gps_lat=target_scan.get("gps_lat"),
            gps_lon=target_scan.get("gps_lon"),
            species_predictions=species_preds,
            scale_status=("calibrated" if _is_cal3 else "uncalibrated"),
            scale_factor_used=scale_factor,
            calibration_source=_src3,
            height_used=hinfo["height_used"],
            total_height_used_m=hinfo["total_height_used_m"],
            segment_height_m=hinfo["segment_height_m"],
            height_fallback_reason=hinfo["height_fallback_reason"],
            height_validated=hinfo["height_validated"],
            height_validation_reason=hinfo["height_validation_reason"],
            quality_status=target_scan.get("quality_status") or "ok",
            root_to_shoot_ratio=carbon_result["root_to_shoot_ratio"],
            co2e_uncertainty_pct=carbon_result["co2e_uncertainty_pct"],
            co2e_low_kg=carbon_result["co2e_low_kg"],
            co2e_high_kg=carbon_result["co2e_high_kg"],
        )
        print(f"[ADJUST GEOMETRY] Successfully updated D1 record id {scan_id} for {tree_code} via manual adjustment.")

        return {
            "success": True,
            "tree_code": tree_code,
            "dbh_cm": float(round(dbh_cm, 2)),
            "height_m": float(round(height_m, 2)),
            "biomassa_kg": carbon_result["total_biomass_kg"],
            "karbon_kg": carbon_result["carbon_kg"],
            "co2e_kg": carbon_result["co2e_kg"],
            "scale_status": ("calibrated" if _is_cal3 else "uncalibrated"),
            "height_used": hinfo["height_used"],
            "total_height_used_m": hinfo["total_height_used_m"],
            "segment_height_m": hinfo["segment_height_m"],
            "height_fallback_reason": hinfo["height_fallback_reason"],
            "height_validated": hinfo["height_validated"],
            "height_validation_reason": hinfo["height_validation_reason"],
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# ── Auth & Plot Endpoints ─────────────────────────────────────────────────────
from fastapi import Response

COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() == "true"
COOKIE_SAMESITE = os.environ.get("COOKIE_SAMESITE", "none").lower()

class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None

class GridPositionItem(BaseModel):
    tree_code: str
    grid_position_x: int
    grid_position_y: int

class PlotAreaItem(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    x1: int
    y1: int
    x2: int
    y2: int

class SaveLayoutRequest(BaseModel):
    layout: list[GridPositionItem]
    area_x1: Optional[int] = None
    area_y1: Optional[int] = None
    area_x2: Optional[int] = None
    area_y2: Optional[int] = None
    areas: Optional[list[PlotAreaItem]] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class CreatePlotRequest(BaseModel):
    name: str
    description: Optional[str] = None
    privacy: Optional[str] = "private"
    gps_centroid_lat: Optional[float] = None
    gps_centroid_lon: Optional[float] = None
    target_co2e_kg: Optional[float] = None
    area_x1: Optional[int] = None
    area_y1: Optional[int] = None
    area_x2: Optional[int] = None
    area_y2: Optional[int] = None

class UpdatePlotRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    privacy: Optional[str] = None
    gps_centroid_lat: Optional[float] = None
    gps_centroid_lon: Optional[float] = None
    target_co2e_kg: Optional[float] = None
    area_x1: Optional[int] = None
    area_y1: Optional[int] = None
    area_x2: Optional[int] = None
    area_y2: Optional[int] = None

class ClaimScanRequest(BaseModel):
    tree_code: str

class RemoveScanRequest(BaseModel):
    tree_code: str

# ── Auth Endpoints ────────────────────────────────────────────────────────────

@app.post("/auth/register", summary="Register a new user")
async def register_user(body: RegisterRequest):
    username = body.username.strip().lower()
    if not username or not body.password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    
    from storage.d1_client import execute_d1_query
    # Check if username exists
    existing = execute_d1_query("SELECT id FROM users WHERE username = ?", [username])
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Hash password
    from storage.auth_utils import hash_password
    pwd_hash, pwd_salt = hash_password(body.password)
    
    created_at = datetime.now(timezone.utc).isoformat()
    display_name = body.display_name or body.username
    
    sql = """
    INSERT INTO users (username, password_hash, password_salt, display_name, is_demo_account, created_at)
    VALUES (?, ?, ?, ?, 0, ?)
    """
    execute_d1_query(sql, [username, pwd_hash, pwd_salt, display_name, created_at])
    
    return {"success": True, "message": "User registered successfully"}


@app.post("/auth/login", summary="Login and establish httpOnly session cookie")
async def login_user(body: LoginRequest, response: Response):
    username = body.username.strip().lower()
    from storage.d1_client import execute_d1_query
    users = execute_d1_query("SELECT * FROM users WHERE username = ?", [username])
    if not users:
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    user = users[0]
    from storage.auth_utils import verify_password
    if not verify_password(body.password, user["password_hash"], user["password_salt"]):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    # Create session
    token = secrets.token_urlsafe(32)
    created_at = datetime.now(timezone.utc).isoformat()
    expires_at = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    
    execute_d1_query(
        "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        [token, user["id"], created_at, expires_at]
    )
    
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=7 * 24 * 60 * 60, # 7 days
        path="/"
    )
    
    return {
        "success": True,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "is_demo_account": bool(user["is_demo_account"])
        }
    }


@app.post("/auth/logout", summary="Logout and invalidate session token")
async def logout_user(response: Response, session_token: Optional[str] = Cookie(None)):
    if session_token:
        from storage.d1_client import execute_d1_query
        execute_d1_query("DELETE FROM sessions WHERE token = ?", [session_token])
    
    response.delete_cookie(
        key="session_token",
        path="/",
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE
    )
    return {"success": True, "message": "Logged out successfully"}


@app.get("/auth/me", summary="Get logged-in user profile")
async def me_profile(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "display_name": current_user["display_name"],
        "is_demo_account": bool(current_user["is_demo_account"])
    }


# ── Plot Endpoints ────────────────────────────────────────────────────────────

@app.post("/plots", summary="Create a new plot")
async def create_plot(body: CreatePlotRequest, current_user: dict = Depends(get_current_user)):
    from storage.d1_client import execute_d1_query
    
    # Generate PLOT-XXXX
    import string
    chars = string.ascii_uppercase + string.digits
    while True:
        code = "PLOT-" + "".join(secrets.choice(chars) for _ in range(4))
        # check uniqueness
        existing = execute_d1_query("SELECT id FROM plots WHERE plot_code = ?", [code])
        if not existing:
            break
            
    created_at = datetime.now(timezone.utc).isoformat()
    sql = """
    INSERT INTO plots (plot_code, owner_user_id, name, description, privacy, gps_centroid_lat, gps_centroid_lon, session_active, created_at, updated_at, target_co2e_kg, area_x1, area_y1, area_x2, area_y2)
    VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)
    """
    execute_d1_query(sql, [
        code,
        current_user["id"],
        body.name,
        body.description,
        body.privacy or "private",
        body.gps_centroid_lat,
        body.gps_centroid_lon,
        created_at,
        created_at,
        body.target_co2e_kg,
        body.area_x1,
        body.area_y1,
        body.area_x2,
        body.area_y2
    ])
    
    return {"success": True, "plot_code": code}


@app.get("/plots/{plot_code}", summary="Get plot details, owner info, and statistics aggregation")
async def get_plot(plot_code: str, optional_user: Optional[dict] = Depends(get_optional_user)):
    import math
    from storage.d1_client import execute_d1_query, populate_scan_defaults
    
    # Fetch plot
    plots = execute_d1_query("SELECT * FROM plots WHERE plot_code = ?", [plot_code])
    if not plots:
        raise HTTPException(status_code=404, detail="Plot not found")
        
    plot = plots[0]
    is_owner = optional_user and optional_user["id"] == plot["owner_user_id"]
    
    if plot["privacy"] == "private" and not is_owner:
        raise HTTPException(status_code=403, detail="Access denied")
        
    # Fetch owner info
    owners = execute_d1_query("SELECT username, display_name FROM users WHERE id = ?", [plot["owner_user_id"]])
    owner_info = owners[0] if owners else {"username": "unknown", "display_name": "Unknown User"}
    
    # Fetch scans in this plot
    scans = execute_d1_query("SELECT * FROM tree_scans WHERE plot_id = ?", [plot["id"]])
    
    total_co2e_kg = 0.0
    sum_sigma_sq = 0.0
    valid_scans = []
    
    for r in scans:
        if r.get("geometry_3d"):
            try:
                r["geometry_3d"] = json.loads(r["geometry_3d"])
            except Exception:
                pass
        if r.get("species_predictions"):
            try:
                r["species_predictions"] = json.loads(r["species_predictions"])
            except Exception:
                pass
        populate_scan_defaults(r)
        valid_scans.append(r)
        
        co2e = r.get("co2e_kg") or 0.0
        unc_pct = r.get("co2e_uncertainty_pct")
        if unc_pct is None:
            unc_pct = 15.0 # fallback default uncertainty
            
        total_co2e_kg += co2e
        sigma = co2e * (unc_pct / 100.0)
        sum_sigma_sq += sigma ** 2
        
    combined_uncertainty_kg = math.sqrt(sum_sigma_sq)
    combined_uncertainty_pct = (100.0 * combined_uncertainty_kg / total_co2e_kg) if total_co2e_kg > 0 else 0.0
    
    # Fetch plot areas
    areas = execute_d1_query("SELECT id, name, x1, y1, x2, y2 FROM plot_areas WHERE plot_id = ?", [plot["id"]])
    areas_list = [{
        "id": a["id"],
        "name": a.get("name"),
        "x1": a["x1"],
        "y1": a["y1"],
        "x2": a["x2"],
        "y2": a["y2"]
    } for a in areas]
    
    return {
        "success": True,
        "plot": {
            "id": plot["id"],
            "plot_code": plot["plot_code"],
            "name": plot["name"],
            "description": plot["description"],
            "privacy": plot["privacy"],
            "gps_centroid_lat": plot["gps_centroid_lat"],
            "gps_centroid_lon": plot["gps_centroid_lon"],
            "session_active": bool(plot["session_active"]),
            "created_at": plot["created_at"],
            "updated_at": plot["updated_at"],
            "target_co2e_kg": plot["target_co2e_kg"],
            "area_x1": plot.get("area_x1"),
            "area_y1": plot.get("area_y1"),
            "area_x2": plot.get("area_x2"),
            "area_y2": plot.get("area_y2"),
            "areas": areas_list,
            "owner": owner_info
        },
        "scans_count": len(valid_scans),
        "scans": valid_scans,
        "aggregation": {
            "total_co2e_kg": float(round(total_co2e_kg, 2)),
            "combined_uncertainty_kg": float(round(combined_uncertainty_kg, 2)),
            "combined_uncertainty_pct": float(round(combined_uncertainty_pct, 1))
        }
    }


@app.patch("/plots/{plot_id}", summary="Update plot metadata")
async def update_plot_details(plot_id: int, body: UpdatePlotRequest, current_user: dict = Depends(get_current_user)):
    from storage.d1_client import execute_d1_query
    
    plots = execute_d1_query("SELECT * FROM plots WHERE id = ?", [plot_id])
    if not plots:
        raise HTTPException(status_code=404, detail="Plot not found")
        
    plot = plots[0]
    if plot["owner_user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
        
    name = body.name if body.name is not None else plot["name"]
    description = body.description if body.description is not None else plot["description"]
    privacy = body.privacy if body.privacy is not None else plot["privacy"]
    gps_lat = body.gps_centroid_lat if body.gps_centroid_lat is not None else plot["gps_centroid_lat"]
    gps_lon = body.gps_centroid_lon if body.gps_centroid_lon is not None else plot["gps_centroid_lon"]
    target_co2e = body.target_co2e_kg if body.target_co2e_kg is not None else plot["target_co2e_kg"]
    if target_co2e is not None and target_co2e <= 0:
        target_co2e = None
    
    area_x1 = body.area_x1 if body.area_x1 is not None else plot["area_x1"]
    area_y1 = body.area_y1 if body.area_y1 is not None else plot["area_y1"]
    area_x2 = body.area_x2 if body.area_x2 is not None else plot["area_x2"]
    area_y2 = body.area_y2 if body.area_y2 is not None else plot["area_y2"]
    
    updated_at = datetime.now(timezone.utc).isoformat()
    
    sql = """
    UPDATE plots
    SET name = ?, description = ?, privacy = ?, gps_centroid_lat = ?, gps_centroid_lon = ?, target_co2e_kg = ?, area_x1 = ?, area_y1 = ?, area_x2 = ?, area_y2 = ?, updated_at = ?
    WHERE id = ?
    """
    execute_d1_query(sql, [name, description, privacy, gps_lat, gps_lon, target_co2e, area_x1, area_y1, area_x2, area_y2, updated_at, plot_id])
    
    return {"success": True, "message": "Plot updated successfully"}


@app.get("/users/{user_id}/plots", summary="List plots owned by a user")
async def list_user_plots(user_id: int, optional_user: Optional[dict] = Depends(get_optional_user)):
    from storage.d1_client import execute_d1_query
    is_owner = optional_user and optional_user["id"] == user_id
    
    if is_owner:
        plots = execute_d1_query("SELECT * FROM plots WHERE owner_user_id = ? ORDER BY created_at DESC", [user_id])
    else:
        plots = execute_d1_query("SELECT * FROM plots WHERE owner_user_id = ? AND privacy = 'public' ORDER BY created_at DESC", [user_id])
        
    result_plots = []
    for p in plots:
        scans = execute_d1_query("SELECT thumbnail_url, co2e_kg FROM tree_scans WHERE plot_id = ?", [p["id"]])
        scans_count = len(scans)
        total_co2e = sum((r.get("co2e_kg") or 0.0) for r in scans)
        thumbnails = [r.get("thumbnail_url") for r in scans if r.get("thumbnail_url")]
        thumbnails = thumbnails[:3]
        
        p_dict = dict(p)
        p_dict["scans_count"] = scans_count
        p_dict["total_co2e_kg"] = float(round(total_co2e, 2))
        p_dict["thumbnails"] = thumbnails
        result_plots.append(p_dict)
        
    return {"success": True, "plots": result_plots}


@app.get("/plots", summary="List all public plots")
async def list_public_plots():
    from storage.d1_client import execute_d1_query
    plots = execute_d1_query("SELECT * FROM plots WHERE privacy = 'public' ORDER BY created_at DESC")
    
    result_plots = []
    for p in plots:
        scans = execute_d1_query("SELECT thumbnail_url, co2e_kg FROM tree_scans WHERE plot_id = ?", [p["id"]])
        scans_count = len(scans)
        total_co2e = sum((r.get("co2e_kg") or 0.0) for r in scans)
        thumbnails = [r.get("thumbnail_url") for r in scans if r.get("thumbnail_url")]
        thumbnails = thumbnails[:3]
        
        p_dict = dict(p)
        p_dict["scans_count"] = scans_count
        p_dict["total_co2e_kg"] = float(round(total_co2e, 2))
        p_dict["thumbnails"] = thumbnails
        result_plots.append(p_dict)
        
    return {"success": True, "plots": result_plots}


@app.get("/users/{user_id}/scans", summary="List all scans claimed by a user")
async def list_user_scans(user_id: int, optional_user: Optional[dict] = Depends(get_optional_user)):
    from storage.d1_client import execute_d1_query, populate_scan_defaults
    is_owner = optional_user and optional_user["id"] == user_id
    if not is_owner:
        raise HTTPException(status_code=403, detail="Access denied")
        
    scans = execute_d1_query("SELECT * FROM tree_scans WHERE claimed_by_user_id = ? ORDER BY scan_date DESC", [user_id])
    
    for r in scans:
        if r.get("geometry_3d"):
            try:
                r["geometry_3d"] = json.loads(r["geometry_3d"])
            except Exception:
                pass
        if r.get("species_predictions"):
            try:
                r["species_predictions"] = json.loads(r["species_predictions"])
            except Exception:
                pass
        populate_scan_defaults(r)
        
    return {"success": True, "scans": scans}


@app.delete("/scans/{tree_code}", summary="Delete a scan record and its R2 files")
async def delete_scan(tree_code: str, current_user: dict = Depends(get_current_user)):
    from storage.d1_client import execute_d1_query
    
    # 1. Fetch the scan
    scans = execute_d1_query("SELECT * FROM tree_scans WHERE tree_code = ?", [tree_code])
    if not scans:
        raise HTTPException(status_code=404, detail="Scan not found")
    scan = scans[0]
    
    # 2. Check ownership/claim status
    if scan.get("claimed_by_user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="You do not have permission to delete this scan")
        
    # 3. Clean up files on R2 storage
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("R2_BUCKET_NAME")
    public_url_prefix = os.environ.get("R2_PUBLIC_URL_PREFIX", "").rstrip("/")
    
    if all([account_id, access_key, secret_key, bucket_name]):
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(signature_version="s3v4"),
                region_name="auto"
            )
            
            # Delete PLY/Splat file from R2
            ply_url = scan.get("splat_file_url")
            if ply_url:
                key = ply_url.replace(public_url_prefix + "/", "") if public_url_prefix else ply_url
                key = key.lstrip("/")
                await asyncio.to_thread(s3.delete_object, Bucket=bucket_name, Key=key)
                
            # Delete Thumbnail file from R2
            thumb_url = scan.get("thumbnail_url")
            if thumb_url:
                key = thumb_url.replace(public_url_prefix + "/", "") if public_url_prefix else thumb_url
                key = key.lstrip("/")
                await asyncio.to_thread(s3.delete_object, Bucket=bucket_name, Key=key)
        except Exception as e:
            # Log but don't fail HTTP request if R2 delete fails
            print(f"[DELETE-SCAN] Failed to delete from R2: {e}")
            
    # 4. Delete database record
    execute_d1_query("DELETE FROM tree_scans WHERE tree_code = ?", [tree_code])
    
    return {"success": True, "message": f"Successfully deleted scan {tree_code}"}


@app.post("/plots/{plot_id}/claim-scan", summary="Claim an anonymous scan into a user plot")
async def claim_scan(plot_id: int, body: ClaimScanRequest, current_user: dict = Depends(get_current_user)):
    from storage.d1_client import execute_d1_query
    
    plots = execute_d1_query("SELECT * FROM plots WHERE id = ?", [plot_id])
    if not plots:
        raise HTTPException(status_code=404, detail="Plot not found")
    plot = plots[0]
    if plot["owner_user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
        
    scans = execute_d1_query("SELECT * FROM tree_scans WHERE tree_code = ?", [body.tree_code])
    if not scans:
        raise HTTPException(status_code=404, detail="Scan not found")
        
    scan = scans[0]
    if scan.get("claimed_by_user_id") is not None:
        raise HTTPException(status_code=409, detail="Scan has already been claimed by another user")
        
    execute_d1_query(
        "UPDATE tree_scans SET plot_id = ?, claimed_by_user_id = ? WHERE tree_code = ?",
        [plot_id, current_user["id"], body.tree_code]
    )
    
    # Auto update centroid if not set
    if plot["gps_centroid_lat"] is None and scan.get("gps_lat") is not None:
        execute_d1_query(
            "UPDATE plots SET gps_centroid_lat = ?, gps_centroid_lon = ? WHERE id = ?",
            [scan["gps_lat"], scan["gps_lon"], plot_id]
        )
        
    return {"success": True, "message": f"Successfully claimed scan {body.tree_code}"}


@app.post("/plots/{plot_id}/remove-scan", summary="Remove a scan from a plot")
async def remove_scan(plot_id: int, body: RemoveScanRequest, current_user: dict = Depends(get_current_user)):
    from storage.d1_client import execute_d1_query
    
    plots = execute_d1_query("SELECT * FROM plots WHERE id = ?", [plot_id])
    if not plots:
        raise HTTPException(status_code=404, detail="Plot not found")
    plot = plots[0]
    if plot["owner_user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
        
    scans = execute_d1_query("SELECT * FROM tree_scans WHERE tree_code = ? AND plot_id = ?", [body.tree_code, plot_id])
    if not scans:
        raise HTTPException(status_code=404, detail="Scan not found in this plot")
        
    execute_d1_query(
        "UPDATE tree_scans SET plot_id = NULL, grid_position_x = NULL, grid_position_y = NULL WHERE tree_code = ?",
        [body.tree_code]
    )
    
    return {"success": True, "message": f"Successfully removed scan {body.tree_code} from plot"}



@app.post("/plots/{plot_id}/layout", summary="Save visual tree grid positions layout for a plot")
async def save_plot_layout(plot_id: int, body: SaveLayoutRequest, current_user: dict = Depends(get_current_user)):
    from storage.d1_client import execute_d1_query
    
    # 1. Fetch plot & verify ownership
    plots = execute_d1_query("SELECT * FROM plots WHERE id = ?", [plot_id])
    if not plots:
        raise HTTPException(status_code=404, detail="Plot not found")
    plot = plots[0]
    if plot["owner_user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
        
    # 2. Update positions in batch
    for item in body.layout:
        execute_d1_query(
            "UPDATE tree_scans SET grid_position_x = ?, grid_position_y = ? WHERE plot_id = ? AND tree_code = ?",
            [item.grid_position_x, item.grid_position_y, plot_id, item.tree_code]
        )
        
    # 3. Update plot bounds
    execute_d1_query(
        "UPDATE plots SET area_x1 = ?, area_y1 = ?, area_x2 = ?, area_y2 = ?, updated_at = ? WHERE id = ?",
        [body.area_x1, body.area_y1, body.area_x2, body.area_y2, datetime.now(timezone.utc).isoformat(), plot_id]
    )
    
    # 4. Update plot areas list
    if body.areas is not None:
        # Delete existing plot areas
        execute_d1_query("DELETE FROM plot_areas WHERE plot_id = ?", [plot_id])
        # Insert the new list of areas
        for a in body.areas:
            execute_d1_query(
                "INSERT INTO plot_areas (plot_id, name, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?, ?)",
                [plot_id, a.name, a.x1, a.y1, a.x2, a.y2]
            )
        
    return {"success": True, "message": "Layout saved successfully"}


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)
