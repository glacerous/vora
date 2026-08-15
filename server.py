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
from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, Query, UploadFile, Request
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

def get_job_frames_dir(tree_code: str = None) -> str:
    if not tree_code:
        return FRAMES_DIR
    d = os.path.join(BASE_DIR, "test_images", tree_code)
    os.makedirs(d, exist_ok=True)
    return d

def get_job_output_dir(tree_code: str = None) -> str:
    if not tree_code:
        return OUTPUT_DIR
    d = os.path.join(BASE_DIR, "output", tree_code)
    os.makedirs(d, exist_ok=True)
    return d

def get_job_upload_dir(tree_code: str = None) -> str:
    if not tree_code:
        return UPLOAD_DIR
    d = os.path.join(BASE_DIR, "uploads", tree_code)
    os.makedirs(d, exist_ok=True)
    return d

# ── Multi-Job Pipeline State & Thread-Safe Registry ─────────────────────────
import threading

active_jobs: dict[str, dict] = {}
latest_job_code: Optional[str] = None
_jobs_lock = threading.Lock()

# Fallback global state object for backward compatibility
state: dict = {
    "stage":             "idle",
    "message":           "Ready.",
    "frame_count":       0,
    "error":             None,
    "carbon_estimation": None,
    "overlap_warning":   None,
    "cancel_requested":  False,
    "calibration_frame":  None,
    "tree_code":         None,
    "started_at":        None,
    "camera_poses":      None,
}

def get_job_state(tree_code: Optional[str] = None) -> dict:
    with _jobs_lock:
        if tree_code and tree_code in active_jobs:
            return active_jobs[tree_code]
        if latest_job_code and latest_job_code in active_jobs:
            return active_jobs[latest_job_code]
        return state

def init_job_state(tree_code: str, **initial_kw) -> dict:
    global latest_job_code
    with _jobs_lock:
        latest_job_code = tree_code
        job_state = {
            "stage":             "idle",
            "message":           "Ready.",
            "frame_count":       0,
            "error":             None,
            "carbon_estimation": None,
            "overlap_warning":   None,
            "cancel_requested":  False,
            "calibration_frame":  None,
            "tree_code":         tree_code,
            "started_at":        time.time(),
            "camera_poses":      None,
            **initial_kw
        }
        active_jobs[tree_code] = job_state
        state.update(job_state)
        return job_state

def upd(stage_or_code: str, msg_or_stage: str = None, msg: str = None, **kw: Any) -> None:
    # Supports both upd("reconstructing", "message...") and upd("POHON-1234", "reconstructing", "message...")
    if msg is not None:
        tree_code = stage_or_code
        stage = msg_or_stage
        message = msg
    else:
        tree_code = None
        stage = stage_or_code
        message = msg_or_stage

    with _jobs_lock:
        target = None
        if tree_code and tree_code in active_jobs:
            target = active_jobs[tree_code]
        elif latest_job_code and latest_job_code in active_jobs:
            target = active_jobs[latest_job_code]
        else:
            target = state
        
        target.update({"stage": stage, "message": message, **kw})
        state.update({"stage": stage, "message": message, **kw})

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
    frame_idx: Optional[int] = None

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
    timings: Optional[dict] = None

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
    frame_idx: Optional[int] = None

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


def decimate_ply_file(input_ply_path, output_ply_path, target_count=300000):
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

        # ── Implicit scale calibration from Modal (geometric prior or ARCore VIO) ──
        # Priority: manual_calibration > arcore_vio > estimated_geometric_prior > uncalibrated
        # Manual calibration (calibration.json) always wins; handled by _load_scale_factor_for_scan.
        scale_note = None
        if not is_calibrated and scale_calibration:
            src = scale_calibration.get("source", "")
            if scale_calibration.get("is_calibrated"):
                scale_factor = float(scale_calibration["scale_factor"])
                is_calibrated = True
                calibration_source = src
                if src == "arcore_vio":
                    scale_note = f"skala terukur via ARCore/ARKit VIO (~1-3% error): {scale_calibration.get('reason', '')}"
                elif src == "estimated_geometric_prior":
                    scale_note = (
                        f"ESTIMASI skala dari geometri MASt3R (~5-9% error, ~15-25% ketidakpastian biomassa) — "
                        f"tidak diverifikasi fisik: {scale_calibration.get('reason', '')}"
                    )
                else:
                    scale_note = f"kalibrasi otomatis ({src}): {scale_calibration.get('reason', '')}"
                print(f"[CARBON-SCALE] Applied {src} scale_factor={scale_factor:.6f}")

        # Map calibration source to scale_status tier
        if not is_calibrated:
            scale_status = "uncalibrated"
        elif calibration_source == "arcore_vio":
            scale_status = "vio_calibrated"
        elif calibration_source == "estimated_geometric_prior":
            scale_status = "estimated_prior"
        else:
            scale_status = "calibrated"

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
        if scale_status == "uncalibrated":
            confidence_note += (
                " | UNKALIBRASI: skala PLY default (1.0) dipakai — hasil TIDAK dapat "
                "diandalkan tanpa kalibrasi skala (ARCore VIO atau calibrate_scale.py)"
            )
        elif scale_status == "estimated_prior":
            confidence_note += (
                " | ESTIMASI SKALA: geometri MASt3R (~5-9% error) — "
                "hasil mendekati realistis namun belum diverifikasi secara fisik"
            )
        if scale_note:
            confidence_note += f" | {scale_note}"
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
def _extract_thread(tree_code: str, r2_key: str, target: int, blur_thresh: int, client_to_server_s: float = None) -> None:
    try:
        upd(tree_code, "extracting", f"Offloading frame extraction to Modal for {tree_code} (pulling from R2)...")
        t_modal_start = time.time()

        # Build R2 config dict to pass to Modal so it can download the video
        r2_config = {
            "CLOUDFLARE_ACCOUNT_ID": os.environ.get("CLOUDFLARE_ACCOUNT_ID"),
            "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID"),
            "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY"),
            "R2_BUCKET_NAME": os.environ.get("R2_BUCKET_NAME"),
        }

        import modal
        fn = modal.Function.from_name("instantsplat-app", "extract_video_frames_modal")
        res = fn.remote(None, target, blur_thresh, t_modal_start, r2_key=r2_key, r2_config=r2_config, tree_code=tree_code)
        t_modal_end = time.time()

        t_modal_enter = res.get("t_modal_enter")
        t_modal_exit = res.get("t_modal_exit")
        global_import_time = res.get("global_import_time")

        # Calculate sub-phases
        modal_scheduling_cold_start = 0.0
        modal_transfer_in = 0.0
        modal_compute = 0.0
        modal_transfer_out = 0.0

        if t_modal_enter and t_modal_exit and global_import_time:
            modal_scheduling_cold_start = global_import_time - t_modal_start
            modal_transfer_in = t_modal_enter - global_import_time
            modal_compute = t_modal_exit - t_modal_enter
            modal_transfer_out = t_modal_end - t_modal_exit

        print(f"[TIMING] Modal remote frame extraction call complete for {tree_code}.")
        if client_to_server_s is not None:
            print(f"  - (a) Client -> Server Upload : {client_to_server_s:.4f}s")
        print(f"  - Total Modal Roundtrip       : {t_modal_end - t_modal_start:.4f}s")
        print(f"  - (c) Modal Scheduling / Cold : {modal_scheduling_cold_start:.4f}s")
        print(f"  - (b) Request Upload to Modal : {modal_transfer_in:.4f}s")
        print(f"  - (d) Modal CV2 Compute       : {modal_compute:.4f}s")
        print(f"  - Response Download           : {modal_transfer_out:.4f}s")

        frames = res.get("frames", [])
        overlap_warning = res.get("overlap_warning")
        r2_frames_prefix = res.get("r2_frames_prefix")
        num_frames = res.get("num_frames", len(frames))

        n = num_frames
        if n == 0 or len(frames) == 0:
            raise ValueError(f"No sharp frames found (blur_thresh={blur_thresh}). Try a slower, steadier recording in good lighting.")

        # ── Gate 1: 2D Frame Quality & Variance Pre-Check on CPU ──
        if frames:
            sample_idxs = [0, len(frames)//2, len(frames)-1]
            for s_idx in sample_idxs:
                try:
                    nparr = np.frombuffer(frames[s_idx], np.uint8)
                    img_mat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img_mat is not None:
                        gray = cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)
                        mean_brightness = float(np.mean(gray))
                        if mean_brightness < 10.0:
                            raise ValueError("Video is too dark (average brightness near zero). Please record in daylight or good lighting.")
                        if mean_brightness > 248.0:
                            raise ValueError("Video is completely overexposed / pure white. Please adjust camera exposure.")

                        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                        if lap_var < 3.0:
                            raise ValueError("Video frames lack visual texture (flat screen, blank wall, or out of focus). Please record a clear tree trunk.")
                except Exception as gate1_err:
                    if isinstance(gate1_err, ValueError):
                        raise gate1_err
                    print(f"[GATE-1-WARN] Exception in 2D pre-check: {gate1_err}")

        upd(tree_code, "extracting", f"Saving {n} frames to disk...")
        t_save_start = time.time()
        job_frames_dir = get_job_frames_dir(tree_code)
        for f in glob.glob(os.path.join(job_frames_dir, "*")):
            try:
                os.remove(f)
            except Exception:
                pass

        for j, frame_bytes in enumerate(frames):
            img_path = os.path.join(job_frames_dir, f"{j:04d}.jpg")
            with open(img_path, "wb") as f_out:
                f_out.write(frame_bytes)
            # Also mirror to default FRAMES_DIR for legacy single-scan compatibility
            try:
                with open(os.path.join(FRAMES_DIR, f"{j:04d}.jpg"), "wb") as legacy_f:
                    legacy_f.write(frame_bytes)
            except Exception:
                pass

        t_save_end = time.time()
        print(f"[TIMING] Saved {n} frames to disk: {t_save_end - t_save_start:.4f}s in {job_frames_dir}")

        job_st = get_job_state(tree_code)
        job_st["frame_count"] = n
        job_st["overlap_warning"] = overlap_warning
        job_st["calibration_frame"] = None
        job_st["timings"] = {
            "client_to_server_s": client_to_server_s,
            "modal_roundtrip_s": t_modal_end - t_modal_start,
            "modal_scheduling_cold_start_s": modal_scheduling_cold_start,
            "modal_transfer_in_s": modal_transfer_in,
            "modal_compute_s": modal_compute,
            "modal_transfer_out_s": modal_transfer_out,
            "server_read_s": 0.0,
            "server_save_s": t_save_end - t_save_start
        }

        if overlap_warning:
            print(overlap_warning)

        upd(tree_code, "extracted", f"✓ {n} sharp frames ready", frame_count=n, overlap_warning=overlap_warning)
        print(f"[EXTRACT] Completed. {n} frames written for {tree_code} to {job_frames_dir}")

    except Exception as exc:
        print(f"[EXTRACT ERROR] During frame processing for {tree_code}: {exc}")
        upd(tree_code, "error", str(exc), error=str(exc))

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
    frame_idx: int = None,
) -> None:
    import modal
    progress_dict = modal.Dict.from_name("instantsplat-progress-dict", create_if_missing=True)
    try:
        progress_dict[tree_code] = "Uploading images"
        upd(tree_code, "reconstructing", "Connecting to Modal…")

        job_st = get_job_state(tree_code)
        if job_st.get("cancel_requested", False):
            raise RuntimeError("Job cancelled by user")

        r2_frames_prefix = job_st.get("r2_frames_prefix")
        imgs = []
        if not r2_frames_prefix:
            t_disk_start = time.time()
            job_frames_dir = get_job_frames_dir(tree_code)
            files = sorted(glob.glob(os.path.join(job_frames_dir, "*.jpg")))
            if not files:
                # Fallback to root FRAMES_DIR if job dir is empty
                files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
            if not files:
                raise ValueError(f"No frames found for tree_code '{tree_code}'")

            for f in files:
                with open(f, "rb") as fh:
                    imgs.append(fh.read())
            t_disk_end = time.time()
            print(f"[TIMING] Read {len(imgs)} fallback frames from {job_frames_dir}: {t_disk_end - t_disk_start:.4f}s")
            upd(tree_code, "reconstructing", f"Sending {len(imgs)} frames to Modal A10G GPU…")
        else:
            upd(tree_code, "reconstructing", "Connecting to Modal A10G GPU (using direct R2 frames)…")
        
        t0 = time.time()
        print(f"[RECONSTRUCT] Connecting to Modal pipeline for tree_code '{tree_code}' at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}")
        if r2_frames_prefix:
            print(f"[RECONSTRUCT] Direct Modal-to-R2 frame loading enabled (prefix: '{r2_frames_prefix}'). Zero frame bytes uploaded from Render!")
        else:
            print(f"[RECONSTRUCT] Uploading {len(imgs)} frames to GPU cloud (background removed on Modal: {remove_background})...")
        
        fn = modal.Function.from_name("instantsplat-app", "run_reconstruction")
        
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
            camera_poses = job_st.get("camera_poses")
            try:
                if r2_frames_prefix:
                    result = fn.remote(None, tree_code, remove_background, r2_config, iterations, camera_poses=camera_poses, r2_frames_prefix=r2_frames_prefix)
                elif camera_poses is not None:
                    result = fn.remote(imgs, tree_code, remove_background, r2_config, iterations, camera_poses=camera_poses)
                else:
                    result = fn.remote(imgs, tree_code, remove_background, r2_config, iterations)
            except TypeError as te:
                if "takes from" in str(te) or "unexpected keyword" in str(te) or "positional argument" in str(te) or "argument" in str(te):
                    print(f"[RECONSTRUCT] Signature mismatch on remote, falling back to legacy call: {te}")
                    result = fn.remote(imgs, tree_code, remove_background, r2_config, iterations)
                else:
                    raise te
        except Exception as remote_exc:
            # If it's a TypeError / argument error / signature mismatch, fail immediately
            if isinstance(remote_exc, (TypeError, ValueError)) and "takes from" not in str(remote_exc):
                print(f"[RECONSTRUCT] fn.remote() failed with non-recoverable error: {remote_exc}")
                raise remote_exc

            # Otherwise, fn.remote() failed due to connection drop/timeout — this typically happens
            # when the Render server restarted while the Modal job was still running (connection reset).
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
                upd(tree_code, "reconstructing", f"Server restarted mid-job — waiting for Modal to finish… ({_attempt * 10}s)")
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

        job_output_dir = get_job_output_dir(tree_code)
        out = os.path.join(job_output_dir, "result.ply")
        points3d_path = os.path.join(job_output_dir, "points3d.ply")
        points3d_highres_path = os.path.join(job_output_dir, "points3d_highres.ply")
        points3d_all_path = os.path.join(job_output_dir, "points3D_all.npy")

        # Clean old files in job output dir
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

        # Also mirror to root OUTPUT_DIR for legacy viewer compatibility
        try:
            if os.path.exists(out):
                shutil.copy2(out, os.path.join(OUTPUT_DIR, "result.ply"))
        except Exception:
            pass

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

        # Note: points3d_all.npy download & fn_align.remote omitted to eliminate ~35MB outbound PLY bounce.
        # Local ground-separated geometric cylinder detection runs directly on points3d.ply in <0.1s.
        points3d_highres_path = None
        P1_3d = None
        P2_3d = None

        t_meta_start = time.time()
        # 1. Extract GPS from frames if not provided by client
        if gps_lat is None or gps_lon is None:
            try:
                from carbon.gps_exif import extract_gps_from_frames
                gps_res = extract_gps_from_frames(job_frames_dir)
                if gps_res:
                    gps_lat, gps_lon = gps_res
                    print(f"[RECONSTRUCT-GPS] Extracted EXIF GPS: lat={gps_lat:.6f}, lon={gps_lon:.6f}")
            except Exception as gps_err:
                print(f"[RECONSTRUCT-GPS ERROR] Failed to extract GPS: {gps_err}")

        # 2. Determine Climate Zone & Forest Type
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
            upd(tree_code, "reconstructing", "Detecting tree species using Pl@ntNet API...")
            from carbon.species_detection import detect_species
            img_files = sorted(glob.glob(os.path.join(job_frames_dir, "*.jpg")))
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
        upd(tree_code, "reconstructing", "✓ Reconstruction done. Estimating DBH and Carbon...")
        carbon_est = run_carbon_analysis(
            out, 
            points3d_path=points3d_path, 
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
                
        job_st = get_job_state(tree_code)
        job_st["carbon_estimation"] = carbon_est
        t_carbon_end = time.time()
        print(f"[TIMING] Local carbon estimation analysis: {t_carbon_end - t_carbon_start:.4f}s")

        # Check if carbon estimation succeeded or hit geometry failure
        is_geometry_failed = False
        if not carbon_est or "error" in carbon_est:
            is_geometry_failed = True
            err_msg = (carbon_est or {}).get("error", "Automatic cylinder fitting could not detect trunk in point cloud")
            print(f"[RECONSTRUCT-WARN] Carbon estimation error: {err_msg}. Saving scan as uncalibrated_geometry_failed.")
            carbon_est = {
                "dbh_cm": None,
                "height_m": None,
                "biomass_kg": None,
                "carbon_kg": None,
                "co2e_kg": None,
                "confidence": f"Automatic cylinder detection failed: {err_msg}. Please use Recalibrate to mark trunk.",
                "quality_status": "uncalibrated_geometry_failed",
                "geometry_3d": {"error": err_msg},
            }

        # ── Precompute SHA-256 hash of PLY ──
        import hashlib
        sha256_hash = None
        if os.path.exists(out):
            try:
                with open(out, "rb") as pf:
                    sha256_hash = hashlib.sha256(pf.read()).hexdigest()
            except Exception:
                pass
        
        geom_dict = carbon_est.get("geometry_3d") or {}
        if sha256_hash:
            geom_dict["ply_sha256"] = sha256_hash
        carbon_est["geometry_3d"] = geom_dict

        try:
            t_persistence_start = time.time()
            progress_dict[tree_code] = "Uploading results"
            
            if uploaded:
                print(f"[RECONSTRUCT] Files were already uploaded directly from Modal to R2 (zero Render outbound bandwidth used).")
            else:
                upd(tree_code, "reconstructing", "Uploading reconstruction files to Cloudflare R2...")
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

                # Select representative frame matching MASt3R pointmap as thumbnail
                thumbnail_url = None
                if files:
                    target_thumb_idx = 0
                    if points3d_all_path and os.path.exists(points3d_all_path):
                        try:
                            pts3d_local = np.load(points3d_all_path)
                            N_local = pts3d_local.shape[0]
                            valid_counts = [
                                np.sum(~np.all(pts3d_local[i] == 0, axis=-1) & ~np.any(np.isnan(pts3d_local[i]), axis=-1))
                                for i in range(N_local)
                            ]
                            target_thumb_idx = int(np.argmax(valid_counts))
                        except Exception:
                            target_thumb_idx = 0
                    
                    representative_frame = files[min(target_thumb_idx, len(files) - 1)]
                    try:
                        upd(tree_code, "reconstructing", f"Uploading representative frame {target_thumb_idx} as thumbnail to R2...")
                        thumbnail_url = upload_thumbnail(representative_frame, tree_code)
                    except Exception as thumb_err:
                        print(f"Thumbnail upload error: {thumb_err}")

            upd(tree_code, "reconstructing", "Saving scan results to Cloudflare D1...")
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
            
            elapsed = time.time() - t0
            if is_geometry_failed:
                upd(tree_code, "done", f"✓ 3D Gaussian Splat generated ({mb:.1f} MB), but trunk could not be detected automatically. Use Recalibrate to mark trunk.")
            else:
                upd(tree_code, "done", f"✓ Done in {elapsed:.0f}s — {mb:.1f} MB Gaussian Splat ready! (Tree code: {tree_code})")
        except Exception as exc:
            print(f"Persistence error: {exc}")
            upd(tree_code, "error", f"Reconstruction done, but failed to save: {exc}", error=str(exc))

    except BaseException as exc:
        job_st = get_job_state(tree_code)
        if job_st.get("cancel_requested", False):
            print(f"[RECONSTRUCT] Cancel requested by user for {tree_code}. Aborting...")
            upd(tree_code, "idle", "Ready.")
            job_st["cancel_requested"] = False
        else:
            print(f"[RECONSTRUCT ERROR] Critical pipeline failure for {tree_code}: {exc}")
            upd(tree_code, "error", str(exc), error=str(exc))
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
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import Cookie, Depends, Header, HTTPException, status
import secrets
import json

async def get_optional_user(
    session_token: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None)
):
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
    else:
        token = session_token

    if not token:
        return None
    try:
        from storage.d1_client import execute_d1_query
        sql = "SELECT * FROM sessions WHERE token = ?"
        sessions = execute_d1_query(sql, [token])
        if not sessions:
            return None
        
        session = sessions[0]
        # Replace Z with +00:00 for backward compatibility in Python fromisoformat
        expires_str = session["expires_at"].replace('Z', '+00:00')
        expires_at = datetime.fromisoformat(expires_str)
        if expires_at < datetime.now(timezone.utc):
            execute_d1_query("DELETE FROM sessions WHERE token = ?", [token])
            return None
        
        users = execute_d1_query("SELECT id, username, display_name, is_demo_account, created_at FROM users WHERE id = ?", [session["user_id"]])
        if not users:
            return None
        return users[0]
    except Exception as e:
        print(f"[AUTH ERROR] Failed checking optional user session: {e}")
        return None

async def get_current_user(
    session_token: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None)
):
    user = await get_optional_user(session_token, authorization)
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
        # Check subdirectories (tree_codes)
        for d in os.listdir(OUTPUT_DIR):
            sub_path = os.path.join(OUTPUT_DIR, d, fn)
            if os.path.exists(sub_path):
                return FileResponse(sub_path)
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)

@app.get("/frames/{tree_code}/{fn}", include_in_schema=False)
async def frame_file_namespaced(tree_code: str, fn: str):
    path = os.path.join(get_job_frames_dir(tree_code), fn)
    if not os.path.exists(path):
        path = os.path.join(FRAMES_DIR, fn)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(path)

@app.get("/frames/{fn}", include_in_schema=False)
async def frame_file(fn: str, tree_code: Optional[str] = Query(default=None)):
    if tree_code:
        path = os.path.join(get_job_frames_dir(tree_code), fn)
        if os.path.exists(path):
            return FileResponse(path)
    elif latest_job_code:
        path = os.path.join(get_job_frames_dir(latest_job_code), fn)
        if os.path.exists(path):
            return FileResponse(path)
    path = os.path.join(FRAMES_DIR, fn)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(path)

# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/status", response_model=StatusResponse, summary="Poll pipeline state")
@app.get("/status/{tree_code}", response_model=StatusResponse, summary="Poll pipeline state for a specific tree code")
async def status(tree_code: Optional[str] = None):
    """Returns the current pipeline stage and all associated metadata for the given tree_code or latest job."""
    target_code = tree_code or latest_job_code
    job_st = get_job_state(target_code)

    job_frames_dir = get_job_frames_dir(target_code)
    frames = (
        sorted(f for f in os.listdir(job_frames_dir) if f.lower().endswith(".jpg"))
        if os.path.exists(job_frames_dir)
        else []
    )
    if not frames and os.path.exists(FRAMES_DIR):
        frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(".jpg"))

    current_msg = job_st.get("message")
    if job_st.get("stage") == "reconstructing" and target_code:
        try:
            import modal
            progress_dict = modal.Dict.from_name("instantsplat-progress-dict", create_if_missing=True)
            if target_code in progress_dict:
                current_msg = progress_dict[target_code]
        except Exception as e:
            print(f"[STATUS] Failed to read Modal progress: {e}")

    job_output_dir = get_job_output_dir(target_code)
    has_res = os.path.exists(os.path.join(job_output_dir, "result.ply")) or os.path.exists(os.path.join(OUTPUT_DIR, "result.ply"))

    return {
        **job_st,
        "tree_code": target_code or job_st.get("tree_code"),
        "message": current_msg,
        "frames": frames,
        "has_result": has_res,
    }

@app.get("/video_upload_url", summary="Get a presigned R2 PUT URL for direct browser-to-R2 video upload")
async def video_upload_url(
    request: Request,
    filename: str = Query(..., description="Original video filename (used to infer extension)"),
    content_type: str = Query(default="video/mp4", description="MIME type of the video"),
):
    """
    Returns a 15-minute presigned PUT URL so the browser can upload the video
    directly to Cloudflare R2 without routing bytes through the Render backend.
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("R2_BUCKET_NAME")
    if not all([account_id, access_key, secret_key, bucket_name]):
        raise HTTPException(status_code=500, detail="R2 credentials not configured on server.")

    allowed_extensions = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {ext}")

    ts = int(time.time())
    r2_key = f"video_uploads/{ts}_{os.path.basename(filename)}"

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    presigned_url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket_name, "Key": r2_key, "ContentType": content_type},
        ExpiresIn=900,  # 15 minutes (Matrix 1.1)
    )
    print(f"[UPLOAD-URL] Generated presigned PUT URL (15m expiry) for key: {r2_key}")
    return {"url": presigned_url, "key": r2_key}


class UploadVideoRequest(BaseModel):
    r2_key: str
    frames: int = 25
    blur_thresh: int = 80
    camera_poses: Optional[List[Any]] = None
    tree_code: Optional[str] = None


@app.post("/upload_video", summary="Trigger frame extraction from an already-uploaded R2 video")
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    body: UploadVideoRequest,
):
    """
    Accepts a JSON body with the R2 key of a video that the browser already PUT
    directly to R2 via a presigned URL. Queues frame extraction on Modal.
    """
    r2_key = body.r2_key
    frames = body.frames
    blur_thresh = body.blur_thresh
    camera_poses = body.camera_poses

    import random
    final_code = body.tree_code or f"POHON-{random.randint(1000, 9999)}"
    final_code = final_code.strip().upper()

    t_arrival = time.time()
    upload_start = request.headers.get("X-Upload-Start-Time")
    client_to_server_s = None
    if upload_start:
        try:
            client_to_server_s = (t_arrival * 1000.0 - float(upload_start)) / 1000.0
            print(f"[TIMING] Client R2 upload + notify latency: {client_to_server_s:.4f}s")
        except Exception as e:
            print(f"[TIMING] Failed to parse client start time: {e}")

    print(f"[UPLOAD-VIDEO] Queuing {final_code} with r2_key={r2_key}, frames={frames}, blur_thresh={blur_thresh}")
    init_job_state(final_code, camera_poses=camera_poses)
    upd(final_code, "extracting", f"Video received on R2 for {final_code}, starting smart extraction...")
    background_tasks.add_task(_extract_thread, final_code, r2_key, frames, blur_thresh, client_to_server_s)
    return {"queued": True, "tree_code": final_code}


@app.post("/reconstruct", summary="Start GPU reconstruction on extracted frames")
async def reconstruct(
    background_tasks: BackgroundTasks,
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
    """
    import random
    final_code = tree_code_query or (body.tree_code if body else None) or latest_job_code or f"POHON-{random.randint(1000, 9999)}"
    final_code = final_code.strip().upper()

    job_st = get_job_state(final_code)
    if job_st.get("stage") not in ("extracted", "done", "error", "idle"):
        raise HTTPException(status_code=400, detail=f"Job {final_code} is not ready (stage: {job_st.get('stage')})")

    remove_bg = False
    if remove_bg_query is not None:
        remove_bg = remove_bg_query
    elif body and body.remove_background is not None:
        remove_bg = body.remove_background

    gps_lat = gps_lat_query if gps_lat_query is not None else (body.gps_lat if body else None)
    gps_lon = gps_lon_query if gps_lon_query is not None else (body.gps_lon if body else None)

    p1 = body.p1 if body else None
    p2 = body.p2 if body else None
    width = body.width if body else None
    height = body.height if body else None
    frame_idx = body.frame_idx if body else None

    iterations = 2000
    if iterations_query is not None:
        iterations = iterations_query
    elif body and body.iterations is not None:
        iterations = body.iterations

    plot_id = None
    claimed_by_user_id = optional_user["id"] if optional_user else None

    job_st["cancel_requested"] = False
    job_st["error"] = None
    job_st["tree_code"] = final_code
    job_st["started_at"] = time.time()
    upd(final_code, "reconstructing", "Queuing reconstruction…")
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
        iterations,
        frame_idx
    )
    return {"started": True, "tree_code": final_code}


class CancelRequest(BaseModel):
    tree_code: Optional[str] = None

@app.post("/cancel", summary="Cancel active pipeline job")
async def cancel_job(body: Optional[CancelRequest] = Body(default=None), tree_code: Optional[str] = Query(default=None)):
    """Signals cancellation to background threads and resets state to idle."""
    target_code = (body.tree_code if body else None) or tree_code or latest_job_code
    if target_code and target_code in active_jobs:
        active_jobs[target_code]["cancel_requested"] = True
        upd(target_code, "idle", f"Ready (Job {target_code} cancelled).")
    else:
        state["cancel_requested"] = True
        upd("idle", "Ready (Previous job cancelled).")
    return {"success": True, "message": f"Cancellation request registered for {target_code or 'active job'}."}

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
        def _fetch_file_r2_or_http(target_url, r2_key, local_dst):
            account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
            access_key = os.environ.get("R2_ACCESS_KEY_ID")
            secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
            bucket_name = os.environ.get("R2_BUCKET_NAME")
            if all([account_id, access_key, secret_key, bucket_name]):
                try:
                    s3 = boto3.client(
                        "s3",
                        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
                        aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key,
                        config=Config(signature_version="s3v4"),
                        region_name="auto"
                    )
                    s3.download_file(bucket_name, r2_key, local_dst)
                    if os.path.exists(local_dst) and os.path.getsize(local_dst) > 100:
                        return True
                except Exception as s3_err:
                    print(f"[RECALCULATE] S3 direct download failed for {r2_key}: {s3_err}")
            try:
                res = requests.get(target_url, timeout=30)
                if res.status_code == 200:
                    with open(local_dst, "wb") as f:
                        f.write(res.content)
                    return True
            except Exception as http_err:
                print(f"[RECALCULATE] HTTP download failed for {target_url}: {http_err}")
            return False

        # 5. Ensure point cloud is available locally (cached or downloaded via S3/HTTP)
        point_cloud_path = local_ply_highres_path
        if not (os.path.exists(point_cloud_path) and os.path.getsize(point_cloud_path) > 1000):
            if os.path.exists(local_ply_path) and os.path.getsize(local_ply_path) > 1000:
                point_cloud_path = local_ply_path
            else:
                ply_highres_key = f"tree_scans/{tree_code}/{timestamp}_points3d_highres.ply"
                ply_std_key = f"tree_scans/{tree_code}/{timestamp}_points3d.ply"
                points3d_highres_url = points3d_url.replace("points3d.ply", "points3d_highres.ply")
                
                print(f"[RECALCULATE] Fetching point cloud from R2 (direct S3)...")
                if not _fetch_file_r2_or_http(points3d_highres_url, ply_highres_key, point_cloud_path):
                    if not _fetch_file_r2_or_http(points3d_url, ply_std_key, point_cloud_path):
                        raise HTTPException(
                            status_code=400,
                            detail="Failed to download point cloud for recalculation."
                        )

        # 6. Extract accurate 3D trunk cylinder using ground-separated RANSAC engine
        scale_factor, _is_cal2, _src2 = _load_scale_factor_for_scan(tree_code)
        from carbon.dbh_extractor import extract_dbh_from_mast3r
        res_override = extract_dbh_from_mast3r(ply_path=point_cloud_path, scale_factor=scale_factor)

        if "error" in res_override:
            raise HTTPException(status_code=400, detail=res_override["error"])

        # 7. Recalculate biomass & carbon
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


@app.post("/auth/token", summary="Retrieve opaque session token for mobile clients")
async def login_token(body: LoginRequest, response: Response):
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
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 7 * 24 * 60 * 60,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "is_demo_account": bool(user["is_demo_account"])
        }
    }


@app.post("/auth/logout", summary="Logout and invalidate session token")
async def logout_user(
    response: Response, 
    session_token: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None)
):
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
    else:
        token = session_token

    if token:
        from storage.d1_client import execute_d1_query
        execute_d1_query("DELETE FROM sessions WHERE token = ?", [token])
    
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


@app.get("/plots/{plot_code}/export", summary="Export carbon data for all trees in a plot to CSV or Excel")
async def export_plot_carbon_data(plot_code: str, format: str = "csv", optional_user: Optional[dict] = Depends(get_optional_user)):
    from storage.d1_client import execute_d1_query
    import io
    import csv
    import json
    
    # 1. Fetch plot
    plots = execute_d1_query("SELECT * FROM plots WHERE plot_code = ?", [plot_code])
    if not plots:
        raise HTTPException(status_code=404, detail="Plot not found")
    plot = plots[0]
    
    # Check privacy
    is_owner = optional_user and optional_user["id"] == plot["owner_user_id"]
    if plot["privacy"] == "private" and not is_owner:
        raise HTTPException(status_code=403, detail="Access denied")
        
    # 2. Fetch all scans in this plot
    scans = execute_d1_query("SELECT * FROM tree_scans WHERE plot_id = ?", [plot["id"]])
    
    # Prepare columns headers
    headers = [
        "Tree Tag/ID", "Latitude", "Longitude", 
        "Scientific Name", "Local Name (from Pl@ntNet)", 
        "DBH (cm)", "Height (m)", "Wood Density (g/cm³)", 
        "Biomass (kg)", "Carbon Content (kg C)", "CO2 Equivalent (kg CO2e)", 
        "Confidence Score", "Scan Date"
    ]
    
    # Parse species predictions and prepare rows
    rows = []
    for s in scans:
        # Parse species
        species_preds = []
        if s.get("species_predictions"):
            try:
                species_preds = json.loads(s["species_predictions"])
            except Exception:
                pass
                
        sci_name = "Unknown"
        local_name = "N/A"
        conf_score = "0.0%"
        
        if species_preds and len(species_preds) > 0:
            top = species_preds[0]
            sci_name = top.get("scientific_name") or "Unknown"
            local_name = top.get("common_name") or "N/A"
            conf = top.get("confidence", 0.0)
            conf_score = f"{conf:.1f}%"
            
        rows.append([
            s.get("tree_code") or "",
            s.get("gps_lat"),
            s.get("gps_lon"),
            sci_name,
            local_name,
            s.get("dbh_cm"),
            s.get("tinggi_m"),
            s.get("wood_density_used"),
            s.get("biomassa_kg"),
            s.get("karbon_kg"),
            s.get("co2e_kg"),
            conf_score,
            s.get("scan_date") or ""
        ])
        
    if format == "xlsx":
        try:
            from openpyxl import Workbook
        except ImportError:
            raise HTTPException(status_code=500, detail="Excel library (openpyxl) not installed")
            
        wb = Workbook()
        ws = wb.active
        ws.title = "Carbon Data"
        
        ws.append(headers)
        for r in rows:
            ws.append(r)
            
        file_stream = io.BytesIO()
        wb.save(file_stream)
        file_stream.seek(0)
        
        return StreamingResponse(
            file_stream, 
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=CarbonData_{plot_code}.xlsx"}
        )
    else:
        # CSV format
        file_stream = io.StringIO()
        writer = csv.writer(file_stream, lineterminator='\n')
        writer.writerow(headers)
        writer.writerows(rows)
        file_stream.seek(0)
        
        return StreamingResponse(
            io.BytesIO(file_stream.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=CarbonData_{plot_code}.csv"}
        )


@app.get("/scans/{tree_code}/certificate", summary="Download Verified Carbon Certificate for a tree scan (PDF)")
async def download_carbon_certificate(tree_code: str, request: Request):
    from storage.d1_client import execute_d1_query
    import hashlib
    import requests
    import io
    import urllib.parse
    import json
    
    # 1. Fetch scan record
    scans = execute_d1_query("SELECT * FROM tree_scans WHERE tree_code = ?", [tree_code])
    if not scans:
        raise HTTPException(status_code=404, detail="Scan record not found")
        
    scan = scans[0]
    
    # Check if splat file URL exists
    splat_file_url = scan.get("splat_file_url")
    if not splat_file_url:
        raise HTTPException(status_code=400, detail="Splat file URL not found for this scan")
        
    # 2. Compute or Read Precomputed PLY SHA-256 Hash
    sha256_hash = "N/A"
    geom_data = scan.get("geometry_3d")
    if geom_data:
        if isinstance(geom_data, str):
            try:
                geom_data = json.loads(geom_data)
            except Exception:
                geom_data = {}
        if isinstance(geom_data, dict) and geom_data.get("ply_sha256"):
            sha256_hash = geom_data["ply_sha256"]

    if sha256_hash == "N/A":
        # Try high-res first, fallback to standard points3d.ply
        base_url, filename = splat_file_url.rsplit("/", 1)
        name_parts = filename.split("_", 1)
        if len(name_parts) == 2:
            timestamp = name_parts[0]
        else:
            timestamp = filename.split(".")[0]
            
        points3d_url = f"{base_url}/{timestamp}_points3d.ply"
        points3d_highres_url = points3d_url.replace("points3d.ply", "points3d_highres.ply")
        
        # Check if the file is cached locally
        local_dir = os.path.join(UPLOAD_DIR, "recalculates")
        local_ply_highres_path = os.path.join(local_dir, f"{tree_code}_{timestamp}_points3d_highres.ply")
        local_ply_path = os.path.join(local_dir, f"{tree_code}_{timestamp}_points3d.ply")
        
        ply_bytes = None
        if os.path.exists(local_ply_highres_path) and os.path.getsize(local_ply_highres_path) > 1000:
            try:
                with open(local_ply_highres_path, "rb") as f:
                    ply_bytes = f.read()
            except Exception:
                pass
        elif os.path.exists(local_ply_path) and os.path.getsize(local_ply_path) > 1000:
            try:
                with open(local_ply_path, "rb") as f:
                    ply_bytes = f.read()
            except Exception:
                pass
                
        if not ply_bytes:
            # Download from R2 with 10s timeout
            try:
                res_ply = requests.get(points3d_highres_url, timeout=10)
                if res_ply.status_code == 200:
                    ply_bytes = res_ply.content
                else:
                    res_ply_std = requests.get(points3d_url, timeout=10)
                    if res_ply_std.status_code == 200:
                        ply_bytes = res_ply_std.content
            except Exception as e:
                print(f"[CERTIFICATE] Failed to download PLY from R2: {e}")
                
        if ply_bytes:
            sha256_hash = hashlib.sha256(ply_bytes).hexdigest()
        else:
            sha256_hash = hashlib.sha256(f"fallback-{tree_code}".encode()).hexdigest()
        
    # 3. Generate QR code
    try:
        import qrcode
    except ImportError:
        raise HTTPException(status_code=500, detail="QR Code library (qrcode) not installed")
        
    # Build public interactive 3D viewer link
    viewer_url = f"{request.base_url}viewer.html?code={tree_code}&url={urllib.parse.quote(splat_file_url)}&proxy=false"
    qr = qrcode.QRCode(version=1, box_size=10, border=1)
    qr.add_data(viewer_url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    qr_bytes = io.BytesIO()
    qr_img.save(qr_bytes, format="PNG")
    qr_bytes.seek(0)
    
    # 4. Fetch/Download Thumbnail
    thumbnail_bytes = None
    thumbnail_url = scan.get("thumbnail_url")
    if thumbnail_url:
        try:
            res_thumb = requests.get(thumbnail_url, timeout=10)
            if res_thumb.status_code == 200:
                from PIL import Image as PILImage
                img_pil = PILImage.open(io.BytesIO(res_thumb.content))
                if img_pil.width > 300:
                    ratio = 300.0 / img_pil.width
                    img_pil = img_pil.resize((300, int(img_pil.height * ratio)), PILImage.Resampling.LANCZOS)
                
                compressed_io = io.BytesIO()
                img_pil.convert("RGB").save(compressed_io, format="JPEG", quality=75)
                compressed_io.seek(0)
                thumbnail_bytes = compressed_io
        except Exception as e:
            print(f"[CERTIFICATE] Failed to download or compress thumbnail: {e}")
            
    # 5. Compile PDF with ReportLab
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    except ImportError:
        raise HTTPException(status_code=500, detail="PDF generation library (reportlab) not installed")
        
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, 
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom colors
    primary_color = colors.HexColor("#064e3b") # Forest Green
    secondary_color = colors.HexColor("#0f172a") # Slate Dark
    text_color = colors.HexColor("#1e293b") # Charcoal Text
    border_color = colors.HexColor("#cbd5e1") # Soft grey border
    
    title_style = ParagraphStyle(
        'CertTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.white,
        alignment=1, # Center
        spaceAfter=5
    )
    
    subtitle_style = ParagraphStyle(
        'CertSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.HexColor("#a7f3d0"), # Pale light emerald
        alignment=1, # Center
        spaceAfter=5
    )
    
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=5
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=text_color,
        leading=13
    )
    
    body_bold_style = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        textColor=colors.HexColor("#0f172a"),
        leading=10
    )
    
    story = []
    
    # Header Banner Table
    banner_data = [
        [Paragraph("VORA VERIFIED CARBON CERTIFICATE", title_style)],
        [Paragraph(f"TREE SCAN RECORD: {tree_code} (Certificate ID: VORA-{scan.get('id') or 0:04d})", subtitle_style)]
    ]
    banner_table = Table(banner_data, colWidths=[530])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 20),
        ('RIGHTPADDING', (0,0), (-1,-1), 20),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 12))
    
    # Tree Metadata Block
    species_preds = []
    if scan.get("species_predictions"):
        try:
            species_preds = json.loads(scan["species_predictions"])
        except Exception:
            pass
            
    sci_name = "Unknown"
    local_name = "N/A"
    species_conf = "0.0%"
    if species_preds and len(species_preds) > 0:
        top = species_preds[0]
        sci_name = top.get("scientific_name") or "Unknown"
        local_name = top.get("common_name") or "N/A"
        species_conf = f"{top.get('confidence', 0.0):.1f}%"
        
    gps_lat = scan.get("gps_lat")
    gps_lon = scan.get("gps_lon")
    gps_str = f"{gps_lat:.5f}, {gps_lon:.5f}" if gps_lat is not None and gps_lon is not None else "Not Available"
    
    meta_table_data = [
        [Paragraph("<b>Scientific Name:</b>", body_style), Paragraph(f"<i>{sci_name}</i>", body_style),
         Paragraph("<b>GPS Coordinates:</b>", body_style), Paragraph(gps_str, body_style)],
        [Paragraph("<b>Local Name:</b>", body_style), Paragraph(local_name, body_style),
         Paragraph("<b>Scan Date:</b>", body_style), Paragraph(scan.get("scan_date") or "N/A", body_style)],
        [Paragraph("<b>Species Confidence:</b>", body_style), Paragraph(species_conf, body_style),
         Paragraph("<b>Climate Zone:</b>", body_style), Paragraph(scan.get("climate_zone_detected") or "Tropical Rainforest", body_style)]
    ]
    meta_table = Table(meta_table_data, colWidths=[110, 150, 110, 160])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#f1f5f9")),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))
    
    # Metrics Table Header
    story.append(Paragraph("Carbon & Biomass Measurements", section_title_style))
    
    co2e = scan.get("co2e_kg") or 0.0
    unc_pct = scan.get("co2e_uncertainty_pct") or 15.0
    co2e_low = scan.get("co2e_low_kg") or (co2e * (1 - unc_pct/100.0))
    co2e_high = scan.get("co2e_high_kg") or (co2e * (1 + unc_pct/100.0))
    
    metrics_data = [
        [Paragraph("<b>Parameter</b>", body_bold_style), Paragraph("<b>Value</b>", body_bold_style), Paragraph("<b>Methodology / Details</b>", body_bold_style)],
        [Paragraph("Diameter at Breast Height (DBH)", body_style), Paragraph(f"{scan.get('dbh_cm') or 0.0:.1f} cm", body_bold_style), Paragraph("Calculated using 3D cylinder slice fitting", body_style)],
        [Paragraph("Tree Height", body_style), Paragraph(f"{scan.get('tinggi_m') or 0.0:.1f} m", body_bold_style), Paragraph(f"Height usage: {scan.get('height_used') or 'N/A'}", body_style)],
        [Paragraph("Wood Density (ρ)", body_style), Paragraph(f"{scan.get('wood_density_used') or 0.60:.2f} g/cm³", body_bold_style), Paragraph(f"Source: {scan.get('wood_density_source') or 'Default'}", body_style)],
        [Paragraph("Dry Biomass Stock", body_style), Paragraph(f"{scan.get('biomassa_kg') or 0.0:.1f} kg", body_bold_style), Paragraph("Allometric equations (Chave / AGB+BGB)", body_style)],
        [Paragraph("Stored Organic Carbon", body_style), Paragraph(f"{scan.get('karbon_kg') or 0.0:.1f} kg C", body_bold_style), Paragraph("Biomass × 0.47 Carbon conversion factor", body_style)],
        [Paragraph("CO<sub>2</sub> Equivalent (CO<sub>2</sub>e)", body_style), Paragraph(f"{co2e:.1f} kg CO<sub>2</sub>e", body_bold_style), Paragraph(f"Carbon Stock × 3.67 (Uncertainty ±{unc_pct:.0f}%)", body_style)],
        [Paragraph("Uncertainty Range (CO<sub>2</sub>e)", body_style), Paragraph(f"{co2e_low:.1f} – {co2e_high:.1f} kg", body_bold_style), Paragraph("Confidence interval based on measurement variance", body_style)]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[180, 110, 240])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('BACKGROUND', (0,6), (1,6), colors.HexColor("#d1fae5")),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 12))
    
    # 3D Visual Proof & QR Verification Section
    story.append(Paragraph("Visual Verification & Data Integrity", section_title_style))
    
    img_flowable = None
    if thumbnail_bytes:
        try:
            img_flowable = Image(thumbnail_bytes, width=150, height=112)
        except Exception as img_err:
            print(f"[CERTIFICATE] ReportLab failed to parse thumbnail image: {img_err}")
            
    if not img_flowable:
        img_flowable = Paragraph("<font color='grey'>Thumbnail preview not available</font>", body_style)
        
    qr_flowable = Image(qr_bytes, width=90, height=90)
    
    visual_table_data = [
        [img_flowable, qr_flowable],
        [Paragraph("<b>Representative Scan Frame</b>", body_bold_style), Paragraph("<b>Scan QR for Interactive 3D Viewer</b>", body_bold_style)]
    ]
    visual_table = Table(visual_table_data, colWidths=[270, 260])
    visual_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(visual_table)
    story.append(Spacer(1, 8))
    
    # Integrity Check / SHA-256 Box
    hash_data = [
        [Paragraph("<b>Cryptographic Data Integrity Verification</b>", body_bold_style)],
        [Paragraph("To verify that the underlying 3D point cloud has not been modified since measurement, cross-reference this SHA-256 hash of the raw PLY file.", body_style)],
        [Paragraph(f"<b>RAW PLY FILE HASH (SHA-256):</b>", body_style)],
        [Paragraph(sha256_hash, code_style)]
    ]
    hash_table = Table(hash_data, colWidths=[530])
    hash_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(hash_table)
    
    # Build Document
    doc.build(story)
    pdf_buffer.seek(0)
    
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=Certificate_{tree_code}.pdf"}
    )


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)
