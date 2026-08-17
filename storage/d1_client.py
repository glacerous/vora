import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone
import secrets
import string
import time
import json
import logging

_logger = logging.getLogger("D1Client")

# Persistent connection pool for Cloudflare D1
_d1_session = None

def get_d1_session() -> requests.Session:
    global _d1_session
    if _d1_session is None:
        s = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=25,
            pool_maxsize=25,
            max_retries=Retry(total=2, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _d1_session = s
    return _d1_session

def get_d1_headers():
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        raise ValueError("Missing CLOUDFLARE_API_TOKEN in environment variables")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

def execute_d1_query(sql: str, params: list = None, max_retries: int = 3, retry_delay: float = 0.5, return_meta: bool = False):
    """
    Sends an HTTP POST query request to the Cloudflare D1 HTTP API with persistent session pooling and exponential backoff retry.
    If return_meta=True, returns (results, meta_dict) tuple. Otherwise returns results list.
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    db_id = os.environ.get("CLOUDFLARE_D1_DATABASE_ID")
    
    if not account_id or not db_id:
        raise ValueError("Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_D1_DATABASE_ID in environment variables")
        
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{db_id}/query"
    payload = {
        "sql": sql,
        "params": params or []
    }
    
    headers = get_d1_headers()
    session = get_d1_session()
    last_exc = None

    for attempt in range(max_retries):
        try:
            # 10s connect timeout, 35s read timeout with connection reuse
            response = session.post(url, json=payload, headers=headers, timeout=(10, 35))
            
            if response.status_code != 200:
                raise RuntimeError(f"Cloudflare D1 API request failed with status {response.status_code}: {response.text}")
                
            resp_data = response.json()
            if not resp_data.get("success"):
                errors = resp_data.get("errors", [])
                error_msg = "; ".join([e.get("message", "") for e in errors]) or "Unknown API error"
                raise RuntimeError(f"Cloudflare D1 Query execution failed: {error_msg}")
                
            result_list = resp_data.get("result", [])
            if not result_list:
                raise RuntimeError("Cloudflare D1 Query returned empty results wrapper")
                
            query_result = result_list[0]
            if not query_result.get("success"):
                raise RuntimeError("SQL execution failed inside D1")
                
            results = query_result.get("results", [])
            meta = query_result.get("meta", {})
            if return_meta:
                return results, meta
            return results
        except Exception as exc:
            last_exc = exc
            _logger.warning(f"[D1-RETRY] Query attempt {attempt + 1}/{max_retries} failed: {exc}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))

    raise last_exc

import json

def save_scan_result(tree_code: str, dbh_cm: float, tinggi_m: float, biomassa_kg: float,
                     karbon_kg: float, co2e_kg: float, splat_file_url: str, confidence_note: str,
                     thumbnail_url: str = None, geometry_3d: dict = None, species_predictions: list = None,
                     wood_density_used: float = None, wood_density_source: str = None,
                     climate_zone_detected: str = None, formula_used: str = None,
                     agb_kg: float = None, bgb_kg: float = None,
                     gps_lat: float = None, gps_lon: float = None,
                     scale_status: str = None, scale_factor_used: float = None,
                     calibration_source: str = None, height_used: str = None,
                     total_height_used_m: float = None, segment_height_m: float = None,
                     height_fallback_reason: str = None, quality_status: str = None,
                     root_to_shoot_ratio: float = None, co2e_uncertainty_pct: float = None,
                     co2e_low_kg: float = None, co2e_high_kg: float = None,
                     plot_id: int = None, claimed_by_user_id: int = None,
                     grid_position_x: int = None, grid_position_y: int = None,
                     inlier_ratio: float = None,
                     species_detection_status: str = None,
                     species_detection_frame_used: str = None,
                     dbh_equivalent_cm: float = None):
    """
    Inserts a new scan record into the tree_scans database table on Cloudflare D1.
    """
    sql = """
    INSERT INTO tree_scans (
        tree_code, scan_date, dbh_cm, tinggi_m, biomassa_kg, karbon_kg, co2e_kg, 
        splat_file_url, confidence_note, thumbnail_url, geometry_3d, species_predictions,
        wood_density_used, wood_density_source, climate_zone_detected, formula_used,
        agb_kg, bgb_kg, gps_lat, gps_lon,
        scale_status, scale_factor_used, calibration_source, height_used,
        total_height_used_m, segment_height_m, height_fallback_reason, quality_status,
        root_to_shoot_ratio, co2e_uncertainty_pct, co2e_low_kg, co2e_high_kg,
        plot_id, claimed_by_user_id, grid_position_x, grid_position_y, inlier_ratio,
        species_detection_status, species_detection_frame_used, dbh_equivalent_cm
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    # Use ISO 8601 UTC format for scan_date
    scan_date = datetime.now(timezone.utc).isoformat()
    geom_str = json.dumps(geometry_3d) if geometry_3d else None
    species_str = json.dumps(species_predictions) if species_predictions else None
    params = [
        tree_code, 
        scan_date, 
        dbh_cm, 
        tinggi_m, 
        biomassa_kg, 
        karbon_kg, 
        co2e_kg, 
        splat_file_url, 
        confidence_note,
        thumbnail_url,
        geom_str,
        species_str,
        wood_density_used,
        wood_density_source,
        climate_zone_detected,
        formula_used,
        agb_kg,
        bgb_kg,
        gps_lat,
        gps_lon,
        scale_status,
        scale_factor_used,
        calibration_source,
        height_used,
        total_height_used_m,
        segment_height_m,
        height_fallback_reason,
        quality_status,
        root_to_shoot_ratio,
        co2e_uncertainty_pct,
        co2e_low_kg,
        co2e_high_kg,
        plot_id,
        claimed_by_user_id,
        grid_position_x,
        grid_position_y,
        inlier_ratio,
        species_detection_status,
        species_detection_frame_used,
        dbh_equivalent_cm
    ]
    try:
        execute_d1_query(sql, params)
    except Exception as exc:
        _logger.error(f"[D1-SAVE-FAIL] Failed to save scan to D1 after retries: {exc}. Writing to local disk buffer.")
        try:
            backup_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads", "failed_scans_buffer")
            os.makedirs(backup_dir, exist_ok=True)
            backup_file = os.path.join(backup_dir, f"{tree_code}_{int(time.time())}.json")
            with open(backup_file, "w") as bf:
                json.dump({
                    "tree_code": tree_code,
                    "scan_date": scan_date,
                    "params": [p if not isinstance(p, (bytes, bytearray)) else "<bytes>" for p in params],
                    "error": str(exc),
                }, bf, indent=2)
        except Exception as buf_err:
            _logger.error(f"[D1-BUFFER-ERROR] Could not write fallback buffer: {buf_err}")
        raise exc

def update_scan_result(scan_id: int, dbh_cm: float, tinggi_m: float, biomassa_kg: float,
                       karbon_kg: float, co2e_kg: float, confidence_note: str,
                       geometry_3d: dict = None, wood_density_used: float = None,
                       wood_density_source: str = None, climate_zone_detected: str = None,
                       formula_used: str = None, agb_kg: float = None, bgb_kg: float = None,
                       gps_lat: float = None, gps_lon: float = None,
                       species_predictions: list = None,
                       scale_status: str = None, scale_factor_used: float = None,
                       calibration_source: str = None, height_used: str = None,
                       total_height_used_m: float = None, segment_height_m: float = None,
                       height_fallback_reason: str = None, height_validated: bool = None,
                       height_validation_reason: str = None, quality_status: str = None,
                       root_to_shoot_ratio: float = None, co2e_uncertainty_pct: float = None,
                       co2e_low_kg: float = None, co2e_high_kg: float = None,
                       inlier_ratio: float = None,
                       species_detection_status: str = None,
                       species_detection_frame_used: str = None,
                     dbh_equivalent_cm: float = None):
    """
    Updates an existing scan record in the tree_scans database table on Cloudflare D1 by scan_id.
    All accuracy-metadata fields are optional and default to None (no change / preserve column).
    """
    sql = """
    UPDATE tree_scans 
    SET dbh_cm = ?, tinggi_m = ?, biomassa_kg = ?, karbon_kg = ?, co2e_kg = ?, 
        confidence_note = ?, geometry_3d = ?, wood_density_used = ?, 
        wood_density_source = ?, climate_zone_detected = ?, formula_used = ?,
        agb_kg = ?, bgb_kg = ?, gps_lat = ?, gps_lon = ?, species_predictions = ?,
        scale_status = ?, scale_factor_used = ?, calibration_source = ?,
        height_used = ?, total_height_used_m = ?, segment_height_m = ?,
        height_fallback_reason = ?, height_validated = ?, height_validation_reason = ?,
        quality_status = ?, root_to_shoot_ratio = ?, co2e_uncertainty_pct = ?,
        co2e_low_kg = ?, co2e_high_kg = ?, inlier_ratio = ?,
        species_detection_status = COALESCE(?, species_detection_status),
        species_detection_frame_used = COALESCE(?, species_detection_frame_used),
        dbh_equivalent_cm = COALESCE(?, dbh_equivalent_cm)
    WHERE id = ?
    """
    geom_str = json.dumps(geometry_3d) if geometry_3d else None
    species_str = json.dumps(species_predictions) if species_predictions else None
    hv = (1 if height_validated else 0) if height_validated is not None else None
    execute_d1_query(sql, [
        dbh_cm, 
        tinggi_m, 
        biomassa_kg, 
        karbon_kg, 
        co2e_kg, 
        confidence_note,
        geom_str,
        wood_density_used,
        wood_density_source,
        climate_zone_detected,
        formula_used,
        agb_kg,
        bgb_kg,
        gps_lat,
        gps_lon,
        species_str,
        scale_status,
        scale_factor_used,
        calibration_source,
        height_used,
        total_height_used_m,
        segment_height_m,
        height_fallback_reason,
        hv,
        height_validation_reason,
        quality_status,
        root_to_shoot_ratio,
        co2e_uncertainty_pct,
        co2e_low_kg,
        co2e_high_kg,
        inlier_ratio,
        species_detection_status,
        species_detection_frame_used,
        dbh_equivalent_cm,
        scan_id
    ])

def populate_scan_defaults(r: dict):
    if r.get("dbh_cm") is not None:
        if r.get("wood_density_used") is None:
            r["wood_density_used"] = 0.6
        if r.get("wood_density_source") is None:
            r["wood_density_source"] = "generic-default"
        if r.get("climate_zone_detected") is None:
            r["climate_zone_detected"] = "Unknown"
        # Scale calibration status — conservative default for legacy rows
        if r.get("scale_status") is None:
            r["scale_status"] = "uncalibrated"
        if r.get("quality_status") is None:
            r["quality_status"] = "ok"
        # Root-to-shoot ratio — derive from forest type fallback to moist (0.37)
        if r.get("root_to_shoot_ratio") is None:
            r["root_to_shoot_ratio"] = 0.37
        if r.get("formula_used") is None:
            if r.get("height_used") == "full_height":
                r["formula_used"] = "Chave 2005 (height-based)"
            else:
                h = r.get("tinggi_m")
                if h is not None and h > 0:
                    r["formula_used"] = "Chave 2005 (moist forest with height)"
                else:
                    r["formula_used"] = "Chave 2005 (moist forest, DBH-only)"
        if r.get("agb_kg") is None:
            total_biomass = r.get("biomassa_kg") or 0.0
            rs = r.get("root_to_shoot_ratio") or 0.37
            r["agb_kg"] = float(round(total_biomass / (1.0 + rs), 2))
        if r.get("bgb_kg") is None:
            total_biomass = r.get("biomassa_kg") or 0.0
            rs = r.get("root_to_shoot_ratio") or 0.37
            r["bgb_kg"] = float(round(total_biomass * rs / (1.0 + rs), 2))

def get_scan_history(tree_code: str):
    """
    Retrieves all scan history for a specific tree_code sorted by scan_date.
    """
    sql = "SELECT * FROM tree_scans WHERE tree_code = ? ORDER BY scan_date DESC"
    rows = execute_d1_query(sql, [tree_code])
    for r in rows:
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
    return rows

def get_all_scans(limit: int = 20, offset: int = 0, include_invalid: bool = False):
    """
    Retrieves all scan records from the database sorted by scan_date descending with limit & offset.
    """
    if include_invalid:
        sql = "SELECT * FROM tree_scans ORDER BY scan_date DESC LIMIT ? OFFSET ?"
    else:
        sql = "SELECT * FROM tree_scans WHERE dbh_cm IS NOT NULL ORDER BY scan_date DESC LIMIT ? OFFSET ?"
    rows = execute_d1_query(sql, [int(limit), int(offset)])
    for r in rows:
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
    return rows

def generate_tree_code() -> str:
    """
    Generates a unique tree code of format POHON-XXXX where XXXX is random alphanumeric.
    """
    chars = string.ascii_uppercase + string.digits
    code = "".join(secrets.choice(chars) for _ in range(4))
    return f"POHON-{code}"
