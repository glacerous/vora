import os
import requests
from datetime import datetime, timezone
import secrets
import string

def get_d1_headers():
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        raise ValueError("Missing CLOUDFLARE_API_TOKEN in environment variables")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

def execute_d1_query(sql: str, params: list = None):
    """
    Sends an HTTP POST query request to the Cloudflare D1 HTTP API.
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
    response = requests.post(url, json=payload, headers=headers)
    
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
        
    return query_result.get("results", [])

import json

def save_scan_result(tree_code: str, dbh_cm: float, tinggi_m: float, biomassa_kg: float,
                     karbon_kg: float, co2e_kg: float, splat_file_url: str, confidence_note: str,
                     thumbnail_url: str = None, geometry_3d: dict = None, species_predictions: list = None,
                     wood_density_used: float = None, wood_density_source: str = None,
                     climate_zone_detected: str = None, formula_used: str = None,
                     agb_kg: float = None, bgb_kg: float = None,
                     gps_lat: float = None, gps_lon: float = None):
    """
    Inserts a new scan record into the tree_scans database table on Cloudflare D1.
    """
    sql = """
    INSERT INTO tree_scans (
        tree_code, scan_date, dbh_cm, tinggi_m, biomassa_kg, karbon_kg, co2e_kg, 
        splat_file_url, confidence_note, thumbnail_url, geometry_3d, species_predictions,
        wood_density_used, wood_density_source, climate_zone_detected, formula_used,
        agb_kg, bgb_kg, gps_lat, gps_lon
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    # Use ISO 8601 UTC format for scan_date
    scan_date = datetime.now(timezone.utc).isoformat()
    geom_str = json.dumps(geometry_3d) if geometry_3d else None
    species_str = json.dumps(species_predictions) if species_predictions else None
    execute_d1_query(sql, [
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
        gps_lon
    ])

def update_scan_result(scan_id: int, dbh_cm: float, tinggi_m: float, biomassa_kg: float,
                       karbon_kg: float, co2e_kg: float, confidence_note: str,
                       geometry_3d: dict = None, wood_density_used: float = None,
                       wood_density_source: str = None, climate_zone_detected: str = None,
                       formula_used: str = None, agb_kg: float = None, bgb_kg: float = None,
                       gps_lat: float = None, gps_lon: float = None,
                       species_predictions: list = None):
    """
    Updates an existing scan record in the tree_scans database table on Cloudflare D1 by scan_id.
    """
    sql = """
    UPDATE tree_scans 
    SET dbh_cm = ?, tinggi_m = ?, biomassa_kg = ?, karbon_kg = ?, co2e_kg = ?, 
        confidence_note = ?, geometry_3d = ?, wood_density_used = ?, 
        wood_density_source = ?, climate_zone_detected = ?, formula_used = ?,
        agb_kg = ?, bgb_kg = ?, gps_lat = ?, gps_lon = ?, species_predictions = ?
    WHERE id = ?
    """
    geom_str = json.dumps(geometry_3d) if geometry_3d else None
    species_str = json.dumps(species_predictions) if species_predictions else None
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
        if r.get("formula_used") is None:
            h = r.get("tinggi_m")
            if h is not None and h > 0:
                r["formula_used"] = "Chave 2005 (moist forest with height)"
            else:
                r["formula_used"] = "Chave 2005 (moist forest, DBH-only)"
        if r.get("agb_kg") is None:
            total_biomass = r.get("biomassa_kg") or 0.0
            r["agb_kg"] = float(round(total_biomass / 1.24, 2))
        if r.get("bgb_kg") is None:
            total_biomass = r.get("biomassa_kg") or 0.0
            r["bgb_kg"] = float(round(total_biomass * 0.24 / 1.24, 2))

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
