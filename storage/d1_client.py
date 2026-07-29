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
                     thumbnail_url: str = None, geometry_3d: dict = None):
    """
    Inserts a new scan record into the tree_scans database table on Cloudflare D1.
    """
    sql = """
    INSERT INTO tree_scans (tree_code, scan_date, dbh_cm, tinggi_m, biomassa_kg, karbon_kg, co2e_kg, splat_file_url, confidence_note, thumbnail_url, geometry_3d)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    # Use ISO 8601 UTC format for scan_date
    scan_date = datetime.now(timezone.utc).isoformat()
    geom_str = json.dumps(geometry_3d) if geometry_3d else None
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
        geom_str
    ])

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
    return rows

def generate_tree_code() -> str:
    """
    Generates a unique tree code of format POHON-XXXX where XXXX is random alphanumeric.
    """
    chars = string.ascii_uppercase + string.digits
    code = "".join(secrets.choice(chars) for _ in range(4))
    return f"POHON-{code}"
