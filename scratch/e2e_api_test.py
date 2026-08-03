import os
import sys
import time
import requests
import secrets
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

BACKEND_URL = "http://localhost:8000"

def run_test():
    print("=== STARTING E2E API INTEGRATION TEST ===")
    
    # 1. Register a new user
    username = f"e2e_user_{int(time.time())}"
    password = "testpassword123"
    display_name = "E2E Test Agent"
    
    print(f"Registering user '{username}'...")
    reg_res = requests.post(f"{BACKEND_URL}/auth/register", json={
        "username": username,
        "password": password,
        "display_name": display_name
    })
    
    if reg_res.status_code != 200:
        print(f"Registration failed: {reg_res.text}")
        sys.exit(1)
    print("SUCCESS: Registration successful!")
    
    # 2. Login to get session cookie
    print("Logging in...")
    session = requests.Session()
    login_res = session.post(f"{BACKEND_URL}/auth/login", json={
        "username": username,
        "password": password
    })
    
    if login_res.status_code != 200:
        print(f"Login failed: {login_res.text}")
        sys.exit(1)
    
    login_data = login_res.json()
    print(f"SUCCESS: Login successful! User ID: {login_data['user']['id']}")
    print(f"Cookies received: {session.cookies.get_dict()}")
    
    # 3. Create a new plot
    plot_name = f"E2E Plot {int(time.time())}"
    print(f"Creating plot '{plot_name}'...")
    plot_res = session.post(f"{BACKEND_URL}/plots", json={
        "name": plot_name,
        "description": "E2E automated testing plot.",
        "privacy": "private",
        "gps_centroid_lat": -6.1234,
        "gps_centroid_lon": 106.5678
    })
    
    if plot_res.status_code != 200:
        print(f"Plot creation failed: {plot_res.text}")
        sys.exit(1)
        
    plot_data = plot_res.json()
    plot_code = plot_data["plot_code"]
    
    # Query database to get plot ID
    from storage.d1_client import execute_d1_query
    db_plots = execute_d1_query("SELECT id FROM plots WHERE plot_code = ?", [plot_code])
    if not db_plots:
        print("FAIL: Created plot not found in database!")
        sys.exit(1)
    plot_id = db_plots[0]["id"]
    print(f"SUCCESS: Plot created! ID: {plot_id}, Code: {plot_code}")
    
    # 4. Start plot session
    print(f"Starting scan session on plot {plot_id}...")
    start_res = session.post(f"{BACKEND_URL}/plots/{plot_id}/session/start")
    if start_res.status_code != 200:
        print(f"Failed to start plot session: {start_res.text}")
        sys.exit(1)
    print("SUCCESS: Plot session started successfully!")
    
    # 5. Call save_scan_result directly or trigger pipeline
    # Since reconstruct runs in background and requires Modal app, let's call save_scan_result directly or trigger a mock reconstruct.
    # Actually, we can invoke save_scan_result directly from storage.d1_client using python, or we can mock it!
    # Let's import save_scan_result and call it with plot_id and claimed_by_user_id to simulate what the pipeline does!
    from storage.d1_client import save_scan_result, execute_d1_query
    
    test_tree_code = f"E2E-TREE-{secrets.token_hex(3).upper()}"
    print(f"Simulating scan pipeline complete for tree '{test_tree_code}' under active session...")
    
    save_scan_result(
        tree_code=test_tree_code,
        dbh_cm=22.4,
        tinggi_m=11.2,
        biomassa_kg=180.5,
        karbon_kg=90.25,
        co2e_kg=331.0,
        splat_file_url="https://files.azzaky.web.id/test.splat",
        confidence_note="E2E test auto-association",
        gps_lat=-6.12345,
        gps_lon=106.56785,
        plot_id=plot_id,
        claimed_by_user_id=login_data['user']['id'],
        co2e_uncertainty_pct=15.0,
        co2e_low_kg=281.35,
        co2e_high_kg=380.65
    )
    print("SUCCESS: Save scan result called successfully!")
    
    # 6. Verify scan in database
    print("Querying database to verify scan was inserted and linked correctly...")
    db_scans = execute_d1_query(
        "SELECT * FROM tree_scans WHERE tree_code = ?",
        [test_tree_code]
    )
    
    if not db_scans:
        print("FAIL: Scan not found in database!")
        sys.exit(1)
        
    scan = db_scans[0]
    print(f"SUCCESS: Scan record found!")
    print(f"  - tree_code: {scan['tree_code']}")
    print(f"  - plot_id: {scan['plot_id']} (Expected: {plot_id})")
    print(f"  - claimed_by_user_id: {scan['claimed_by_user_id']} (Expected: {login_data['user']['id']})")
    
    if scan['plot_id'] == plot_id and scan['claimed_by_user_id'] == login_data['user']['id']:
        print("SUCCESS: Scan auto-associated correctly!")
    else:
        print("FAIL: Association IDs mismatch")
        sys.exit(1)

    # 7. Test save grid layout positions API
    print("Testing grid layout save API...")
    layout_payload = {
        "layout": [
            {
                "tree_code": test_tree_code,
                "grid_position_x": 3,
                "grid_position_y": 4
            }
        ]
    }
    
    layout_res = session.post(f"{BACKEND_URL}/plots/{plot_id}/layout", json=layout_payload)
    if layout_res.status_code != 200:
        print(f"FAIL: Grid layout save failed with status {layout_res.status_code}: {layout_res.text}")
        sys.exit(1)
        
    print("SUCCESS: Grid layout positions saved successfully!")
    
    # 8. Verify coordinates in database
    print("Verifying coordinates in D1 database...")
    db_scans_updated = execute_d1_query(
        "SELECT grid_position_x, grid_position_y FROM tree_scans WHERE tree_code = ?",
        [test_tree_code]
    )
    if not db_scans_updated:
        print("FAIL: Scan not found after coordinate save!")
        sys.exit(1)
        
    updated_scan = db_scans_updated[0]
    print(f"Updated scan coordinate in D1: x={updated_scan['grid_position_x']}, y={updated_scan['grid_position_y']}")
    if updated_scan['grid_position_x'] != 3 or updated_scan['grid_position_y'] != 4:
        print("FAIL: Grid coordinates mismatch in D1!")
        sys.exit(1)
        
    print("SUCCESS: Grid coordinates verified in database!")
    
    # 9. Verify coordinates in get plot details API
    print("Verifying coordinates returned in plot details API...")
    details_res = session.get(f"{BACKEND_URL}/plots/{plot_code}")
    if details_res.status_code != 200:
        print(f"FAIL: Get plot details failed: {details_res.text}")
        sys.exit(1)
        
    details_data = details_res.json()
    scans_list = details_data.get("scans", [])
    matched_scan = next((s for s in scans_list if s["tree_code"] == test_tree_code), None)
    if not matched_scan:
        print("FAIL: Matching scan not found in API response scans list!")
        sys.exit(1)
        
    print(f"Coordinates returned from API: x={matched_scan.get('grid_position_x')}, y={matched_scan.get('grid_position_y')}")
    if matched_scan.get('grid_position_x') != 3 or matched_scan.get('grid_position_y') != 4:
        print("FAIL: Coordinates from API mismatch!")
        sys.exit(1)
        
    print("SUCCESS: Grid coordinates returned correctly in API get plot details!")
    
    print("\n=== E2E TEST PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    # Wait 2 seconds for server to reload changes if needed
    time.sleep(2)
    run_test()
