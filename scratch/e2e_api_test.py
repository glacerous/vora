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
        print("\n=== E2E TEST PASSED SUCCESSFULLY! ===")
    else:
        print("\n=== E2E TEST FAILED: Association IDs mismatch ===")
        sys.exit(1)

if __name__ == "__main__":
    # Wait 2 seconds for server to reload changes if needed
    time.sleep(2)
    run_test()
