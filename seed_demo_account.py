import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from storage.auth_utils import hash_password
from storage.d1_client import execute_d1_query

def seed_demo():
    print("Starting database seeding for Demo Account...")
    
    # 1. Create demo user
    username = "juri_demo"
    password = "demo123" # Secure but easy to remember password for the jury
    display_name = "Juri Demo Vora"
    
    # Check if user already exists
    existing_users = execute_d1_query("SELECT id FROM users WHERE username = ?", [username])
    if existing_users:
        user_id = existing_users[0]["id"]
        print(f"User '{username}' already exists with ID: {user_id}. Reusing existing user.")
    else:
        pwd_hash, pwd_salt = hash_password(password)
        created_at = datetime.now(timezone.utc).isoformat()
        
        sql = """
        INSERT INTO users (username, password_hash, password_salt, display_name, is_demo_account, created_at)
        VALUES (?, ?, ?, ?, 1, ?)
        """
        execute_d1_query(sql, [username, pwd_hash, pwd_salt, display_name, created_at])
        
        # Get user ID
        new_user = execute_d1_query("SELECT id FROM users WHERE username = ?", [username])
        user_id = new_user[0]["id"]
        print(f"Created demo user '{username}' (password: '{password}') with ID: {user_id}")

    # 2. Create demo plots
    demo_plots = [
        {
            "code": "PLOT-MNGR",
            "name": "Plot Mangrove Muara Angke",
            "description": "Plot konservasi hutan mangrove di pesisir utara Jakarta. Pemantauan berkala emisi karbon biru (blue carbon).",
            "privacy": "public",
            "lat": -6.1088,
            "lon": 106.7891
        },
        {
            "code": "PLOT-BOGR",
            "name": "Plot Hutan Hujan Kebun Raya",
            "description": "Area penelitian keanekaragaman hayati dan biomasa pohon tropis di Kebun Raya Bogor.",
            "privacy": "private",
            "lat": -6.5975,
            "lon": 106.7997
        }
    ]
    
    plot_ids = {}
    for p in demo_plots:
        # Check if plot already exists
        existing_plots = execute_d1_query("SELECT id FROM plots WHERE plot_code = ?", [p["code"]])
        if existing_plots:
            plot_id = existing_plots[0]["id"]
            plot_ids[p["code"]] = plot_id
            print(f"Plot '{p['name']}' ({p['code']}) already exists with ID: {plot_id}. Reusing.")
        else:
            created_at = datetime.now(timezone.utc).isoformat()
            sql = """
            INSERT INTO plots (plot_code, owner_user_id, name, description, privacy, gps_centroid_lat, gps_centroid_lon, session_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """
            execute_d1_query(sql, [
                p["code"],
                user_id,
                p["name"],
                p["description"],
                p["privacy"],
                p["lat"],
                p["lon"],
                created_at,
                created_at
            ])
            new_plot = execute_d1_query("SELECT id FROM plots WHERE plot_code = ?", [p["code"]])
            plot_id = new_plot[0]["id"]
            plot_ids[p["code"]] = plot_id
            print(f"Created plot '{p['name']}' ({p['code']}) with ID: {plot_id}")

    # 3. Associate some scans with these plots
    # We will search the database for existing scans that are not yet claimed
    unclaimed_scans = execute_d1_query("SELECT * FROM tree_scans WHERE claimed_by_user_id IS NULL AND dbh_cm IS NOT NULL LIMIT 4")
    
    if len(unclaimed_scans) >= 2:
        print(f"Found {len(unclaimed_scans)} unclaimed scans in database. Linking them to demo plots...")
        # Link first scan to Mangrove plot
        execute_d1_query(
            "UPDATE tree_scans SET plot_id = ?, claimed_by_user_id = ?, gps_lat = ?, gps_lon = ? WHERE id = ?",
            [plot_ids["PLOT-MNGR"], user_id, -6.10885, 106.78915, unclaimed_scans[0]["id"]]
        )
        print(f"Linked scan '{unclaimed_scans[0]['tree_code']}' to plot PLOT-MNGR.")
        
        # Link second scan to Bogor plot
        execute_d1_query(
            "UPDATE tree_scans SET plot_id = ?, claimed_by_user_id = ?, gps_lat = ?, gps_lon = ? WHERE id = ?",
            [plot_ids["PLOT-BOGR"], user_id, -6.59755, 106.79975, unclaimed_scans[1]["id"]]
        )
        print(f"Linked scan '{unclaimed_scans[1]['tree_code']}' to plot PLOT-BOGR.")
        
        # If there are more scans, link them as well
        if len(unclaimed_scans) > 2:
            execute_d1_query(
                "UPDATE tree_scans SET plot_id = ?, claimed_by_user_id = ?, gps_lat = ?, gps_lon = ? WHERE id = ?",
                [plot_ids["PLOT-MNGR"], user_id, -6.10890, 106.78920, unclaimed_scans[2]["id"]]
            )
            print(f"Linked scan '{unclaimed_scans[2]['tree_code']}' to plot PLOT-MNGR.")
    else:
        # If no scans are found, insert mock scan records to ensure demo contains data
        print("No unclaimed scans found in database. Creating mock scan records for demonstration...")
        mock_scans = [
            {
                "code": "POHON-MNGR1",
                "dbh": 24.5,
                "tinggi": 12.3,
                "biomassa": 220.4,
                "karbon": 110.2,
                "co2e": 404.4,
                "unc_pct": 14.5,
                "lat": -6.10885,
                "lon": 106.78915,
                "plot_id": plot_ids["PLOT-MNGR"]
            },
            {
                "code": "POHON-MNGR2",
                "dbh": 18.2,
                "tinggi": 9.8,
                "biomassa": 135.0,
                "karbon": 67.5,
                "co2e": 247.7,
                "unc_pct": 15.2,
                "lat": -6.10892,
                "lon": 106.78922,
                "plot_id": plot_ids["PLOT-MNGR"]
            },
            {
                "code": "POHON-BOGR1",
                "dbh": 45.2,
                "tinggi": 24.5,
                "biomassa": 950.0,
                "karbon": 475.0,
                "co2e": 1743.2,
                "unc_pct": 12.1,
                "lat": -6.59755,
                "lon": 106.79975,
                "plot_id": plot_ids["PLOT-BOGR"]
            }
        ]
        
        for idx, m in enumerate(mock_scans, 1):
            # Check if exists
            existing = execute_d1_query("SELECT id FROM tree_scans WHERE tree_code = ?", [m["code"]])
            if existing:
                print(f"Mock scan '{m['code']}' already exists. Skipping insertion.")
                continue
                
            scan_date = datetime.now(timezone.utc).isoformat()
            sql = """
            INSERT INTO tree_scans (
                tree_code, scan_date, dbh_cm, tinggi_m, biomassa_kg, karbon_kg, co2e_kg, 
                confidence_note, gps_lat, gps_lon, plot_id, claimed_by_user_id,
                co2e_uncertainty_pct, co2e_low_kg, co2e_high_kg, quality_status, scale_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Seeded mock data for demo', ?, ?, ?, ?, ?, ?, ?, 'ok', 'calibrated')
            """
            
            # calculate co2e range
            low_co2e = m["co2e"] * (1 - m["unc_pct"] / 100.0)
            high_co2e = m["co2e"] * (1 + m["unc_pct"] / 100.0)
            
            execute_d1_query(sql, [
                m["code"],
                scan_date,
                m["dbh"],
                m["tinggi"],
                m["biomassa"],
                m["karbon"],
                m["co2e"],
                m["lat"],
                m["lon"],
                m["plot_id"],
                user_id,
                m["unc_pct"],
                low_co2e,
                high_co2e
            ])
            print(f"Inserted mock scan '{m['code']}' into plot.")

    print("\nDatabase seeding completed successfully!")
    print(f"Demo credentials for jury: Username: '{username}' | Password: '{password}'")

if __name__ == "__main__":
    seed_demo()
