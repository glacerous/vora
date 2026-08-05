import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dotenv
dotenv.load_dotenv()
from storage.d1_client import execute_d1_query

def main():
    try:
        sql = "SELECT id, tree_code, scan_date, dbh_cm, tinggi_m, confidence_note, geometry_3d FROM tree_scans ORDER BY id DESC LIMIT 1"
        scans = execute_d1_query(sql)
        if scans:
            s = scans[0]
            print(f"ID: {s['id']}")
            print(f"Code: {s['tree_code']}")
            print(f"Date: {s['scan_date']}")
            print(f"DBH: {s['dbh_cm']}")
            print(f"Height: {s['tinggi_m']}")
            print(f"Confidence Note:\n{s['confidence_note']}")
            print(f"Geometry 3D:\n{str(s['geometry_3d'])[:400]}...")
        else:
            print("No scans found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
