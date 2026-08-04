import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from storage.d1_client import execute_d1_query

def main():
    try:
        sql = "SELECT id, tree_code, scan_date, dbh_cm, tinggi_m, splat_file_url, confidence_note FROM tree_scans ORDER BY id DESC LIMIT 10"
        scans = execute_d1_query(sql)
        print("Last 10 scans in D1:")
        for s in scans:
            print(f"ID: {s['id']}, Code: {s['tree_code']}, Date: {s['scan_date']}, DBH: {s['dbh_cm']}, Height: {s['tinggi_m']}, Note: {s['confidence_note']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
