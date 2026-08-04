import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from storage.d1_client import execute_d1_query

def main():
    try:
        tables = execute_d1_query("SELECT name FROM sqlite_master WHERE type='table'")
        print("Tables in D1 database:")
        for t in tables:
            print(f"- {t['name']}")
            try:
                count = execute_d1_query(f"SELECT COUNT(*) as cnt FROM {t['name']}")
                print(f"  Count: {count[0]['cnt']}")
            except Exception as e:
                print(f"  Error counting: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
