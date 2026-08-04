from dotenv import load_dotenv
load_dotenv()

from storage.d1_client import execute_d1_query

def main():
    try:
        tables = execute_d1_query("SELECT name FROM sqlite_master WHERE type='table'")
        print("Tables in D1 database:")
        for t in tables:
            print(f"- {t['name']}")
            
        # check columns of plots
        columns = execute_d1_query("PRAGMA table_info(plots)")
        print("\nColumns in 'plots' table:")
        for col in columns:
            print(f"- {col['name']} ({col['type']})")
    except Exception as e:
        print(f"Error querying database: {e}")

if __name__ == "__main__":
    main()
