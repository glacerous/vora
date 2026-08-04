from dotenv import load_dotenv
load_dotenv()

from storage.d1_client import execute_d1_query

def main():
    try:
        print("Reading migration_012.sql...")
        with open("db/migration_012.sql", "r") as f:
            sql = f.read()
            
        print("Executing migration query on D1...")
        # Since execute_d1_query expects single statements or query, let's make sure it handles it.
        # SQLite queries in Cloudflare D1 can execute multiple statements in one SQL string.
        res = execute_d1_query(sql)
        print("Migration executed successfully. Result:", res)
        
        # Verify the new table exists
        tables = execute_d1_query("SELECT name FROM sqlite_master WHERE type='table'")
        print("Updated tables in database:")
        for t in tables:
            print(f"- {t['name']}")
    except Exception as e:
        print(f"Error running migration: {e}")

if __name__ == "__main__":
    main()
