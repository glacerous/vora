import os
import time
from datetime import datetime, timezone

def main():
    path = "c:\\codes\\3dtest\\output"
    print(f"Checking files in {path}:")
    for f in os.listdir(path):
        fpath = os.path.join(path, f)
        if os.path.isfile(fpath):
            mtime = os.path.getmtime(fpath)
            dt_local = datetime.fromtimestamp(mtime)
            dt_utc = datetime.fromtimestamp(mtime, tz=timezone.utc)
            print(f"- {f}: size={os.path.getsize(fpath)} bytes, mtime_local={dt_local}, mtime_utc={dt_utc}")

if __name__ == "__main__":
    main()
