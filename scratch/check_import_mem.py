import os
import sys
import subprocess

def get_mem_usage_mb():
    pid = os.getpid()
    if sys.platform == 'win32':
        try:
            cmd = f'powershell -NoProfile -Command "(Get-Process -Id {pid}).WorkingSet / 1MB"'
            output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
            # Parse float from output
            return float(output)
        except Exception as e:
            return 0.0
    # Fallback for Linux/macOS
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0

def test_import(module_name):
    mem_before = get_mem_usage_mb()
    try:
        __import__(module_name)
        mem_after = get_mem_usage_mb()
        diff = mem_after - mem_before
        print(f"Imported {module_name:<30} | Memory increased: {diff:6.2f} MB | Total: {mem_after:6.2f} MB")
    except Exception as e:
        print(f"Failed to import {module_name}: {e}")

def main():
    print(f"Initial Memory Usage: {get_mem_usage_mb():.2f} MB\n")
    
    print("--- Eager imports in server.py ---")
    test_import("fastapi")
    test_import("uvicorn")
    test_import("boto3")
    test_import("requests")
    test_import("numpy")
    test_import("cv2")
    
    print("\n--- Optional heavy packages in requirements.txt ---")
    test_import("mediapipe")
    test_import("onnxruntime")
    test_import("rembg")
    test_import("scipy")
    test_import("scipy.spatial")

if __name__ == "__main__":
    main()
