import os
import sys
import subprocess

def get_mem_usage_mb():
    pid = os.getpid()
    if sys.platform == 'win32':
        try:
            cmd = f'powershell -NoProfile -Command "(Get-Process -Id {pid}).WorkingSet / 1MB"'
            output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
            return float(output)
        except Exception:
            return 0.0
    return 0.0

def main():
    print(f"Memory before importing server: {get_mem_usage_mb():.2f} MB")
    
    # Add parent directory to path so we can import server
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    import server
    
    print(f"Memory after importing server: {get_mem_usage_mb():.2f} MB")

if __name__ == "__main__":
    main()
