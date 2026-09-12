import requests
import json
import time

def get_metrics():
    url = "https://vora-52k9.onrender.com/metrics"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"Error getting metrics: {e}")
    return None

def main():
    recalc_url = "https://vora-52k9.onrender.com/scan/46/recalculate"
    
    # 1. Check baseline
    initial = get_metrics()
    print(f"Initial Metrics: {initial}")
    if not initial:
        print("Failed to get initial metrics.")
        return
        
    payload = {
        "p1": [100.0, 200.0],
        "p2": [120.0, 220.0],
        "width": 512,
        "height": 512
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    print("\n--- Starting Stress Test: 10 consecutive recalculations ---")
    memory_history = []
    
    for i in range(1, 11):
        print(f"Iteration {i}/10...")
        t0 = time.time()
        try:
            res = requests.patch(recalc_url, data=json.dumps(payload), headers=headers, timeout=30)
            elapsed = time.time() - t0
            print(f"  Recalculate status: {res.status_code} in {elapsed:.2f}s")
            
            # Record metrics after request
            m = get_metrics()
            if m:
                print(f"  Memory status: Max RSS = {m['max_rss_mb']:.2f} MB, Current RSS = {m['current_rss_mb']:.2f} MB")
                memory_history.append((m['max_rss_mb'], m['current_rss_mb']))
            else:
                print("  Failed to retrieve metrics")
        except Exception as e:
            print(f"  Request failed: {e}")
        time.sleep(1) # short sleep
        
    print("\n--- Stress Test Completed ---")
    print(f"Baseline current memory: {initial.get('current_rss_mb'):.2f} MB")
    for idx, (max_rss, current_rss) in enumerate(memory_history, 1):
        print(f"Iteration {idx:02d}: Max RSS = {max_rss:.2f} MB, Current RSS = {current_rss:.2f} MB")

if __name__ == "__main__":
    main()
