#!/usr/bin/env python3
"""
Vora Backend Keep-Alive Script
-----------------------------
This script pings the Render.com backend API endpoint periodically to prevent
the free-tier instance from sleeping (spinning down after 15 minutes of inactivity).

How to run:
1. Locally (always on PC / server):
   python keep_alive.py --url https://vora-52k9.onrender.com/ping --interval 720

2. Free Cloud alternative (Recommended for production):
   Set up a free check on https://cron-job.org or https://uptimerobot.com
   pointing to: https://vora-52k9.onrender.com/ping
   configured to run every 12 minutes.
"""

import time
import argparse
import urllib.request
import urllib.error
from datetime import datetime

def ping_server(url: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Sending keep-alive request to: {url}")
    try:
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "VoraKeepAlive/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            status = response.status
            body = response.read().decode("utf-8")
            print(f"[{timestamp}] Success! Status Code: {status}. Response: {body}")
    except urllib.error.HTTPError as e:
        print(f"[{timestamp}] HTTP Error: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        print(f"[{timestamp}] URL Error: {e.reason}")
    except Exception as e:
        print(f"[{timestamp}] Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Keep Vora backend warm.")
    parser.add_argument(
        "--url", 
        default="https://vora-52k9.onrender.com/ping",
        help="Target ping endpoint url"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=720,  # 12 minutes in seconds (Render spins down after 15 mins)
        help="Interval between pings in seconds"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(" Vora Backend Keep-Alive Agent Running")
    print(f" Target URL: {args.url}")
    print(f" Interval  : {args.interval} seconds (~{args.interval / 60:.1f} mins)")
    print("=" * 60)
    
    # Run immediate first ping
    ping_server(args.url)
    
    try:
        while True:
            time.sleep(args.interval)
            ping_server(args.url)
    except KeyboardInterrupt:
        print("\nKeep-alive agent stopped.")

if __name__ == "__main__":
    main()
