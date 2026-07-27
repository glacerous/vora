"""
Integration test for Cloudflare R2 and D1 storage.

Tests:
  1. R2 upload via upload_splat() using output/result.ply
  2. File accessibility via presigned URL (HTTP GET)
  3. CORS preflight check notes (documented separately)
  4. D1 insert via save_scan_result()
  5. D1 read via get_scan_history()

Run from project root:
    py test_storage.py
"""

import os
import sys
import requests
from dotenv import load_dotenv

# Load env variables from .env
load_dotenv()

from storage.r2_client import upload_splat
from storage.d1_client import save_scan_result, get_scan_history

TREE_CODE = "TEST-0001"
PLY_FILE  = "output/result.ply"

SEP = "=" * 60


def check_env():
    required = [
        "CLOUDFLARE_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET_NAME",
        "CLOUDFLARE_D1_DATABASE_ID",
        "CLOUDFLARE_API_TOKEN",
    ]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"[ERROR] Missing environment variables: {missing}")
        sys.exit(1)
    print("[OK] All environment variables loaded.")


def test_r2_upload():
    print(f"\n--- STEP 1: R2 Upload ({PLY_FILE} -> tree_code={TREE_CODE}) ---")
    if not os.path.exists(PLY_FILE):
        print(f"[ERROR] File not found: {PLY_FILE}")
        sys.exit(1)

    size_mb = os.path.getsize(PLY_FILE) / 1024 / 1024
    print(f"[OK] File exists: {PLY_FILE} ({size_mb:.2f} MB)")

    url = upload_splat(PLY_FILE, TREE_CODE)
    print("[OK] R2 upload succeeded!")
    print(f"     URL: {url}")
    return url


def test_file_download(url: str):
    """
    Tests file accessibility using the generated URL.

    NOTE ON SSL & PUBLIC r2.dev URLs:
    If R2_PUBLIC_URL_PREFIX points to a pub-xxx.r2.dev subdomain and the
    local network intercepts HTTPS (corporate proxy, VPN, antivirus), you
    will see:
        SSLError: hostname mismatch, certificate is not valid for '..r2.dev'
    This is a local network issue — the domain resolves to a local proxy IP
    instead of Cloudflare's CDN (172.x.x.x range).

    Presigned URLs via r2.cloudflarestorage.com bypass this because they use
    the underlying S3-compatible endpoint, which is not intercepted.

    FIX OPTIONS:
      A) Use a custom Cloudflare-proxied domain (e.g. assets.yourdomain.com
         -> CNAME to bucket.r2.dev) — avoids the intercept entirely.
      B) Set R2_PUBLIC_URL_PREFIX to empty to fall back to presigned URLs for
         server-side access, and use the r2.dev link only for frontend display.
    """
    print(f"\n--- STEP 2: File Download Accessibility Check ---")
    is_presigned = "X-Amz-Signature" in url
    if is_presigned:
        url_type = "presigned (r2.cloudflarestorage.com)"
    elif "r2.dev" in url:
        url_type = "public r2.dev"
    else:
        url_type = "public custom domain"
    print(f"     URL type: {url_type}")

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            print(f"[OK] File accessible! HTTP {resp.status_code} — {len(resp.content) / 1024 / 1024:.2f} MB received.")
        else:
            print(f"[WARN] HTTP {resp.status_code} — {resp.text[:200]}")
    except requests.exceptions.SSLError as e:
        print(f"[WARN] SSL error (likely local network intercept): {e}")
        print("       See docstring above for root cause and fix options.")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")


def test_cors(url: str):
    """
    Tests CORS preflight response.

    NOTE ON CORS:
    - r2.cloudflarestorage.com (S3 API endpoint): CORS is 403 for OPTIONS
      because this endpoint is for private S3 API access and not designed
      for browser-direct use.
    - pub-xxx.r2.dev (public CDN endpoint): CORS must be configured via
      the Cloudflare dashboard: R2 -> bucket -> Settings -> CORS Policy.
      Add: AllowedOrigins=["*"], AllowedMethods=["GET"], AllowedHeaders=["*"]
    - For viewer.html, the r2.dev public URL (or a custom domain) should be
      used — not the private S3 endpoint.
    """
    print(f"\n--- STEP 3: CORS Preflight Check ---")
    try:
        opts = requests.options(
            url,
            headers={
                "Origin": "http://localhost:8000",
                "Access-Control-Request-Method": "GET",
            },
            timeout=10,
        )
        allow_origin = opts.headers.get("Access-Control-Allow-Origin")
        if allow_origin:
            print(f"[OK] CORS enabled! Access-Control-Allow-Origin: {allow_origin}")
        else:
            is_s3 = "r2.cloudflarestorage.com" in url
            if is_s3:
                print("[INFO] OPTIONS 403 on S3 endpoint is expected (private API).")
                print("       Configure CORS on the public r2.dev URL in Cloudflare Dashboard.")
            else:
                print("[WARN] CORS headers missing on public URL.")
                print("       Action required: Add CORS policy in Cloudflare Dashboard.")
                print("       R2 -> vora-splats -> Settings -> CORS Policy:")
                print('       [{"AllowedOrigins":["*"],"AllowedMethods":["GET"],"AllowedHeaders":["*"]}]')
    except Exception as e:
        print(f"[ERROR] CORS check failed: {e}")


def test_d1_insert(splat_url: str):
    print(f"\n--- STEP 4: D1 Database Insert (tree_code={TREE_CODE}) ---")
    save_scan_result(
        tree_code=TREE_CODE,
        dbh_cm=25.5,
        tinggi_m=12.3,
        biomassa_kg=350.2,
        karbon_kg=175.1,
        co2e_kg=642.6,
        splat_file_url=splat_url,
        confidence_note="Validation test via test_storage.py",
    )
    print("[OK] D1 insert succeeded!")


def test_d1_read():
    print(f"\n--- STEP 5: D1 Scan History Read (tree_code={TREE_CODE}) ---")
    history = get_scan_history(TREE_CODE)
    print(f"[OK] D1 read succeeded! Found {len(history)} record(s).")
    for i, rec in enumerate(history, 1):
        print(f"  [{i}] id={rec.get('id')}  date={rec.get('scan_date')}")
        print(f"       dbh={rec.get('dbh_cm')}cm  height={rec.get('tinggi_m')}m")
        print(f"       biomass={rec.get('biomassa_kg')}kg  carbon={rec.get('karbon_kg')}kg  co2e={rec.get('co2e_kg')}kg")
        url_preview = (rec.get("splat_file_url") or "")[:80]
        print(f"       url={url_preview}...")


def main():
    print(SEP)
    print("  Cloudflare R2 & D1 Integration Verification")
    print(SEP)

    check_env()
    url = test_r2_upload()
    test_file_download(url)
    test_cors(url)
    test_d1_insert(url)
    test_d1_read()

    print(f"\n{SEP}")
    print("  All verification steps completed.")
    print(SEP)


if __name__ == "__main__":
    main()
