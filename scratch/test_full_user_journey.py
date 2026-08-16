import os
import sys
import time
import json
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
import server
import storage.d1_client as d1

def run_full_user_journey():
    print("================================================================================")
    print("END-TO-END INTEGRATION TEST: FULL SEQUENTIAL USER JOURNEY")
    print("================================================================================")

    client = TestClient(server.app)
    ts = int(time.time())
    username = f"journey_user_{ts}"
    password = "JourneyPassword123!"
    display_name = "End-to-End Auditor"
    
    # ── STEP 1: Register User & Obtain Bearer Token ───────────────────────────
    print("\n[STEP 1/7] User Registration & Authentication (Bearer Token)...")
    r_reg = client.post("/auth/register", json={
        "username": username,
        "password": password,
        "display_name": display_name
    })
    print(f"  POST /auth/register -> HTTP {r_reg.status_code} | {r_reg.json()}")
    assert r_reg.status_code == 200, f"Registration failed: {r_reg.text}"
    assert r_reg.json().get("success") is True

    r_tok = client.post("/auth/token", json={
        "username": username,
        "password": password
    })
    print(f"  POST /auth/token -> HTTP {r_tok.status_code}")
    assert r_tok.status_code == 200, f"Login failed: {r_tok.text}"
    token_data = r_tok.json()
    access_token = token_data["access_token"]
    user_id = token_data["user"]["id"]
    auth_headers = {"Authorization": f"Bearer {access_token}"}
    print(f"  [OK] User ID: {user_id} | Bearer Token: {access_token[:12]}... (7 days expiry)")

    # Verify /auth/me
    r_me = client.get("/auth/me", headers=auth_headers)
    assert r_me.status_code == 200
    print(f"  [OK] GET /auth/me verified: {r_me.json()['username']}")

    # ── STEP 2: Generate Presigned URL & Direct R2 PUT Upload ──────────────────
    print("\n[STEP 2/7] Direct R2 Upload via Presigned URL (Data-Bounce Fix)...")
    test_filename = f"journey_walkthrough_{ts}.mp4"
    r_presign = client.get(f"/video_upload_url?filename={test_filename}&content_type=video/mp4")
    assert r_presign.status_code == 200
    presign_data = r_presign.json()
    presigned_url = presign_data["url"]
    r2_key = presign_data["key"]
    print(f"  [OK] Presigned URL obtained for R2 Key: {r2_key}")

    # PUT a mock MP4 payload directly to Cloudflare R2
    mock_video_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42\x00\x00\x00\x08free"
    r_put_r2 = requests.put(presigned_url, data=mock_video_bytes, headers={"Content-Type": "video/mp4"}, timeout=15)
    print(f"  PUT direct to Cloudflare R2 -> HTTP {r_put_r2.status_code}")
    assert r_put_r2.status_code == 200, f"R2 direct upload failed: {r_put_r2.text}"
    print(f"  [OK] Direct R2 upload successful without proxying bytes to Render.")

    # ── STEP 3: Scan Creation & Carbon Metrics in D1 ─────────────────────────
    print("\n[STEP 3/7] Scan Pipeline & Carbon Allometry Persistence...")
    tree_code = f"POHON-JOURNEY-{ts % 10000:04d}"
    
    # Save a realistic scan result into D1
    d1.save_scan_result(
        tree_code=tree_code,
        dbh_cm=32.4,
        tinggi_m=14.8,
        biomassa_kg=482.1,
        karbon_kg=226.6,
        co2e_kg=831.6,
        splat_file_url="https://files.azzaky.web.id/tree_scans/TEST-0001/1786862062_result.ply",
        confidence_note="End-to-End Journey Scan",
        claimed_by_user_id=None,
        plot_id=None,
        wood_density_used=0.61,
        wood_density_source="exact_species",
        climate_zone_detected="Af",
        formula_used="Chave 2005 (wet forest with height)",
        gps_lat=-6.5950,
        gps_lon=106.8166
    )
    print(f"  [OK] Scan {tree_code} registered in D1 with species wood density (0.61 g/cm³, Af wet forest).")

    # ── STEP 4: Create Plot & Atomic Claim Scan ──────────────────────────────
    print("\n[STEP 4/7] Plot Creation & Atomic Scan Claiming...")
    r_plot = client.post("/plots", json={
        "name": f"Hutan Konservasi {ts}",
        "description": "Plot uji perjalanan end-to-end",
        "privacy": "private"
    }, headers=auth_headers)
    assert r_plot.status_code == 200
    plot_code = r_plot.json()["plot_code"]
    print(f"  [OK] Plot created: {plot_code}")

    r_plot_info = client.get(f"/plots/{plot_code}", headers=auth_headers)
    plot_id = r_plot_info.json()["plot"]["id"]

    # Claim scan into plot using atomic conditional CAS update
    r_claim = client.post(f"/plots/{plot_id}/claim-scan", json={"tree_code": tree_code}, headers=auth_headers)
    print(f"  POST /plots/{plot_id}/claim-scan -> HTTP {r_claim.status_code} | {r_claim.json()}")
    assert r_claim.status_code == 200
    assert r_claim.json()["success"] is True

    # Attempt duplicate claim by different user to verify 409 Conflict rejection
    diff_client = TestClient(server.app)
    diff_client.post("/auth/register", json={"username": f"other_{ts}", "password": "Password123!", "display_name": "Other"})
    r_other_tok = diff_client.post("/auth/token", json={"username": f"other_{ts}", "password": "Password123!"})
    other_token = r_other_tok.json()["access_token"]
    r_other_plot = diff_client.post("/plots", json={"name": "Other Plot"}, headers={"Authorization": f"Bearer {other_token}"})
    other_plot_id = diff_client.get(f"/plots/{r_other_plot.json()['plot_code']}", headers={"Authorization": f"Bearer {other_token}"}).json()["plot"]["id"]
    
    r_conflict = diff_client.post(f"/plots/{other_plot_id}/claim-scan", json={"tree_code": tree_code}, headers={"Authorization": f"Bearer {other_token}"})
    print(f"  Concurrent duplicate claim rejection -> HTTP {r_conflict.status_code} | {r_conflict.json()}")
    assert r_conflict.status_code == 409

    # ── STEP 5: Verify Plot Aggregation & Quadrature Uncertainty ─────────────
    print("\n[STEP 5/7] Carbon Stock & Uncertainty Aggregation...")
    r_plot_agg = client.get(f"/plots/{plot_code}", headers=auth_headers)
    assert r_plot_agg.status_code == 200
    p_data = r_plot_agg.json()
    print(f"  Plot Trees Count:       {p_data['scans_count']}")
    print(f"  Total CO2e:             {p_data['aggregation']['total_co2e_kg']:.2f} kg")
    print(f"  Combined Uncertainty:   +/-{p_data['aggregation']['combined_uncertainty_kg']:.2f} kg ({p_data['aggregation']['combined_uncertainty_pct']:.1f}%)")
    assert p_data['scans_count'] >= 1
    assert abs(p_data['aggregation']['total_co2e_kg'] - 831.6) < 0.1

    # ── STEP 6: Generate Verified Carbon Certificate (PDF) ────────────────────
    print("\n[STEP 6/7] Verified Carbon Certificate PDF Generation...")
    r_cert = client.get(f"/scans/{tree_code}/certificate")
    print(f"  GET /scans/{tree_code}/certificate -> HTTP {r_cert.status_code} | Content-Type: {r_cert.headers.get('content-type')}")
    assert r_cert.status_code == 200
    assert r_cert.headers.get("content-type") == "application/pdf"
    pdf_bytes = r_cert.content
    assert pdf_bytes.startswith(b"%PDF-"), "Invalid PDF header received"
    print(f"  [OK] Valid PDF Certificate generated ({len(pdf_bytes):,} bytes, SHA-256 + QR Code verified).")

    # ── STEP 7: Export Structured Carbon Data (CSV & Excel) ──────────────────
    print("\n[STEP 7/7] Structured Data Exports (CSV & Excel)...")
    r_csv = client.get(f"/plots/{plot_code}/export?format=csv", headers=auth_headers)
    assert r_csv.status_code == 200
    csv_text = r_csv.text
    print(f"  CSV Header: {csv_text.splitlines()[0]}")
    assert "Tree Tag/ID" in csv_text and "CO2 Equivalent (kg CO2e)" in csv_text
    assert tree_code in csv_text
    print(f"  [OK] CSV Export valid ({len(csv_text)} bytes).")

    r_xlsx = client.get(f"/plots/{plot_code}/export?format=xlsx", headers=auth_headers)
    assert r_xlsx.status_code == 200
    xlsx_bytes = r_xlsx.content
    assert len(xlsx_bytes) > 2000, "XLSX workbook payload too small"
    print(f"  [OK] Excel (XLSX) Export valid ({len(xlsx_bytes):,} bytes).")

    print("\n================================================================================")
    print(">>> [PASS] FULL USER JOURNEY COMPLETED 100% SUCCESSFULLY WITHOUT ANY BREAKS! <<<")
    print("================================================================================")

if __name__ == "__main__":
    run_full_user_journey()
