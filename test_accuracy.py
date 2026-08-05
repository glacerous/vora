"""
test_accuracy.py — Regression tests for the carbon-accuracy improvements in Vora.

Covers Priority 1 (scale calibration flagging + auto-pose), Priority 2 (full-tree
height validation & DBH-only fallback), Priority 3 (per-forest-type root-to-shoot
ratio), plus uncertainty interval and the reconstruction quality gate.

Run:  python test_accuracy.py
"""
import os
import sys
import math
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

PASS = []


def check(name, cond, extra=""):
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name} {extra}")
    else:
        print(f"  [FAIL] {name} {extra}")
        raise AssertionError(name + " " + extra)


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY 3 — Root-to-Shoot ratio per forest_type
# ─────────────────────────────────────────────────────────────────────────────
def test_root_to_shoot_ratio():
    print("\n=== PRIORITY 3: Root-to-Shoot ratio per forest_type ===")
    from carbon import allometric

    rs_wet = allometric.get_root_to_shoot_ratio("wet")
    rs_moist = allometric.get_root_to_shoot_ratio("moist")
    rs_dry = allometric.get_root_to_shoot_ratio("dry")
    check("wet/moist RS default 0.37", rs_wet == 0.37 and rs_moist == 0.37, f"({rs_moist})")
    check("dry RS 0.28", abs(rs_dry - 0.28) < 1e-9, f"({rs_dry})")
    check("unknown -> moist 0.37", allometric.get_root_to_shoot_ratio("bogus") == 0.37)

    D = 30.0
    c_moist = allometric.estimate_carbon(D, wood_density=0.6, forest_type="moist")
    c_dry = allometric.estimate_carbon(D, wood_density=0.6, forest_type="dry")
    check("response has root_to_shoot_ratio", c_moist["root_to_shoot_ratio"] == 0.37)
    check("response has root_to_shoot_source",
          "IPCC" in c_moist.get("root_to_shoot_source", ""))

    check("BGB(moist)=AGB*0.37",
          abs(c_moist["below_ground_biomass_kg"] - c_moist["above_ground_biomass_kg"] * 0.37) < 0.01)
    check("BGB(dry)=AGB*0.28",
          abs(c_dry["below_ground_biomass_kg"] - c_dry["above_ground_biomass_kg"] * 0.28) < 0.01)

    old_bgb = c_moist["above_ground_biomass_kg"] * 0.24
    check("moist BGB increased vs old 0.24",
          c_moist["below_ground_biomass_kg"] > old_bgb,
          f"new={c_moist['below_ground_biomass_kg']:.2f} old={old_bgb:.2f}")
    print(f"      Moist BGB: old*0.24={old_bgb:.2f} kg -> new*0.37={c_moist['below_ground_biomass_kg']:.2f} kg")


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY 3 (cont.) — Uncertainty interval on CO2e
# ─────────────────────────────────────────────────────────────────────────────
def test_uncertainty():
    print("\n=== Uncertainty interval on CO2e ===")
    from carbon import allometric

    c = allometric.estimate_carbon(30, height_m=20, uncertainty_pct=15.0)
    check("co2e_uncertainty_pct stored", c["co2e_uncertainty_pct"] == 15.0)
    low = c["co2e_kg"] * (1 - 0.15)
    high = c["co2e_kg"] * (1 + 0.15)
    check("co2e_low correct", abs(c["co2e_low_kg"] - low) < 0.01, f"({c['co2e_low_kg']} vs {low:.2f})")
    check("co2e_high correct", abs(c["co2e_high_kg"] - high) < 0.01, f"({c['co2e_high_kg']} vs {high:.2f})")
    check("range brackets point estimate", c["co2e_low_kg"] < c["co2e_kg"] < c["co2e_high_kg"])
    print(f"      CO2e = {c['co2e_kg']:.2f} kg  (±15%: {c['co2e_low_kg']:.2f}–{c['co2e_high_kg']:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY 2 — is_full_tree_height heuristic
# ─────────────────────────────────────────────────────────────────────────────
def _trunk_segment_cloud(radius=0.25, height_m=2.0, scale=1.0, seed=0):
    """A mid-trunk cylindrical segment only (no ground, no canopy)."""
    rng = np.random.default_rng(seed)
    n = 3000
    ang = rng.uniform(0, 2 * np.pi, n)
    r = radius * (1 + rng.normal(0, 0.01, n))
    z = np.linspace(0, height_m / scale, n)
    x = r * np.cos(ang)
    y = r * np.sin(ang)
    return np.column_stack((x, y, z))


def _full_tree_cloud(trunk_radius=0.25, height_m=8.0, scale=1.0, seed=1):
    """Ground base -> trunk -> canopy (wide crown at the top)."""
    rng = np.random.default_rng(seed)
    n_trunk = 4000
    ang = rng.uniform(0, 2 * np.pi, n_trunk)
    r = trunk_radius * (1 + rng.normal(0, 0.01, n_trunk))
    zt = np.linspace(0, height_m * 0.7 / scale, n_trunk)
    x = r * np.cos(ang)
    y = r * np.sin(ang)
    z = zt
    n_base = 800
    z_base = rng.uniform(0, 0.5)
    r_base = rng.uniform(0, trunk_radius * 2)
    a_base = rng.uniform(0, 2 * np.pi)
    pts_base = np.column_stack((r_base * np.cos(a_base), r_base * np.sin(a_base), z_base))
    n_top = 5000
    z_top = rng.uniform(height_m * 0.75 / scale, height_m / scale)
    r_top = rng.uniform(0, trunk_radius * 6)
    a_top = rng.uniform(0, 2 * np.pi)
    pts_top = np.column_stack((r_top * np.cos(a_top), r_top * np.sin(a_top), z_top))
    pts = np.vstack([np.column_stack((x, y, z)), pts_base, pts_top])
    return pts


def test_full_tree_height():
    print("\n=== PRIORITY 2: is_full_tree_height ===")
    from carbon.dbh_extractor import is_full_tree_height

    seg = _trunk_segment_cloud(height_m=2.0)
    full = _full_tree_cloud(height_m=8.0)

    is_full_seg, reason_seg = is_full_tree_height(seg, vertical_axis_idx=2, scale=1.0)
    is_full_full, reason_full = is_full_tree_height(full, vertical_axis_idx=2, scale=1.0)

    check("trunk segment -> NOT full height", is_full_seg is False, f"({reason_seg})")
    check("full ground-to-canopy -> full height", is_full_full is True, f"({reason_full})")

    seg_short = _trunk_segment_cloud(height_m=2.0, seed=7)
    seg_long = _trunk_segment_cloud(height_m=3.5, seed=8)
    f_s, _ = is_full_tree_height(seg_short, vertical_axis_idx=2, scale=1.0)
    f_l, _ = is_full_tree_height(seg_long, vertical_axis_idx=2, scale=1.0)
    check("both short & longer trunk segments rejected (DBH-only stable)",
          (f_s is False) and (f_l is False))

    from carbon.allometric import estimate_carbon
    dbh_cm = (2 * 0.25) * 100.0
    c_a = estimate_carbon(dbh_cm, height_m=None)
    c_b = estimate_carbon(dbh_cm, height_m=None)
    check("DBH-only result stable for same DBH",
          c_a["above_ground_biomass_kg"] == c_b["above_ground_biomass_kg"])


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY 1 — auto-pose calibration (pure helpers, cv2/mediapipe not needed)
# ─────────────────────────────────────────────────────────────────────────────
def test_auto_pose_scale():
    print("\n=== PRIORITY 1: auto-pose scale calibration ===")
    import carbon.height_calibration as hc

    rng = np.random.default_rng(42)
    tree = np.column_stack((
        0.1 + rng.normal(0, 0.02, 5000),
        0.1 + rng.normal(0, 0.02, 5000),
        np.linspace(0, 10.0, 5000),
    ))
    person = np.column_stack((
        10.0 + rng.normal(0, 0.05, 2500),
        10.0 + rng.normal(0, 0.05, 2500),
        np.linspace(0, 1.65, 2500),
    ))
    pts = np.vstack([tree, person])

    sf = hc._find_person_scale_in_cloud(pts, person_height_m=1.65, axis_idx=2)
    check("person-scale found ~1.0", sf is not None and abs(sf - 1.0) < 1e-6, f"(sf={sf})")
    print(f"      derived scale_factor = {sf:.6f} (expect ~1.0)")

    original_detect = hc.detect_person_pose

    def fake_detect_high(_path):
        return {"head": (0, 0), "foot": (0, 200), "confidence": 0.9}

    try:
        hc.detect_person_pose = fake_detect_high
        res = hc.auto_calibrate_scale_from_frames([__file__], points_3d=pts,
                                                  person_height_m=1.65, min_confidence=0.6,
                                                  vertical_axis_idx=2)
        check("auto-pose marks calibrated", res and res["is_calibrated"] is True, f"({res})")
        check("auto-pose source is auto_pose", res and res["source"] == "auto_pose")
        check("auto-pose applies person scale", res and abs(res["scale_factor"] - 1.0) < 1e-6)

        hc.detect_person_pose = lambda _p: {"head": (0, 0), "foot": (0, 200), "confidence": 0.3}
        res_low = hc.auto_calibrate_scale_from_frames([__file__], points_3d=pts,
                                                      person_height_m=1.65, min_confidence=0.6)
        check("low-confidence person -> uncalibrated", res_low and res_low["is_calibrated"] is False)

        hc.detect_person_pose = lambda _p: None
        res_none = hc.auto_calibrate_scale_from_frames([__file__], points_3d=pts,
                                                       person_height_m=1.65, min_confidence=0.6)
        check("no person -> uncalibrated", res_none and res_none["is_calibrated"] is False)
    finally:
        hc.detect_person_pose = original_detect


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY 1 — _load_scale_factor_for_scan / scale_status (server side)
# The server module needs cv2/fastapi/boto3; run these only if importable.
# ─────────────────────────────────────────────────────────────────────────────
def test_server_scale_loading():
    print("\n=== PRIORITY 1: server scale loading & scale_status (if server importable) ===")
    try:
        import server  # noqa
    except Exception as e:
        print(f"  [SKIP] server not importable in this env ({type(e).__name__}: {e}). "
              f"Run with full deps (fastapi, boto3, opencv, dotenv).")
        return

    calib_path = os.path.join(server.BASE_DIR, "calibration.json")
    had_file = os.path.exists(calib_path)
    saved = None
    if had_file:
        with open(calib_path, "r") as fh:
            saved = fh.read()
        os.remove(calib_path)

    try:
        # No calibration file -> uncalibrated default fallback (scale 1.0, NOT calibrated)
        sf, is_cal, src = server._load_scale_factor_for_scan("POHON-TEST")
        check("no calibration.json -> uncalibrated_default",
              is_cal is False and sf == 1.0 and src == "uncalibrated_default", f"({is_cal},{src})")

        # Manual default calibration -> calibrated
        with open(calib_path, "w") as fh:
            fh.write('{"default": {"scale_factor": 0.0555}}')
        sf, is_cal, src = server._load_scale_factor_for_scan("POHON-TEST")
        check("manual calibration -> calibrated",
              is_cal is True and abs(sf - 0.0555) < 1e-9 and src == "manual_default")

        # scan-specific beats default
        with open(calib_path, "w") as fh:
            fh.write('{"POHON-AAA": {"scale_factor": 0.02}, "default": {"scale_factor": 0.0555}}')
        sf, is_cal, src = server._load_scale_factor_for_scan("POHON-AAA")
        check("scan-specific beats default",
              is_cal and abs(sf - 0.02) < 1e-9 and src == "manual_scan_specific")
    finally:
        if had_file and saved is not None:
            with open(calib_path, "w") as fh:
                fh.write(saved)
        elif os.path.exists(calib_path):
            os.remove(calib_path)


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY 2 (lanjutan) — height validation di jalur manual override
# resolve_height_usage() adalah helper terpusat yang dipakai oleh run_carbon_analysis
# DAN semua endpoint manual (manual override, recalculate, adjust-geometry).
# ─────────────────────────────────────────────────────────────────────────────
def _write_ascii_ply(path, points):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n" % len(points))
        for p in points:
            f.write("%.6f %.6f %.6f\n" % (p[0], p[1], p[2]))


def test_manual_endpoint_height_validation():
    print("\n=== PRIORITY 2 (lanjutan): height validation di jalur manual ===")
    from carbon.dbh_extractor import resolve_height_usage

    tmp = tempfile.mkdtemp()

    # (a) system height + trunk-segment-only cloud -> forced DBH-only fallback
    seg = _trunk_segment_cloud(height_m=2.0, radius=0.25)
    # swap Y and Z to align with vertical_axis_idx=1 in resolve_height_usage
    seg = seg.copy()
    seg[:, [1, 2]] = seg[:, [2, 1]]
    seg_path = os.path.join(tmp, "seg.ply")
    _write_ascii_ply(seg_path, seg)
    r1 = resolve_height_usage(seg_path, raw_height_m=2.0, height_input_source="system", scale_factor=1.0)
    check("system height + trunk segment -> dbh_only_fallback", r1["height_used"] == "dbh_only_fallback",
          f"({r1['height_used']})")
    check("trunk segment -> height_validated False", r1["height_validated"] is False)
    check("trunk segment -> height_for_formula None", r1["height_for_formula"] is None)
    check("trunk segment -> reason present", bool(r1["height_fallback_reason"]))

    # (b) system height + full ground-to-canopy cloud -> full_height validated
    full = _full_tree_cloud(height_m=8.0)
    # swap Y and Z to align with vertical_axis_idx=1 in resolve_height_usage
    full = full.copy()
    full[:, [1, 2]] = full[:, [2, 1]]
    full_path = os.path.join(tmp, "full.ply")
    _write_ascii_ply(full_path, full)
    r2 = resolve_height_usage(full_path, raw_height_m=8.0, height_input_source="system", scale_factor=1.0)
    check("system height + full tree -> full_height", r2["height_used"] == "full_height",
          f"({r2['height_used']})")
    check("full tree -> height_validated True", r2["height_validated"] is True)
    check("full tree -> height_for_formula = raw height", r2["height_for_formula"] == 8.0)
    check("full tree -> no fallback reason", r2["height_fallback_reason"] is None)

    # (b2) system height + missing point cloud file -> fallback with reason
    r3 = resolve_height_usage(os.path.join(tmp, "missing.ply"), raw_height_m=3.0,
                              height_input_source="system", scale_factor=1.0)
    check("missing point cloud -> dbh_only_fallback", r3["height_used"] == "dbh_only_fallback")
    check("missing point cloud -> validated False", r3["height_validated"] is False)

    # (c) manual height input (user) -> honoured, but explicitly un-validated
    r4 = resolve_height_usage(None, raw_height_m=30.0, height_input_source="manual", scale_factor=1.0)
    check("manual height -> user_manual_height (tidak dipaksa fallback)",
          r4["height_used"] == "user_manual_height", f"({r4['height_used']})")
    check("manual height -> height_for_formula = manual value (pakai height-based)",
          r4["height_for_formula"] == 30.0)
    check("manual height -> height_validated False", r4["height_validated"] is False)
    check("manual height -> reason about user input",
          "manual" in r4["height_validation_reason"] and "user" in r4["height_validation_reason"])

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("      Konsistensi: field response (height_used/total_height_used_m/"
          "segment_height_m/height_fallback_reason/height_validated/height_validation_reason) "
          "identik di semua endpoint karena dipakai via helper yang sama.")


# ─────────────────────────────────────────────────────────────────────────────
# FIX — 2D-clicks h_target clamp (batang terekam terlalu pendek utk breast height)
# Mirror guard extract_dbh_from_mast3r (dbh_extractor.py:457-459). Sebelum fix,
# h_target = P1·v + 1.3/scale jatuh DI ATAS seluruh point cloud kalau batang < 1.3 m,
# sehingga viewer 3D menggambar cylinder/ring di ruang kosong.
# ─────────────────────────────────────────────────────────────────────────────
def test_2d_clicks_short_trunk_clamp():
    print("\n=== FIX: 2D-clicks h_target clamp (short trunk < breast height) ===")
    from carbon.dbh_extractor import extract_dbh_with_2d_clicks
    import shutil

    tmp = tempfile.mkdtemp()
    try:
        # Case 1: trunk segment hanya 0.4 m (jauh lebih pendek dari breast height 1.3 m)
        short = _trunk_segment_cloud(radius=0.13, height_m=0.4, scale=1.0, seed=3)
        short_path = os.path.join(tmp, "short.ply")
        _write_ascii_ply(short_path, short)
        r = extract_dbh_with_2d_clicks(
            short_path, P1=np.array([0.0, 0.0, 0.0]),
            P2=np.array([0.0, 0.0, 0.4]), scale=1.0)
        g = r["geometry_3d"]
        h_min, h_max, h_t = g["h_min"], g["h_max"], g["h_target"]
        total_h = h_max - h_min
        check("short trunk -> h_target DI DALAM [h_min,h_max] (tidak melayang di atas)",
              (h_t >= h_min - 1e-6) and (h_t <= h_max + 1e-6),
              f"(h_target={h_t:.4f}, h_min={h_min:.4f}, h_max={h_max:.4f})")
        check("short trunk -> h_target = h_min + 0.30*total_h (clamp aktif)",
              abs(h_t - (h_min + 0.30 * total_h)) < 1e-6,
              f"(h_target={h_t:.4f}, expected={h_min + 0.30 * total_h:.4f})")
        check("short trunk -> clamp condition benar (1.3 m >= 90% total_h)",
              (1.3 / 1.0) >= total_h * 0.90, f"(1.3 >= {total_h * 0.90:.4f})")
        check("short trunk -> method pakai slice fit (bukan '2D fallback')",
              r["method"] == "Manual override 2D clicks", f"({r['method']})")
        check("short trunk -> DBH slice fit ~ diameter batang",
              abs(r["dbh_cm"] - (2 * 0.13) * 100.0) < 4.0,
              f"(dbh_cm={r['dbh_cm']:.2f})")

        # Case 2: trunk cukup tinggi (5 m) -> clamp TIDAK boleh mengubah h_target
        tall = _trunk_segment_cloud(radius=0.15, height_m=5.0, scale=1.0, seed=4)
        tall_path = os.path.join(tmp, "tall.ply")
        _write_ascii_ply(tall_path, tall)
        r2 = extract_dbh_with_2d_clicks(
            tall_path, P1=np.array([0.0, 0.0, 0.0]),
            P2=np.array([0.0, 0.0, 5.0]), scale=1.0)
        g2 = r2["geometry_3d"]
        check("tall trunk -> h_target TETAP 1.3 m di atas P1 (tidak di-clamp)",
              abs(g2["h_target"] - 1.3) < 1e-6,
              f"(h_target={g2['h_target']:.4f})")
        check("tall trunk -> h_target tetap di dalam batang",
              g2["h_target"] >= g2["h_min"] and g2["h_target"] <= g2["h_max"],
              f"(h_target={g2['h_target']:.4f}, h_min={g2['h_min']:.4f}, h_max={g2['h_max']:.4f})")
        check("tall trunk -> method slice fit",
              r2["method"] == "Manual override 2D clicks", f"({r2['method']})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_unguarded_estimate_carbon_calls():
    print("\n=== PRIORITY 2 (lanjutan): audit caller estimate_carbon/carbon ===")
    server_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
    with open(server_src, "r", encoding="utf-8") as fh:
        text = fh.read()

    # Any call that passes a system/raw height straight into the allometric model
    # (instead of the validated hinfo['height_for_formula']) would bypass the guard.
    forbidden = [
        'height_m=res_override["height_m"]',
        'height_m=dbh_result["height_m"]',
        'height_m=height_m',
    ]
    for pat in forbidden:
        check(f"server.py tidak memanggil estimate_carbon dgn '{pat}'", pat not in text)

    # every estimate_carbon call site must reference the validated height_for_formula
    idx = 0
    n_calls = 0
    while True:
        idx = text.find("estimate_carbon(", idx)
        if idx == -1:
            break
        n_calls += 1
        tail = text[idx:idx + 400]
        check(f"estimate_carbon call #{n_calls} memakai height_for_formula",
              ("height_m=hinfo[" in tail) or ("height_m=hinfo['" in tail),
              f"(call @ offset {idx})")
        idx += len("estimate_carbon(")
    check("menemukan >=1 estimate_carbon call", n_calls >= 1, f"({n_calls})")


def main():
    test_root_to_shoot_ratio()
    test_uncertainty()
    test_full_tree_height()
    test_auto_pose_scale()
    test_manual_endpoint_height_validation()
    test_2d_clicks_short_trunk_clamp()
    test_no_unguarded_estimate_carbon_calls()
    test_server_scale_loading()

    print(f"\n{'='*60}\nALL CHECKS PASSED ({len(PASS)} passed)\n{'='*60}")


if __name__ == "__main__":
    main()

