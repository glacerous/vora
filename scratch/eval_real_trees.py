import os
import sys
import numpy as np

sys.path.insert(0, r"c:\codes\3dtest")
from carbon.dbh_extractor import parse_ply_points, extract_dbh_from_mast3r, extract_dbh_with_2d_clicks

# =============================================================================
# WARNING / DISCLAIMER (AUDIT 2026-08-16):
# Nilai yang tertera di bawah sebagai "Ground Truth DBH" (7.56 cm, 129.89 cm)
# BUKAN merupakan pengukuran fisik independen di lapangan (ground truth fisik).
# Nilai-nilai ini adalah rekaman baseline numerik dari run algoritma/D1 sebelumnya
# (regression baseline) untuk menguji stabilitas kode, BUKAN bukti akurasi fisik.
# JANGAN gunakan nilai di file ini sebagai ground truth eksternal tanpa pita ukur fisik.
# =============================================================================

def evaluate_all_real_trees():
    print("=" * 80)
    print("REGRESSION EVALUATION (BASELINE RECORDED METRICS — NOT PHYSICAL GROUND TRUTH)")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────────
    # SCAN A: TEST-REGRESSION-TREE (output/points3d.ply)
    # Recorded MASt3R / D1 baseline: DBH = 7.56 cm, Dir = [0.095, -0.976, 0.195]
    # ─────────────────────────────────────────────────────────────────────────
    ply_a = r"c:\codes\3dtest\output\points3d.ply"
    p1_a = np.array([-0.01, 0.40, 1.22]) # base
    p2_a = np.array([0.05, -0.15, 1.25]) # top
    res_a = extract_dbh_with_2d_clicks(ply_a, p1_a, p2_a, scale=1.0)
    
    print("\n--- SCAN A: TEST-REGRESSION-TREE (output/points3d.ply) ---")
    print(f"Regression Baseline DBH: {res_a['dbh_cm']:.2f} cm (Ref: 7.56 cm, Delta: {abs(res_a['dbh_cm'] - 7.56):.2f} cm)")
    print(f"Baseline Axis:           [0.0953, -0.9761, 0.1953]")
    print(f"Refined Axis:            [{res_a['geometry_3d']['dir_x']}, {res_a['geometry_3d']['dir_y']}, {res_a['geometry_3d']['dir_z']}]")
    print(f"Pass 1->2 Delta:         {res_a['geometry_3d']['pca_convergence_delta_deg']:.2f}°")
    print(f"Inlier Ratio:            {res_a['inlier_ratio']:.2%}")

    # ─────────────────────────────────────────────────────────────────────────
    # SCAN B: TEST-REGRESSION-TREE High-Res (output/points3d_highres.ply)
    # Recorded baseline: DBH = 7.56 cm
    # ─────────────────────────────────────────────────────────────────────────
    ply_b = r"c:\codes\3dtest\output\points3d_highres.ply"
    res_b = extract_dbh_with_2d_clicks(ply_b, p1_a, p2_a, scale=1.0)
    print("\n--- SCAN B: TEST-REGRESSION-TREE High-Res (points3d_highres.ply) ---")
    print(f"Regression Baseline DBH: {res_b['dbh_cm']:.2f} cm (Ref: 7.56 cm, Delta: {abs(res_b['dbh_cm'] - 7.56):.2f} cm)")
    print(f"Refined Axis:            [{res_b['geometry_3d']['dir_x']}, {res_b['geometry_3d']['dir_y']}, {res_b['geometry_3d']['dir_z']}]")
    print(f"Pass 1->2 Delta:         {res_b['geometry_3d']['pca_convergence_delta_deg']:.2f}°")
    print(f"Inlier Ratio:            {res_b['inlier_ratio']:.2%}")

    # ─────────────────────────────────────────────────────────────────────────
    # SCAN C: POHON-2576 (uploads/recalculates/POHON-2576_1785911535_points3d.ply)
    # Recorded baseline: DBH = 129.89 cm (large trunk with scale=1.0), Axis along X: [0.996, -0.027, -0.082]
    # ─────────────────────────────────────────────────────────────────────────
    ply_c = r"c:\codes\3dtest\uploads\recalculates\POHON-2576_1785911535_points3d.ply"
    p1_c = np.array([-1.20, 0.05, 0.65])
    p2_c = np.array([0.40, 0.02, 0.65])
    res_c = extract_dbh_with_2d_clicks(ply_c, p1_c, p2_c, scale=1.0)
    print("\n--- SCAN C: POHON-2576 Large Tree (POHON-2576_..._points3d.ply) ---")
    print(f"Regression Baseline DBH: {res_c['dbh_cm']:.2f} cm (Ref: 129.89 cm)")
    print(f"Refined Axis:            [{res_c['geometry_3d']['dir_x']}, {res_c['geometry_3d']['dir_y']}, {res_c['geometry_3d']['dir_z']}]")
    print(f"Pass 1->2 Delta:         {res_c['geometry_3d']['pca_convergence_delta_deg']:.2f}°")
    print(f"Inlier Ratio:            {res_c['inlier_ratio']:.2%}")

if __name__ == "__main__":
    evaluate_all_real_trees()
