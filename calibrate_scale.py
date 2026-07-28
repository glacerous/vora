#!/usr/bin/env python3
"""
calibrate_scale.py — Scale calibration tool for Vora 3D reconstruction pipeline.

Usage:
    python calibrate_scale.py \\
        --ply output/result.ply \\
        --point1 x1,y1,z1 \\
        --point2 x2,y2,z2 \\
        --real_distance_m 1.23 \\
        [--scan_id POHON-2854]

How to get point coordinates:
    Option A (Open3D viewer — recommended):
        python calibrate_scale.py --pick_points --ply output/result.ply
        Then click two points in the viewer window and copy the printed coordinates.

    Option B (manual):
        Open the PLY in MeshLab or CloudCompare, use "Point Picking" to read two coords.

    Option C (nearest-point search):
        python calibrate_scale.py --find_nearest --ply output/result.ply \\
            --point1 x,y,z --point2 x,y,z
        Finds the actual closest PLY point to each given approximate coordinate.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime


# ── PLY point loader (mirrors dbh_extractor.parse_ply_points) ────────────────

def parse_ply_points(ply_path: str):
    """Returns (N, 3) float32 numpy array of x, y, z coordinates."""
    import numpy as np

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    with open(ply_path, "rb") as f:
        properties = []
        num_vertices = 0
        is_binary = False

        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("format binary_little_endian"):
                is_binary = True
            elif line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3:
                    properties.append((parts[1], parts[2]))
            elif line == "end_header":
                break

        if num_vertices <= 0:
            raise ValueError("No vertices found in PLY file header.")

        if is_binary:
            dtype_map = []
            for p_type, p_name in properties:
                if p_type in ("float", "float32"):
                    dtype_map.append((p_name, "<f4"))
                elif p_type in ("int", "int32", "uint"):
                    dtype_map.append((p_name, "<i4"))
                elif p_type in ("uchar", "uint8"):
                    dtype_map.append((p_name, "u1"))
                else:
                    dtype_map.append((p_name, "<f4"))

            vertex_data = np.fromfile(f, dtype=np.dtype(dtype_map), count=num_vertices)
            points = np.column_stack((vertex_data["x"], vertex_data["y"], vertex_data["z"]))
        else:
            pts = []
            for _ in range(num_vertices):
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line:
                    p = line.split()
                    pts.append([float(p[0]), float(p[1]), float(p[2])])
            points = np.array(pts, dtype=np.float32)

    return points


# ── Calibration helpers ───────────────────────────────────────────────────────

def euclidean_3d(p1, p2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def find_nearest_point(points, query):
    """Returns the closest point in *points* (numpy array) to *query* (list/tuple)."""
    import numpy as np
    q = np.array(query, dtype=np.float32)
    dists = np.linalg.norm(points - q, axis=1)
    idx = int(np.argmin(dists))
    return points[idx].tolist(), float(dists[idx])


def calibrate(p1, p2, real_distance_m: float):
    """Compute scale_factor from two PLY-space points and a known metric distance."""
    ply_distance = euclidean_3d(p1, p2)
    if ply_distance == 0:
        raise ValueError("The two points are identical — distance in PLY space is 0.")
    scale_factor = real_distance_m / ply_distance
    return scale_factor, ply_distance


# ── Calibration JSON I/O ──────────────────────────────────────────────────────

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")


def load_calibrations() -> dict:
    """Load the full calibration registry (dict keyed by scan_id, plus 'default')."""
    if os.path.exists(CALIB_PATH):
        try:
            with open(CALIB_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read existing calibration.json: {e}. Starting fresh.")
    return {}


def save_calibration(scan_id: str, scale_factor: float, p1, p2, real_distance_m: float):
    """
    Save or update calibration entry for *scan_id* in calibration.json.
    If scan_id is 'default', it updates the global fallback.
    """
    registry = load_calibrations()

    entry = {
        "scale_factor": scale_factor,
        "reference_points": [list(p1), list(p2)],
        "reference_distance_m": real_distance_m,
            "calibrated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scan_id": scan_id,
    }

    registry[scan_id] = entry

    with open(CALIB_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"[OK] Saved calibration for scan_id='{scan_id}' -> {CALIB_PATH}")
    return entry


# ── Interactive point picker (requires Open3D) ────────────────────────────────

def pick_points_interactive(ply_path: str):
    """Open an Open3D visualizer to pick two reference points."""
    try:
        import open3d as o3d
    except ImportError:
        print("[ERROR] open3d is not installed. Install it with: pip install open3d")
        sys.exit(1)

    print("\n[INSTRUCTIONS] Open3D Point Picker:")
    print("  1. Press [Shift + Left Click] to pick a point.")
    print("  2. Pick exactly 2 points: reference start and end.")
    print("  3. Press [Q] or close the window when done.\n")

    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        print("[ERROR] PLY file is empty or could not be read by Open3D.")
        sys.exit(1)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick 2 reference points (Shift+Click)", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()
    if len(picked) < 2:
        print(f"[ERROR] Need exactly 2 picked points, but got {len(picked)}.")
        sys.exit(1)

    pts = [list(pcd.points[i]) for i in picked[:2]]
    print(f"\n[Picked Point 1] {pts[0]}")
    print(f"[Picked Point 2] {pts[1]}")
    print(f"\nCopy these for --point1 / --point2 if you want to re-run without the viewer.")
    return pts[0], pts[1]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate scale_factor for Vora 3D PLY reconstruction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input modes
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pick_points",
        action="store_true",
        help="Open an interactive Open3D viewer to pick two reference points.",
    )
    mode.add_argument(
        "--find_nearest",
        action="store_true",
        help=(
            "Snap --point1 / --point2 to the nearest actual vertex in the PLY "
            "(useful when you have approximate coordinates from another tool)."
        ),
    )

    parser.add_argument("--ply", default="output/result.ply", help="Path to the PLY file (default: output/result.ply)")
    parser.add_argument("--point1", help="Point A in PLY space as 'x,y,z'")
    parser.add_argument("--point2", help="Point B in PLY space as 'x,y,z'")
    parser.add_argument(
        "--real_distance_m",
        type=float,
        help="Known real-world distance between the two points in METERS.",
    )
    parser.add_argument(
        "--scan_id",
        default="default",
        help=(
            "Scan identifier to tag this calibration (e.g. POHON-2854). "
            "Use 'default' for a global fallback that applies to all uncalibrated scans. "
            "(default: 'default')"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_calibrations",
        help="List all stored calibrations and exit.",
    )

    args = parser.parse_args()

    # ── List mode
    if args.list_calibrations:
        registry = load_calibrations()
        if not registry:
            print("No calibrations found in calibration.json.")
        else:
            print(f"\nCalibrations stored in {CALIB_PATH}:\n")
            for sid, entry in registry.items():
                print(f"  scan_id : {sid}")
                print(f"    scale_factor       : {entry['scale_factor']:.8f}")
                print(f"    reference_distance : {entry['reference_distance_m']} m")
                print(f"    calibrated_at      : {entry.get('calibrated_at', 'unknown')}")
                print()
        return

    # ── Resolve PLY path
    ply_path = args.ply
    if not os.path.isabs(ply_path):
        ply_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ply_path)

    # ── Point picking mode
    if args.pick_points:
        if not args.real_distance_m:
            parser.error("--real_distance_m is required when using --pick_points.")
        p1, p2 = pick_points_interactive(ply_path)

    # ── Manual / find_nearest mode
    else:
        if not args.point1 or not args.point2:
            parser.error("--point1 and --point2 are required (or use --pick_points).")
        if not args.real_distance_m:
            parser.error("--real_distance_m is required.")

        try:
            p1 = [float(x) for x in args.point1.split(",")]
            p2 = [float(x) for x in args.point2.split(",")]
        except ValueError:
            parser.error("--point1 and --point2 must be comma-separated floats, e.g. '0.1,-0.3,0.5'")

        if args.find_nearest:
            print(f"[INFO] Loading PLY to snap points to nearest vertices…")
            points = parse_ply_points(ply_path)
            p1_snapped, d1 = find_nearest_point(points, p1)
            p2_snapped, d2 = find_nearest_point(points, p2)
            print(f"[Snap] Point 1: {p1} → nearest vertex {p1_snapped} (dist={d1:.6f} units)")
            print(f"[Snap] Point 2: {p2} → nearest vertex {p2_snapped} (dist={d2:.6f} units)")
            p1, p2 = p1_snapped, p2_snapped

    # ── Compute and save
    real_distance_m = args.real_distance_m
    scale_factor, ply_dist = calibrate(p1, p2, real_distance_m)

    print("\n" + "=" * 55)
    print("  Scale Calibration Result")
    print("=" * 55)
    print(f"  PLY space distance  : {ply_dist:.8f} units")
    print(f"  Real-world distance : {real_distance_m:.4f} m")
    print(f"  scale_factor        : {scale_factor:.8f}")
    print(f"  (1 PLY unit ~= {scale_factor * 100:.2f} cm in real world)")
    print(f"  scan_id             : {args.scan_id}")
    print("=" * 55 + "\n")

    entry = save_calibration(args.scan_id, scale_factor, p1, p2, real_distance_m)

    print("Verification:")
    print(f"  To confirm: load the PLY and check that measured objects match expected real dimensions.")
    print(f"  Run with '--list' to view all stored calibrations.\n")


if __name__ == "__main__":
    main()
