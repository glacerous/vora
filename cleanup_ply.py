#!/usr/bin/env python3
"""
cleanup_ply.py — Statistical + opacity outlier removal for Gaussian Splat PLY files.

Removes two types of noise:
  1. Spatial outliers — Gaussians far from their neighbours (floaters in empty space)
  2. Low-opacity Gaussians — nearly-invisible splats that only add visual noise

All 62 Gaussian Splat properties (SH coefficients, opacity, scale, rotation)
are preserved for surviving Gaussians.

Usage:
    # Apply to existing scan (default params)
    py cleanup_ply.py

    # Custom thresholds
    py cleanup_ply.py --input output/result.ply --output output/result_clean.ply \
        --nb_neighbors 20 --std_ratio 2.0 --min_opacity -4.0

    # Preview-only (don't write output)
    py cleanup_ply.py --dry_run

    # Extra aggressive cleanup (removes more floaters)
    py cleanup_ply.py --std_ratio 1.5 --min_opacity -3.0
"""

import argparse
import os
import sys
import time


# ── PLY I/O helpers ───────────────────────────────────────────────────────────

def read_gaussian_ply(path: str):
    """
    Reads a Gaussian Splat PLY file into a numpy structured array.
    Returns (vertex_data, properties, num_vertices).
    properties is a list of (dtype_str, name) tuples matching the PLY header.
    """
    import numpy as np

    if not os.path.exists(path):
        raise FileNotFoundError(f"PLY not found: {path}")

    with open(path, "rb") as f:
        raw_props = []
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
                    raw_props.append((parts[1], parts[2]))  # (ply_type, name)
            elif line == "end_header":
                break

        if num_vertices <= 0:
            raise ValueError("No vertices found in PLY header.")
        if not is_binary:
            raise NotImplementedError("ASCII PLY not supported by this script (Gaussian Splat PLYs are always binary).")

        # Build numpy dtype
        dtype_map = []
        for p_type, p_name in raw_props:
            if p_type in ("float", "float32"):
                dtype_map.append((p_name, "<f4"))
            elif p_type in ("int", "int32", "uint"):
                dtype_map.append((p_name, "<i4"))
            elif p_type in ("uchar", "uint8"):
                dtype_map.append((p_name, "u1"))
            else:
                dtype_map.append((p_name, "<f4"))  # safe fallback

        vertex_data = np.fromfile(f, dtype=np.dtype(dtype_map), count=num_vertices)

    return vertex_data, raw_props, num_vertices


def write_gaussian_ply(path: str, vertex_data, raw_props):
    """
    Writes a filtered Gaussian Splat PLY back to disk, preserving all properties.
    """
    n = len(vertex_data)
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
    ]
    for p_type, p_name in raw_props:
        header_lines.append(f"property {p_type} {p_name}")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        vertex_data.tofile(f)


# ── Outlier removal passes ────────────────────────────────────────────────────

def remove_spatial_outliers_scipy(xyz, nb_neighbors: int, std_ratio: float):
    """
    Pure-numpy/scipy statistical outlier removal.
    For each point, compute mean distance to k nearest neighbours.
    Points whose mean_dist > global_mean + std_ratio * global_std are removed.
    """
    from scipy.spatial import KDTree
    import numpy as np

    print(f"  [Spatial] Building KD-tree for {len(xyz):,} points...")
    t0 = time.time()
    tree = KDTree(xyz)
    # Query k+1 because index 0 is the point itself
    distances, _ = tree.query(xyz, k=nb_neighbors + 1, workers=-1)
    mean_dists = distances[:, 1:].mean(axis=1)  # exclude self (col 0)

    global_mean = mean_dists.mean()
    global_std  = mean_dists.std()
    threshold   = global_mean + std_ratio * global_std

    inlier_mask = mean_dists <= threshold
    n_removed   = int((~inlier_mask).sum())
    print(f"  [Spatial] Done in {time.time()-t0:.1f}s | threshold={threshold:.6f} | removed {n_removed:,} points")
    return inlier_mask


def remove_low_opacity_gaussians(vertex_data, min_opacity_logit: float):
    """
    In Gaussian Splat PLYs, opacity is stored as a logit (inverse sigmoid).
    Values near 0 after sigmoid = nearly invisible Gaussians.
    min_opacity_logit: raw logit threshold (default -4.0 ≈ sigmoid≈0.018)
    """
    import numpy as np

    if "opacity" not in vertex_data.dtype.names:
        print("  [Opacity] 'opacity' property not found, skipping opacity filter.")
        return np.ones(len(vertex_data), dtype=bool)

    opacity_logits = vertex_data["opacity"]
    inlier_mask    = opacity_logits >= min_opacity_logit
    n_removed      = int((~inlier_mask).sum())
    # Show sigma-space stats
    sigmoid_thresh = 1.0 / (1.0 + float(__import__("math").exp(-min_opacity_logit)))
    print(f"  [Opacity] threshold={min_opacity_logit:.2f} (sigmoid={sigmoid_thresh:.4f}) | removed {n_removed:,} low-opacity Gaussians")
    return inlier_mask


def remove_large_gaussians(vertex_data, max_log_scale: float):
    """
    Gaussian scales are stored as log values in scale_0/1/2.
    Very large Gaussians (huge blobs covering huge areas) are usually floaters.
    max_log_scale: upper bound on max(scale_0, scale_1, scale_2).
    """
    import numpy as np

    scale_names = [n for n in vertex_data.dtype.names if n.startswith("scale_")]
    if not scale_names:
        print("  [Scale] No scale_* properties found, skipping scale filter.")
        return np.ones(len(vertex_data), dtype=bool)

    # Take the maximum log-scale across all axes per Gaussian
    scales = np.column_stack([vertex_data[n] for n in scale_names])
    max_scales = scales.max(axis=1)

    inlier_mask = max_scales <= max_log_scale
    n_removed   = int((~inlier_mask).sum())
    print(f"  [Scale]   max_log_scale={max_log_scale:.2f} | removed {n_removed:,} oversized Gaussians")
    return inlier_mask


# ── Main pipeline ─────────────────────────────────────────────────────────────

def cleanup(
    input_path: str,
    output_path: str,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    min_opacity: float = -4.0,
    max_log_scale: float = 2.0,
    skip_spatial: bool = False,
    skip_opacity: bool = False,
    skip_scale: bool = False,
    dry_run: bool = False,
):
    import numpy as np

    print(f"\nLoading: {input_path}")
    vertex_data, raw_props, n_total = read_gaussian_ply(input_path)
    print(f"  Total Gaussians (before): {n_total:,}")
    print(f"  PLY size: {os.path.getsize(input_path) / 1024 / 1024:.1f} MB\n")

    # Extract xyz positions once
    xyz = np.column_stack((vertex_data["x"], vertex_data["y"], vertex_data["z"])).astype(np.float64)

    # Bounding box info (useful for sanity check)
    bb_min = xyz.min(axis=0)
    bb_max = xyz.max(axis=0)
    bb_extent = bb_max - bb_min
    print(f"  Bounding box extent: X={bb_extent[0]:.3f}  Y={bb_extent[1]:.3f}  Z={bb_extent[2]:.3f} units\n")

    # Combined mask — start with all True
    combined_mask = np.ones(n_total, dtype=bool)

    # Pass 1: Spatial outlier removal
    if not skip_spatial:
        print("Pass 1: Spatial outlier removal (scipy KDTree)")
        spatial_mask = remove_spatial_outliers_scipy(xyz, nb_neighbors, std_ratio)
        combined_mask &= spatial_mask
        print(f"  Remaining after pass 1: {combined_mask.sum():,}\n")

    # Pass 2: Low opacity removal
    if not skip_opacity:
        print("Pass 2: Low-opacity Gaussian removal")
        opacity_mask = remove_low_opacity_gaussians(vertex_data, min_opacity)
        combined_mask &= opacity_mask
        print(f"  Remaining after pass 2: {combined_mask.sum():,}\n")

    # Pass 3: Oversized Gaussian removal
    if not skip_scale:
        print("Pass 3: Oversized Gaussian removal (log-scale filter)")
        scale_mask = remove_large_gaussians(vertex_data, max_log_scale)
        combined_mask &= scale_mask
        print(f"  Remaining after pass 3: {combined_mask.sum():,}\n")

    # Stats
    n_kept    = int(combined_mask.sum())
    n_removed = n_total - n_kept
    pct       = n_removed / n_total * 100

    print("=" * 55)
    print(f"  Input  : {n_total:,} Gaussians")
    print(f"  Output : {n_kept:,} Gaussians")
    print(f"  Removed: {n_removed:,} ({pct:.1f}%)")
    print("=" * 55)

    if dry_run:
        print("\n[dry_run=True] No file written.")
        return n_removed, pct, n_kept

    # Write filtered PLY
    filtered = vertex_data[combined_mask]
    print(f"\nWriting: {output_path}")
    write_gaussian_ply(output_path, filtered, raw_props)
    out_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Done. Output size: {out_mb:.1f} MB")

    return n_removed, pct, n_kept


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Remove outlier/floater Gaussians from a 3D Gaussian Splat PLY file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",  default="output/result.ply",       help="Input PLY (default: output/result.ply)")
    parser.add_argument("--output", default=None,                        help="Output PLY (default: overwrites input with .bak backup)")
    parser.add_argument("--nb_neighbors",  type=int,   default=20,      help="KNN count for spatial outlier detection (default: 20)")
    parser.add_argument("--std_ratio",     type=float, default=2.0,     help="Std-dev multiplier threshold (default: 2.0, lower=more aggressive)")
    parser.add_argument("--min_opacity",   type=float, default=-4.0,    help="Min logit opacity to keep (default: -4.0 ≈ sigmoid 0.018)")
    parser.add_argument("--max_log_scale", type=float, default=-1.0,    help="Max log-scale to keep (default: -1.0 = exp ~0.37 units; set higher to be more lenient)")
    parser.add_argument("--scale_pct",     type=float, default=99.5,    help="Alternatively, keep only Gaussians with log-scale below this percentile (default: 99.5)")
    parser.add_argument("--use_scale_pct", action="store_true",         help="Use percentile-based scale filter instead of fixed max_log_scale")
    parser.add_argument("--skip_spatial",  action="store_true",         help="Skip spatial KNN outlier removal pass")
    parser.add_argument("--skip_opacity",  action="store_true",         help="Skip low-opacity removal pass")
    parser.add_argument("--skip_scale",    action="store_true",         help="Skip oversized-scale removal pass")
    parser.add_argument("--dry_run",       action="store_true",         help="Print stats but do not write output")
    parser.add_argument("--no_backup",     action="store_true",         help="Skip creating .bak backup when overwriting input")
    args = parser.parse_args()

    # Resolve paths
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), input_path)

    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_path)
    else:
        # Default: overwrite input (with .bak backup)
        output_path = input_path
        if not args.dry_run and not args.no_backup:
            backup = input_path + ".bak"
            import shutil
            shutil.copy2(input_path, backup)
            print(f"[INFO] Backup created: {backup}")

    cleanup(
        input_path=input_path,
        output_path=output_path,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        min_opacity=args.min_opacity,
        max_log_scale=args.max_log_scale,
        skip_spatial=args.skip_spatial,
        skip_opacity=args.skip_opacity,
        skip_scale=args.skip_scale,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
