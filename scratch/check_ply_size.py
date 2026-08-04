import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from carbon.dbh_extractor import parse_ply_points

def main():
    path = "c:\\codes\\3dtest\\output\\points3d.ply"
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return
        
    pts = parse_ply_points(path)
    print(f"Loaded points type: {type(pts)}")
    print(f"Shape: {pts.shape}")
    print(f"Dtype: {pts.dtype}")
    print(f"Total size in memory: {pts.nbytes / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
