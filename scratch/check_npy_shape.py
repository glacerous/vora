import numpy as np
import os

def main():
    path = "c:\\codes\\3dtest\\output\\points3D_all.npy"
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return
    
    arr = np.load(path)
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"Itemsize: {arr.itemsize} bytes")
    print(f"Number of elements: {arr.size}")
    print(f"Total size in memory: {arr.nbytes / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
