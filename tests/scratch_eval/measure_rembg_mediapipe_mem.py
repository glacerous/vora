import os
import sys
import gc
import subprocess

def get_mem_usage_mb():
    pid = os.getpid()
    if sys.platform == 'win32':
        try:
            cmd = f'powershell -NoProfile -Command "(Get-Process -Id {pid}).WorkingSet / 1MB"'
            output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
            return float(output)
        except Exception:
            return 0.0
    return 0.0

def main():
    print(f"1. Baseline memory: {get_mem_usage_mb():.2f} MB")
    
    # Import rembg
    print("\n2. Importing rembg...")
    mem_before_rembg_import = get_mem_usage_mb()
    try:
        from rembg import remove
        from PIL import Image
        import io
        mem_after_rembg_import = get_mem_usage_mb()
        print(f"Memory after importing rembg: {mem_after_rembg_import:.2f} MB (increase: {mem_after_rembg_import - mem_before_rembg_import:.2f} MB)")
        
        # Run rembg on dummy image
        print("Running rembg.remove on a dummy image (128x128)...")
        mem_before_rembg_run = get_mem_usage_mb()
        dummy_img = Image.new("RGB", (128, 128), (255, 255, 255))
        _ = remove(dummy_img)
        mem_after_rembg_run = get_mem_usage_mb()
        print(f"Memory after running rembg: {mem_after_rembg_run:.2f} MB (increase: {mem_after_rembg_run - mem_before_rembg_run:.2f} MB)")
    except Exception as e:
        print(f"Failed to import/run rembg: {e}")
        
    # Free rembg memory if possible
    gc.collect()
    
    # Import mediapipe
    print("\n3. Importing mediapipe...")
    mem_before_mp_import = get_mem_usage_mb()
    try:
        import mediapipe as mp
        import cv2
        mem_after_mp_import = get_mem_usage_mb()
        print(f"Memory after importing mediapipe: {mem_after_mp_import:.2f} MB (increase: {mem_after_mp_import - mem_before_mp_import:.2f} MB)")
        
        # Initialize and run mediapipe Pose
        print("Initializing MediaPipe Pose (model_complexity=2)...")
        mem_before_mp_pose = get_mem_usage_mb()
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        ) as pose:
            # process dummy image
            dummy_cv = cv2.imread(os.path.join("test_images", "0000.jpg")) if os.path.exists(os.path.join("test_images", "0000.jpg")) else None
            if dummy_cv is None:
                import numpy as np
                dummy_cv = np.zeros((128, 128, 3), dtype=np.uint8)
            
            dummy_rgb = cv2.cvtColor(dummy_cv, cv2.COLOR_BGR2RGB)
            _ = pose.process(dummy_rgb)
            mem_after_mp_pose = get_mem_usage_mb()
            print(f"Memory after running MediaPipe: {mem_after_mp_pose:.2f} MB (increase: {mem_after_mp_pose - mem_before_mp_pose:.2f} MB)")
    except Exception as e:
        print(f"Failed to import/run mediapipe: {e}")

if __name__ == "__main__":
    main()
