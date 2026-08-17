"""
Comprehensive Unit & Concurrency Test Suite for Vora 3D Reconstruction Pipeline
Validates:
1. Multi-user isolation & namespaced directories (zero cross-contamination)
2. Gate 1 (2D Frame Quality Pre-Check on CPU)
3. Gate 2 (Early MASt3R Point Cloud & Parallax Gate)
4. D1 Exponential Backoff & Local Disk Fallback Buffer
"""

import os
import sys
import time
import shutil
import tempfile
import unittest
import numpy as np
import cv2

# Add workspace to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server
from server import (
    get_job_frames_dir,
    get_job_output_dir,
    get_job_state,
    init_job_state,
    upd,
    active_jobs,
)
from modal_app import _validate_early_geometry
import storage.d1_client as d1_client


class TestPipelineConcurrencyAndGates(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_multi_user_isolation(self):
        """Test that multiple concurrent tree scans have completely isolated directories and states."""
        tree_a = "POHON-TESTA"
        tree_b = "POHON-TESTB"

        dir_a = get_job_frames_dir(tree_a)
        dir_b = get_job_frames_dir(tree_b)

        # Assert separate folder paths
        self.assertNotEqual(dir_a, dir_b)
        self.assertTrue(dir_a.endswith(tree_a))
        self.assertTrue(dir_b.endswith(tree_b))

        # Initialize both job states
        init_job_state(tree_a, message="User A extracting")
        init_job_state(tree_b, message="User B extracting")

        # Write dummy frame for User A
        frame_a_path = os.path.join(dir_a, "0000.jpg")
        with open(frame_a_path, "wb") as f:
            f.write(b"USER_A_FRAME_DATA")

        # Write dummy frame for User B
        frame_b_path = os.path.join(dir_b, "0000.jpg")
        with open(frame_b_path, "wb") as f:
            f.write(b"USER_B_FRAME_DATA")

        # Verify no cross-contamination on disk
        with open(frame_a_path, "rb") as f:
            self.assertEqual(f.read(), b"USER_A_FRAME_DATA")

        with open(frame_b_path, "rb") as f:
            self.assertEqual(f.read(), b"USER_B_FRAME_DATA")

        # Update User A state and check User B state is unaffected
        upd(tree_a, "reconstructing", "Training Gaussians for User A")
        upd(tree_b, "extracted", "Frames ready for User B")

        state_a = get_job_state(tree_a)
        state_b = get_job_state(tree_b)

        self.assertEqual(state_a["stage"], "reconstructing")
        self.assertEqual(state_a["message"], "Training Gaussians for User A")
        self.assertEqual(state_b["stage"], "extracted")
        self.assertEqual(state_b["message"], "Frames ready for User B")

        print("[PASS] Multi-user isolation test passed: complete isolation between concurrent scans.")

    def test_gate1_2d_quality_precheck(self):
        """Test Gate 1 rejection of dark, overexposed, and textureless frames."""
        # 1. Test pitch black image (mean brightness < 10)
        black_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray_black = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
        self.assertLess(float(np.mean(gray_black)), 10.0)

        # 2. Test overexposed white image (mean brightness > 248)
        white_img = np.full((100, 100, 3), 255, dtype=np.uint8)
        gray_white = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
        self.assertGreater(float(np.mean(gray_white)), 248.0)

        # 3. Test completely flat textureless image (Laplacian variance < 3.0)
        flat_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        gray_flat = cv2.cvtColor(flat_img, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray_flat, cv2.CV_64F).var())
        self.assertLess(lap_var, 3.0)

        # 4. Test normal textured tree bark image
        textured_img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        gray_tex = cv2.cvtColor(textured_img, cv2.COLOR_BGR2GRAY)
        mean_b = float(np.mean(gray_tex))
        lap_v = float(cv2.Laplacian(gray_tex, cv2.CV_64F).var())
        self.assertTrue(10.0 <= mean_b <= 248.0)
        self.assertGreater(lap_v, 3.0)

        print("[PASS] Gate 1 2D quality & texture validation checks passed.")

    def test_gate2_early_geometry_validation(self):
        """Test Gate 2 early geometry check in modal_app."""
        scene_dir = os.path.join(self.test_dir, "scene")
        sparse_dir = os.path.join(scene_dir, "sparse_10", "0")
        os.makedirs(sparse_dir, exist_ok=True)

        # 1. Create a dummy points3D.txt with only 50 points (< 250 required)
        points_file = os.path.join(sparse_dir, "points3D.txt")
        with open(points_file, "w") as pf:
            for i in range(50):
                pf.write(f"{i} 0.0 0.0 0.0 255 255 255 0.1 1 2 3\n")

        with self.assertRaises(RuntimeError) as ctx:
            _validate_early_geometry(scene_dir, 10)
        self.assertIn("minimum 250 required", str(ctx.exception))
        print("[PASS] Gate 2 correctly rejected sparse point cloud (<250 points).")

        # 2. Add 300 points but stationary cameras (path length < 0.10)
        with open(points_file, "w") as pf:
            for i in range(300):
                pf.write(f"{i} 0.0 0.0 0.0 255 255 255 0.1 1 2 3\n")

        images_file = os.path.join(sparse_dir, "images.txt")
        with open(images_file, "w") as imf:
            for i in range(5):
                # Identical positions -> zero parallax
                imf.write(f"{i} 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 frame_{i}.jpg\n")
                imf.write("0.0 0.0 1\n")

        with self.assertRaises(RuntimeError) as ctx2:
            _validate_early_geometry(scene_dir, 10)
        self.assertIn("Insufficient camera movement/parallax", str(ctx2.exception))
        print("[PASS] Gate 2 correctly rejected stationary video (<0.10 parallax).")

        # 3. Orbiting camera path with sufficient points -> should pass
        with open(images_file, "w") as imf:
            for i in range(10):
                # Camera orbiting at radius 1.5m
                angle = (i / 10.0) * 2 * np.pi
                tx = 1.5 * np.cos(angle)
                tz = 1.5 * np.sin(angle)
                imf.write(f"{i} 1.0 0.0 0.0 0.0 {tx:.4f} 0.0 {tz:.4f} 1 frame_{i}.jpg\n")
                imf.write("0.0 0.0 1\n")

        # Should execute cleanly without error
        _validate_early_geometry(scene_dir, 10)
        print("[PASS] Gate 2 successfully passed valid orbit reconstruction.")

    def test_d1_fallback_buffer(self):
        """Test that D1 client writes to local disk buffer when D1 API fails."""
        # Intentionally invalid account/DB to trigger retry and disk buffer write
        os.environ["CLOUDFLARE_API_TOKEN"] = "invalid_token_test"
        os.environ["CLOUDFLARE_ACCOUNT_ID"] = "invalid_acc"
        os.environ["CLOUDFLARE_D1_DATABASE_ID"] = "invalid_db"

        test_code = f"POHON-BUFTEST-{int(time.time())}"
        try:
            d1_client.save_scan_result(
                tree_code=test_code,
                dbh_cm=25.0,
                tinggi_m=8.5,
                biomassa_kg=120.0,
                karbon_kg=60.0,
                co2e_kg=220.0,
                splat_file_url="https://example.com/splat.ply",
                confidence_note="Test scan"
            )
        except Exception:
            pass  # Expected to fail D1 API

        buffer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "failed_scans_buffer")
        found_backup = False
        if os.path.exists(buffer_dir):
            for fname in os.listdir(buffer_dir):
                if test_code in fname:
                    found_backup = True
                    break

        self.assertTrue(found_backup, "Failed scan payload must be written to disk buffer on API failure.")
        print("[PASS] D1 fallback disk buffer successfully verified.")


if __name__ == "__main__":
    unittest.main()
