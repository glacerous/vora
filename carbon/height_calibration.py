import numpy as np
import logging

logger = logging.getLogger("HeightCalibration")

def detect_person_pose(frame_path):
    """
    Detects a person in the given frame and returns key pixel coordinates:
    - Head (represented by Nose landmark or eye/ear midpoints)
    - Foot (represented by average of Ankle landmarks)

    cv2 / mediapipe are imported lazily to keep this module importable and testable
    in environments without those heavy dependencies (the pure helpers below only
    need numpy).
    """
    import cv2
    import mediapipe as mp
    try:
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        # We need static_image_mode=True for single image inference
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # high complexity for better accuracy
            enable_segmentation=False,
            min_detection_confidence=0.5
        ) as pose:
            image = cv2.imread(frame_path)
            if image is None:
                logger.error(f"Failed to read image at: {frame_path}")
                return None
                
            h, w, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if not results.pose_landmarks:
                return None
                
            landmarks = results.pose_landmarks.landmark
            
            # Extract landmarks
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            # Confidence threshold for landmark visibility
            VISIBILITY_THRESHOLD = 0.5
            
            # Ensure head (nose) and at least one ankle is visible
            if nose.visibility < VISIBILITY_THRESHOLD:
                logger.info(f"Nose landmark visibility too low: {nose.visibility:.2f}")
                return None
                
            ankle_visible_count = 0
            ankle_x, ankle_y = 0.0, 0.0
            
            if left_ankle.visibility >= VISIBILITY_THRESHOLD:
                ankle_x += left_ankle.x
                ankle_y += left_ankle.y
                ankle_visible_count += 1
                
            if right_ankle.visibility >= VISIBILITY_THRESHOLD:
                ankle_x += right_ankle.x
                ankle_y += right_ankle.y
                ankle_visible_count += 1
                
            if ankle_visible_count == 0:
                logger.info("Neither left nor right ankle is visible enough.")
                return None
                
            # Average ankle position as foot
            foot_x = ankle_x / ankle_visible_count
            foot_y = ankle_y / ankle_visible_count
            avg_ankle_visibility = (left_ankle.visibility + right_ankle.visibility) / 2.0
            
            # Average confidence
            confidence = (nose.visibility + avg_ankle_visibility) / 2.0
            
            # Convert normalized coordinates to pixel coordinates
            head_px = (int(nose.x * w), int(nose.y * h))
            foot_px = (int(foot_x * w), int(foot_y * h))
            
            logger.info(f"Pose detected in {frame_path}: Head {head_px}, Foot {foot_px}, Confidence {confidence:.2f}")
            return {
                "head": head_px,
                "foot": foot_px,
                "confidence": float(confidence)
            }
            
    except Exception as e:
        logger.error(f"Error during pose detection: {e}")
        return None


def _find_person_scale_in_cloud(points, person_height_m: float, axis_idx: int = 2):
    """
    Heuristic: locate a person (a compact, tall vertical cluster near the ground)
    inside the reconstructed point cloud and derive the scale factor:

        scale_factor = real_person_height_m / person_extent_in_units

    Because the reconstruction is in arbitrary units, we find a horizontal "column"
    that contains a small vertical run of points near the bottom of the cloud whose
    vertical extent plausibly matches a standing human (rather than the full tree).

    Returns the scale_factor (meters per unit) or None if no candidate is found.
    """
    import numpy as np

    if points is None or len(points) < 30:
        return None

    z = np.asarray(points[:, axis_idx], dtype=float)
    if axis_idx == 1:
        z = -z
    z_min = float(z.min())
    z_max = float(z.max())
    total_h = z_max - z_min
    if total_h <= 0:
        return None

    proj_axes = [i for i in range(3) if i != axis_idx]
    x = np.asarray(points[:, proj_axes[0]], dtype=float)
    y = np.asarray(points[:, proj_axes[1]], dtype=float)

    grid = 30
    hist, xedges, yedges = np.histogram2d(x, y, bins=grid)

    best = None  # (count, extent_units)
    for ix in range(grid):
        for iy in range(grid):
            if hist[ix, iy] < 25:
                continue
            mask = (
                (x >= xedges[ix]) & (x < xedges[ix + 1])
                & (y >= yedges[iy]) & (y < yedges[iy + 1])
            )
            pts = points[mask]
            if len(pts) < 25:
                continue
            pz = np.asarray(pts[:, axis_idx], dtype=float)
            extent = float(pz.max() - pz.min())
            # Person's vertical extent should be a modest fraction of total cloud
            if not (0.04 * total_h < extent < 0.45 * total_h):
                continue
            # Person must stand near the ground (bottom 30% of the cloud)
            if (pz.min() - z_min) > 0.30 * total_h:
                continue
            if best is None or int(hist[ix, iy]) > best[0]:
                best = (int(hist[ix, iy]), extent)

    if best is None:
        return None
    _, extent_units = best
    if extent_units <= 0:
        return None
    return float(person_height_m / extent_units)


def auto_calibrate_scale_from_frames(frame_paths, points_3d=None, person_height_m: float = 1.65,
                                     vertical_axis_idx: int = 1, min_confidence: float = 0.6,
                                     max_frames: int = 8):
    """
    Attempts an implicit (automatic) scale calibration using a person visible in the
    frame(s). This lets the pipeline self-calibrate without the user running the
    `calibrate_scale.py` CLI.

    Steps:
      1. Detect a full-body person in the provided frames via MediaPipe pose.
      2. If a person is found with sufficient confidence, try to localise that person
         as a vertical cluster inside the reconstructed point cloud.
      3. If localisation succeeds, derive scale_factor = person_height_m / extent_units.

    Returns a dict with keys:
        detected, is_calibrated, source, scale_factor, reason
    or None if no frame could be parsed / pose failed.
    """
    import os

    if not frame_paths:
        return None

    frames = [p for p in frame_paths if os.path.exists(p)][:max_frames]
    if not frames:
        return None

    best_confidence = 0.0
    for p in frames:
        try:
            det = detect_person_pose(p)
        except Exception as e:
            logger.warning(f"[AUTO-CALIB] Pose detection failed for {p}: {e}")
            det = None
        if det and det.get("confidence", 0.0) > best_confidence:
            best_confidence = det["confidence"]

    if best_confidence < min_confidence:
        return {
            "detected": False,
            "is_calibrated": False,
            "source": "uncalibrated",
            "scale_factor": 1.0,
            "reason": f"tidak ada orang terdeteksi dengan confidence cukup (best={best_confidence:.2f} < {min_confidence})",
        }

    if points_3d is None or len(points_3d) == 0:
        return {
            "detected": True,
            "is_calibrated": False,
            "source": "uncalibrated",
            "scale_factor": 1.0,
            "reason": "orang terdeteksi di frame tetapi point cloud untuk kalibrasi tidak tersedia",
        }

    sf = _find_person_scale_in_cloud(points_3d, person_height_m, vertical_axis_idx)
    if sf is None or sf <= 0:
        return {
            "detected": True,
            "is_calibrated": False,
            "source": "uncalibrated",
            "scale_factor": 1.0,
            "reason": "orang terdeteksi di frame tetapi tidak ditemukan klaster orang yang valid di point cloud",
        }

    logger.info(
        f"[AUTO-CALIB] Auto pose calibration succeeded: scale_factor={sf:.6f} "
        f"(person {person_height_m}m, confidence={best_confidence:.2f})"
    )
    return {
        "detected": True,
        "is_calibrated": True,
        "source": "auto_pose",
        "scale_factor": sf,
        "reason": f"auto-kalibrasi via pose (tinggi asumsi {person_height_m}m, confidence={best_confidence:.2f})",
    }
