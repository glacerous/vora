import cv2
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger("HeightCalibration")

def detect_person_pose(frame_path):
    """
    Detects a person in the given frame and returns key pixel coordinates:
    - Head (represented by Nose landmark or eye/ear midpoints)
    - Foot (represented by average of Ankle landmarks)
    
    Returns:
      A dict containing:
        - "head": (x_pixel, y_pixel)
        - "foot": (x_pixel, y_pixel)
        - "confidence": average visibility of head and foot landmarks
      Or None if no person with high confidence (especially full body) is detected.
    """
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
