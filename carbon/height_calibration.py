"""
carbon/height_calibration.py

MediaPipe Pose calibration removed — 2026-08

The auto-calibration approach using human body pose detection was removed because:
1. Tree scans almost never include a full-body person in frame.
2. Even when detected, localising a person-cluster in the point cloud was
   unreliable (grid histogram heuristic with strict geometry constraints).

Scale calibration is now handled by two superior methods:

  1. estimated_geometric_prior (Phase 1, active):
     Reads COLMAP camera centres from MASt3R's init_geo.py sparse output
     (sparse_N/images.bin). Mean camera spacing is used to verify the
     coordinate system is in metres. Accuracy: ~5-9% relative error.
     Implemented in modal_app.py :: _derive_mast3r_scale_prior().

  2. arcore_vio (Phase 2, mobile):
     ARCore / ARKit camera-pose sidecar (poses.json) recorded during video
     capture. Scale is derived from the metric camera-path-length ratio.
     Accuracy: ~1-3% relative error. Implemented in the vora-mobile
     VoraArModule native module.

This file is kept as an empty stub to avoid ImportError from any remaining
references to the module during transition.
"""
