"""Bounded, anchor-referenced monocular AR tracking (Python 3.10+)."""
import os
import sys

# Set before importing cv2: MSMF hardware transforms can stall webcam startup.
# Preserve an explicit user override, and apply the same default to calibration.
if sys.platform == "win32":
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
