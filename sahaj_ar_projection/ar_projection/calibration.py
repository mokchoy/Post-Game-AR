"""Shared capture settings; checkerboard calibration necessarily uses raw images."""
import argparse
from pathlib import Path
import os
import tempfile
import cv2
import numpy as np
from .camera import backend
from .config import DEFAULT_PATH, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_PATH))
    parser.add_argument("--camera", type=int)
    parser.add_argument("--backend")
    parser.add_argument("--columns", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--square-size", type=float)
    parser.add_argument("--views", type=int)
    parser.add_argument("--output")
    parser.add_argument("--replace-calibration", action="store_true")
    args = parser.parse_args()
    mapping = {"camera": "camera.index", "backend": "camera.backend", "columns": "calibration.columns", "rows": "calibration.rows",
               "square_size": "calibration.square_size_m", "views": "calibration.views", "output": "camera.calibration_path"}
    overrides = {label: getattr(args, flag) for flag, label in mapping.items() if getattr(args, flag) is not None}
    config = load_config(args.config, overrides)
    if config.camera.source_type != "camera":
        raise ValueError("calibration requires camera.source_type=camera")
    target = Path(config.camera.calibration_path)
    if target.exists() and not args.replace_calibration:
        raise ValueError(f"{target} exists; use --replace-calibration to replace explicitly")
    c = config.calibration
    cap = cv2.VideoCapture(config.camera.index, backend(config.camera.backend))
    object_points, image_points = [], []
    image_size = None
    try:
        if not cap.isOpened():
            raise ValueError("cannot open calibration camera")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
        pattern = (c.columns, c.rows)
        template = np.zeros((c.rows*c.columns, 3), np.float32)
        template[:, :2] = np.mgrid[0:c.columns, 0:c.rows].T.reshape(-1, 2)*c.square_size_m
        print("Varied board positions/tilts: SPACE capture, Q/Esc finish. Raw images; runtime undistortion uses the resulting calibration.")
        while len(object_points) < c.views:
            ok, frame = cap.read()
            if not ok:
                raise ValueError("calibration capture returned no frame")
            size = (frame.shape[1], frame.shape[0])
            if image_size is not None and image_size != size:
                raise ValueError("camera dimensions changed during calibration")
            image_size = size
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern)
            view = frame.copy()
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .001))
                cv2.drawChessboardCorners(view, pattern, corners, found)
            cv2.putText(view, f"Views {len(object_points)}/{c.views} | SPACE capture | Q finish", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            cv2.imshow("Camera calibration", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and found:
                object_points.append(template.copy())
                image_points.append(corners.copy())
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if len(object_points) < 8:
        raise ValueError(f"need >=8 usable views; captured {len(object_points)}")
    rms, K, d, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    if not np.isfinite(K).all() or not np.isfinite(d).all() or not np.isfinite(rms):
        raise ValueError("nonfinite calibration result")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".calibration-", suffix=".npz", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, camera_matrix=K, dist_coeffs=d, image_size=np.array(image_size, np.int32))
            f.flush()
            os.fsync(f.fileno())
        if args.replace_calibration:
            os.replace(name, target)
        else:
            os.link(name, target)
    finally:
        Path(name).unlink(missing_ok=True)
    print(f"Saved {target} ({image_size[0]}x{image_size[1]}), reprojection RMS {rms:.3f} px")
