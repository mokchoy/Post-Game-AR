"""Step 9b follow-up: visually compare actual image crops of the same landmark across the build
recording and the separate test clip (same lighting/exposure/focus per Sahaj), per the Step 9b
decision table's last row ("investigate offline first: compare build and test frames of the same
landmark"). Read-only, offline, no tracker change.

Reports a blur proxy (Laplacian variance) and mean brightness per crop, and saves a composite PNG
grid for visual inspection.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar_projection.camera import Capture
from ar_projection.config import load_config
from ar_projection.geometry import project
from ar_projection.pose import PoseSolver
from ar_projection.tags import TagDetector
from descriptor_validity_check import replay_build_with_provenance  # noqa: E402

CROP_RADIUS = 20


def crop(image, pixel, radius=CROP_RADIUS):
    x, y = int(round(pixel[0])), int(round(pixel[1]))
    h, w = image.shape
    x0, x1 = max(0, x-radius), min(w, x+radius)
    y0, y1 = max(0, y-radius), min(h, y+radius)
    return image[y0:y1, x0:x1]


def crop_stats(c):
    return {"blur_laplacian_var": float(cv2.Laplacian(c, cv2.CV_64F).var()), "mean_brightness": float(c.mean()), "shape": c.shape}


def find_test_clip_view(clip_config_path, landmark_id, landmark_point, tight_radius=3.0, max_frames=None):
    """First frame where this landmark has a keypoint within tight_radius of its projection."""
    cfg = load_config(clip_config_path)
    capture = Capture(cfg.camera)
    packet = capture.read()
    camera = packet.camera
    tag_detector = TagDetector(cfg.tags)
    reference_solver = PoseSolver(replace(cfg, pose=replace(cfg.pose, mode="tag_only"), uncertainty=replace(cfg.uncertainty, enabled=False)))
    index = 0
    while packet is not None and (max_frames is None or index < max_frames):
        tags = tag_detector.detect(packet.gray, camera)
        reference = reference_solver.estimate(tags, [], [], camera)
        if reference.valid and reference.tag_rms is not None and reference.tag_rms <= 1.0:
            proj = project(reference.T_CW, landmark_point[None], camera)[0]
            keys = cv2.ORB_create().detect(packet.gray, None)
            if keys:
                pixels = np.array([k.pt for k in keys])
                d = np.linalg.norm(pixels-proj, axis=1)
                i = int(np.argmin(d))
                if d[i] <= tight_radius:
                    capture.close()
                    return packet.gray.copy(), pixels[i], index
        packet = capture.read()
        index += 1
    capture.close()
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-config", default="runs/20261019T090841-72509636/config.json")
    parser.add_argument("--clip-config", default="runs/20261019T133214-ecda4960/config.json")
    parser.add_argument("--landmark-ids", nargs="+", type=int, default=None, help="default: first 4 landmark ids in the map")
    parser.add_argument("--output", default="runs/build-vs-test-crops.png")
    args = parser.parse_args()

    build_map, provenance, _ = replay_build_with_provenance(args.build_config)
    points = {int(id): p for id, p in zip(build_map.ids, build_map.points)}
    ids = args.landmark_ids or [int(i) for i in build_map.ids[:4]]

    rows = []
    for id in ids:
        views = provenance.get(id, [])
        if not views:
            print(f"landmark {id}: no build provenance, skipping")
            continue
        _, source_pixel, source_image = views[0]
        source_crop = crop(source_image, source_pixel)
        other_crop = other_stats = None
        if len(views) > 1:
            _, other_pixel, other_image = views[1]
            other_crop = crop(other_image, other_pixel)
        test_image, test_pixel, test_frame = find_test_clip_view(args.clip_config, id, points[id])
        test_crop = crop(test_image, test_pixel) if test_image is not None else None
        print(f"landmark {id}:")
        print(f"  build source: {crop_stats(source_crop)}")
        if other_crop is not None:
            print(f"  build other view: {crop_stats(other_crop)}")
        if test_crop is not None:
            print(f"  test clip (frame {test_frame}): {crop_stats(test_crop)}")
        else:
            print("  test clip: no tight-radius view found")
        rows.append((id, source_crop, other_crop, test_crop))

    size = 2*CROP_RADIUS
    tile = size+4
    grid = np.full((len(rows)*tile, 3*tile), 255, np.uint8)
    for r, (id, source_crop, other_crop, test_crop) in enumerate(rows):
        for c, crop_image in enumerate((source_crop, other_crop, test_crop)):
            if crop_image is None:
                continue
            h, w = crop_image.shape
            grid[r*tile:r*tile+h, c*tile:c*tile+w] = crop_image
    cv2.imwrite(args.output, grid)
    print(f"saved comparison grid (rows=landmarks {ids}, cols=[build source, build other view, test clip]) to {args.output}")


if __name__ == "__main__":
    main()
