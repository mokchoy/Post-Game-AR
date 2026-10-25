"""Extract every output MP4 into output/frames/<clip_id>/00001.jpg, etc.

Run from repo root: python hand-detect-test/extract_overlay_frames.py
Existing frame folders are protected from overwriting.
"""

from pathlib import Path

import cv2


OUTPUT = Path(__file__).resolve().parent / "output"


def extract_frames(video):
    folder = OUTPUT / "frames" / video.stem
    capture = cv2.VideoCapture(str(video))
    count = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        folder.mkdir(parents=True)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            path = folder / f"{count + 1:05d}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Cannot save frame: {path}")
            count += 1
        if count == 0:
            raise RuntimeError(f"No readable frames in {video}")
    finally:
        capture.release()
    print(f"{video.name}: {count} frames saved to {folder}")


def main():
    videos = sorted(OUTPUT.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No MP4 videos in {OUTPUT}")
    for video in videos:
        extract_frames(video)


if __name__ == "__main__":
    main()
