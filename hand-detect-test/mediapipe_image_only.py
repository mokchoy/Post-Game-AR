"""Save IMAGE-mode overlays for all clips listed in output/*.mp4.

Run from repo root: py -3.14 hand-detect-test/mediapipe_image_only.py
Reads clean cached JPGs at native resolution. No cropping, resizing or tracking.
Writes output/image_only/<clip_id>.mp4; existing outputs are protected.
"""

from mediapipe_jester import FRAMES, MODEL, OUTPUT, mp, render_clip, vision


def main():
    videos = sorted(OUTPUT.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No existing clips in {OUTPUT}")
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    destination = OUTPUT / "image_only"
    destination.mkdir()
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    with vision.HandLandmarker.create_from_options(options) as hands:
        for video in videos:
            render_clip(FRAMES / video.stem, destination / video.name, hands, image_mode=True)


if __name__ == "__main__":
    main()
