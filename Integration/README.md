# Integration (Step 50 workspace)

All Step 50 work happens here. `../sahaj_ar_projection` and `../dynamic-gesture-classification` are read-only sources: files were copied, never edited or run in place.

## Layout and provenance

| Folder | Copied from | Notes |
|---|---|---|
| `ar/` | `sahaj_ar_projection/` | `ar_projection/`, entry points, `config*.json`, calibration, `tests/`, docs, `scripts/` (the tests import them), `data/logitech-landmark-map-occ.npz`, plus `data/logitech-landmark-map-floor.npz` (Sahaj's new floor map, not in the Step 50 list) and `rally_002_projection.mp4`. No `runs/`, no venvs. |
| `gestures/` | `dynamic-gesture-classification/` | `*.py`, `config.json`, `requirements.txt`, `licenses/`, `tests/`, `DDNET_REFERENCE.md`, `README.md`, `training_config.json`, `ddnet_final/`. `models/hand_landmarker.task` is from `demo_hand_gesture_recognition/`. |
| `runs/` | new | All new runs (configs resolve `logging.output_dir` to `../runs`). |

`SOURCES.tsv` lists every copied file with the SHA-256 of its source and of the Integration copy, and marks the 6 files edited after copying: `ar/ar_projection/{cli,config,occlusion,pipeline,rendering}.py` and `gestures/webcam.py` (adds `load_job`). New files (not in the sources): `ar/ar_projection/{array,content,gestures}.py`, `ar/tests/test_array.py`, `ar/scripts/step50_report.py`, `ar/config-array*.json`, `ar/config-single-replay.json`, `gestures/smoke_ddnet.py`. Paths inside configs are relative to the config file, so they point at the Integration copies. The depth model stays at `/home/sahaj/Desktop/Studies/compvis/Depth-Anything-V2` (read-only). `ar/config-exp-handoff.json` still holds old Windows paths (historical, unused).

## Environment (`Integration/.venv`, system Python 3.13.12; mediapipe has a cp3-none wheel, no other Python needed)

torch 2.14.1 (CUDA available), torchvision 0.29.1, numpy 2.5.4, scipy 1.18.1, opencv-contrib-python 4.14.0.94 (pinned `<5`: pip's default is 5.0, and `.venv-depth` uses 4.14), mediapipe 1.1.1, pytest 9.1.1.
Not installed: `rtmlib`, `papermill` (training/RTMPose only). 2 gesture tests that need `rtmlib` fail and `gestures/tests/test_training.py` cannot be collected; the 47 other gesture tests pass.

## Run

From `Integration/ar`, with `../.venv/bin/python`:

```bash
../.venv/bin/python april-tag-projection.py --config config-array.json --occlusion --panel-video rally_002_projection.mp4 --gesture-camera 0 --record-input
```

Tests: `../.venv/bin/python -m pytest -q tests` (110 pass). Gesture smoke test: `cd ../gestures && ../.venv/bin/python smoke_ddnet.py 0 30`.
