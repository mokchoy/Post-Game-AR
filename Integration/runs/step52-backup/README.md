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

## Data flow

![Data flow: tracking, gestures, per-game data and AR projection](dataflow.png)

## Commands

All from `Integration/ar` with `../.venv/bin/python` (shown as `PY`). Camera index and backend live in the config (`camera.index`, currently 4 for the Logitech, v4l2); the laptop camera for gestures is `--gesture-camera` (default 0).

**1. Build a map for a new scene** (tag + textured static surroundings in view; move slowly through *translated* viewpoints, not just rotation; no hands or moving objects). `config-logitech-floor.json` is the build-mode config; use a new map name, never `--replace-map` over an existing one:

```bash
PY april-tag-projection.py --config config-logitech-floor.json --mapping-mode build --map data/logitech-landmark-map-scene2.npz --save-map --record-input
```

It saves once the bootstrap criteria are met (key `s` saves manually, `r` restarts the bootstrap). `--record-input` writes `runs/<id>/input.avi` so the map can be rebuilt offline.

**2. Localize only** (no panels from the array, single panel, occlusion on):

```bash
PY april-tag-projection.py --config config-logitech-floor.json --mapping-mode localize --map data/logitech-landmark-map-scene2.npz --occlusion --panel-video rally_002_projection.mp4 --record-input
```

**3. Full system** (4-panel array + gesture control). Point `mapping.map_path` in `config-array.json` at the new map (or pass `--map`):

```bash
PY april-tag-projection.py --config config-array.json --map data/logitech-landmark-map-scene2.npz --occlusion --gesture-camera 0 --record-input
```

Useful flags: `--no-gestures` (keyboard only), `--panel-video FILE` (override the left panel video for all games), `--headless`, `--max-frames N`, `--replay-run runs/<id>` (replay a recording), `--array-script "50:yaw:1,120:game:1"` (scripted actions for replay), `--depth-model relative|metric`.

Camera calibration (if the camera or resolution changes): `PY calibrate-camera.py` (keys `space` capture, `f` fullscreen, `q`/Esc quit).

## Keys (AR window focused)

| Key | Action |
|---|---|
| `q` / Esc | quit |
| `d` | toggle debug overlay |
| `f` | toggle fullscreen |
| `o` | toggle depth occlusion |
| `v` | toggle side-by-side view with depth |
| `p` | save the current display to `runs/<id>/sidebyside/` |
| `r` | reset tracking / map bootstrap |
| `s` | save map (build mode; `--replace-map` needed to overwrite) |
| `,` / `.` | previous / next game |
| `j` / `k` | yaw array left / right (15 deg steps, limit 60 deg) |
| `=` / `-` | scale array up / down (x1.1, range 0.5-2.0) |
| `t` | play / pause videos |
| `b` | reset yaw, scale, game |
| `x` | stop playback |
| `g` | show / hide the gesture pane (laptop camera + prediction) |
| `[` / `]` | gesture threshold -5 / +5 % |

Trackbars in the window: **Threshold %** (min confidence), **Vote window**, **Required votes** (gesture stability, as in `gestures/webcam_stable.py`).

## Gestures

The gesture thread triggers the same actions as the keys (table `gestures.actions` in the config). Each gesture fires once, then waits `gestures.cooldown_s` (1.0 s).

| Gesture | Action |
|---|---|
| Swiping Left / Right | previous / next game |
| Turning Hand Clockwise / Counterclockwise | yaw +/- |
| Zooming In / Out (full hand or two fingers) | scale +/- |
| Thumb Up | play |
| Thumb Down | reset |
| Stop Sign | stop |

## Where to configure things

Everything is a JSON config in `ar/` (`--config FILE`; paths inside are relative to the file). Omitted keys take defaults from `ar/ar_projection/config.py` (the dataclasses there list every key); unknown keys or bad values are rejected at start-up. Copy a config to a new name to experiment. Command-line flags override the config for the matching key (e.g. `--map` = `mapping.map_path`).

| Config file | Use |
|---|---|
| `ar/config-array.json` | full system: 4-panel array + gestures, localize mode, floor map |
| `ar/config-logitech-floor.json` | map building / single-panel localization for the floor scene |
| `ar/config-array-replay.json`, `ar/config-single-replay.json` | offline replay of a recorded run |
| `ar/config-logitech*.json`, `config-exp*.json` | older Logitech / experiment configs (history) |
| `gestures/config.json`, `gestures/training_config.json` | dataset and training only (not needed at run time) |

**Gestures** (`gestures` block of the AR config)

| Want to change | Key |
|---|---|
| What a gesture does (e.g. clockwise rotation) | `gestures.actions`, e.g. `"Turning Hand Clockwise": ["yaw", 1]`. Actions: `game`, `yaw`, `scale`, `play`, `stop`, `reset`; the number is the direction/step count (`-1` reverses) |
| Laptop camera | `gestures.camera_index` (or `--gesture-camera`) |
| Turn gestures off | `gestures.enabled: false` (or `--no-gestures`) |
| Confidence threshold / vote window / votes needed | `gestures.min_confidence` (0.70), `gestures.vote_window` (5), `gestures.required_votes` (4); also live via the window trackbars |
| Seconds of hand motion per prediction | `gestures.window_seconds` (3.0) |
| Pause between triggers | `gestures.cooldown_s` (1.0) |
| Classifier | `gestures.model` (`ddnet`), weights in `gestures/ddnet_final/` |

**Array / panels** (`array` block)

| Want to change | Key |
|---|---|
| Height above the tag | `array.lift_m` (0.5) |
| Yaw step / limit | `array.yaw_step_deg` (15), `array.yaw_limit_deg` (60) |
| Zoom step / range | `array.scale_step` (1.1), `array.scale_min` / `scale_max` (0.5 / 2.0) |
| Which way the array faces | `array.facing_yaw_deg` (180 = viewer on the +Y side) |
| Gap between panels, texture resolution | `array.gap_m`, `array.texture_long_px` |
| Panel sizes / layout | `array.panels` (name, side, width_m, height_m, toe_in_deg, above) |
| Games (videos, flags, scores, stats) | `array.games_manifest` -> `DataToProject/games.json`; regenerate with `ar/scripts/build_game_info.py` |
| Single (non-array) panel size and offset behind the tag | `panel.width_m`, `panel.height_m`, `panel.offset_y_m` |

**Tracking** (AR config)

| Want to change | Key |
|---|---|
| Camera, resolution, calibration file, undistort | `camera.*` |
| Tag size, tag ID | `tags.size_m`, `tags.anchor_id` |
| Number of features, detector settings | `features.max_features` (500), `features.grid_rows` / `grid_cols`, `features.fast_threshold`, `features.nlevels` |
| Matching strictness | `features.ratio_test`, `features.max_hamming_distance`, `features.guided_matching.*` |
| Optical-flow tracking | `features.lk.*` |
| Which map to localize against / build against | `mapping.map_path`, `mapping.mode` (`build` / `localize`) |
| Map size and quality gates | `mapping.max_landmarks`, `mapping.min_ready_landmarks`, `mapping.min_parallax_deg`, `mapping.min_baseline_m` |
| Pose source and robustness | `pose.mode` (`joint`, `landmark_only`, `tag_only`), `pose.ransac_*`, `pose.min_landmark_inliers` |
| Smoothing (jitter vs lag) | `filter.mode` (`ema`), `filter.measurement_weight` (0.4; higher = follows measurements faster) |
| Debug overlay on start | `display.debug`, `display.draw_features` |
| Where runs/recordings go | `logging.output_dir`, `logging.save_input_video` |

**Occlusion** (`occlusion` block): `occlusion.enabled`, `model` (`relative`), `repo` / `checkpoint` (Depth-Anything-V2 paths), `input_size` (252 / 392 / 518, bigger = slower), `margin_floor_m` / `margin_fraction` (how close to the panel before something occludes it), `reference` (`plane`), `background_prior.*`.

Tests: `../.venv/bin/python -m pytest -q tests` (110 pass). Gesture smoke test: `cd ../gestures && ../.venv/bin/python smoke_ddnet.py 0 30`.
- `ar/scripts/build_game_info.py` — regenerates `DataToProject/games.json` (per-game scores, flags, stats).
