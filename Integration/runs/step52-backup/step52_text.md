### Step 52: demo video recording (shot list, recording commands, offline rendering; set by Sahaj 30 Oct)
- [ ] Goal: record short clips and stitch them into one demo video with transitions between differently arranged video arrays (grids of synchronised panes). Sahaj records; Claude prepared the code, the compositor and this shot list on 30 Oct.

**Segments (Sahaj, 30 Oct)**

| # | Segment | Grid | Length | Panes (left to right, top to bottom) |
|---|---|---|---|---|
| 1 | Map building | 1x2 | 5 s | camera view while building (features, landmark count) / third-person phone video of the person building the map |
| 2 | RGB + depth, occlusion | 1x3 | 7 s | camera / this frame's depth / AR with occlusion; one person walks in front of the array, one walks behind it |
| 3 | RGB + camera pose with covariance (post-processed) | 1x2 | 5 s | camera with the array / top-down map: landmarks, tag, array, trajectory coloured by pose source, camera frustum, 2-sigma position ellipse, sigma strip |
| 4 | Gestures | 3 rows of 1x2 | ~10 s | each row: AR / laptop camera with the classifier overlay; rows: rotate, swipe, zoom, all playing at once |
| 5 | RGB + game + video classification | 1x3 | 10 s | AR / the broadcast game video the centre panel shows / the top-down shot-classification video the left panel shows, at the same frames |

Assumption (ask Sahaj if wrong): "video classification" in segment 5 is the top-down current-shot video (`DataToProject/TopDownDerived`, the right half of `FullOriginal/*_projection_current_shot.mp4`). Set `"source": "full"` in the plan to show the whole side-by-side original instead.

**Principle: record live at full rate, render offline.** Live, run without the depth model (`--no-occlusion`) so the loop is not limited to about 10 fps by depth, and record the lossless input (`--record-input`). Then render every AR view offline with `--replay-run` (same poses: a replay of a recording reproduced the live `T_CW` to 2e-9 on 120 frames, 30 Oct) with occlusion, the clean or minimal HUD and `--save-video`. A replay never opens the laptop camera; `--replay-actions` re-applies the gesture/key actions logged live at the same frames, and the panel videos now follow the recorded source time, so the replay shows the same panel frames as the live run. Only the laptop-camera gesture video must be captured live (written automatically when gestures run with `--record-input`).

**Code added for this step (30 Oct)**
- CLI: `--no-occlusion` (never load the depth model), `--minimal-hud` (`display.minimal`: features, tag outline, corner ellipses, one large status line), `--replay-actions` (with `--replay-run`). `--replay-run` now forces `gestures.enabled=false` unless `--gesture-camera` is given.
- `--save-video` also writes `rgb.avi` (rectified camera image, no overlay). `--depth-video` also writes `scene_depth.avi` (this frame's aligned depth; `depth.avi` stays the stored background when the background prior is on, which does not show people).
- Gesture worker: when the live run records (`--record-input` or `--save-video`), it writes `gesture.avi` (laptop camera with the webcam_stable overlay, camera rate) and `gesture_frames.jsonl` (`time.monotonic()` per frame, the same clock as the AR `source_time_s`).
- Panel videos (`content.VideoLoop`) run on the frame's source time instead of the wall clock; `diagnostics.array.video_frames` logs `[game, frame index]` per AR frame for the game and left panels.
- Bug fix: `config-array.json` has no `games` key, but the defaults filled in 3 placeholder games before the manifest check, so `array.games_manifest` was ignored (runs from 27 Oct show "Game 1-3", 26 Oct runs show the 9 rallies). The manifest is now loaded whenever the user config has no `games`.
- `scripts/rec_video.py`: `probe RUN` (streams, per-second pose validity, logged actions and gesture events in run-relative seconds, best windows, darkest frames for sync), `segment PLAN NAME`, `stitch PLAN`, `all PLAN`. Frames are resampled to 30 fps by sample-and-hold on the logged times (the .avi nominal fps is wrong for the live loop). Output H.264 via ffmpeg; stitching uses ffmpeg `xfade` transitions (fade, slideleft, wipeleft, slideup, circleopen; per segment in the plan).
- `recording/plan.json`: the five segments with `REPLACE_...` placeholders for run ids, start times and the phone video.
- Tests: `tests/test_recording.py` (4: source-clock video loop, identical panel frames on identical timestamps and the swipe restart, manifest loading, compositor time alignment). Full suite in a copy on 30 Oct: 120 pass, 1 skipped, 4 fail; the 4 failures also fail without the Step 52 changes (3 `test_array.py` geometry tests expect the old 5 cm gap, 5 deg toe-in and +-60 deg limit defaults that `config.py` no longer has; the manifest test failed on the bug above and passes now on a machine with the flag images).

**Before recording**
- Logitech at index 4 (v4l2), laptop camera at index 0. Check the floor tag and the map scene have not moved since the map was built; if they have, record segment 1 first and use its map for the other segments (`--map`).
- Good, constant light. Optional: fix exposure and focus (`v4l2-ctl -d /dev/video4 -c auto_exposure=1 -c focus_automatic_continuous=0`; control names vary by camera, list them with `v4l2-ctl -d /dev/video4 -L`).
- Start every take with 2-3 s of the tag clearly in view, so the pose is valid before the action starts. Record each segment two or three times; keep the best take.
- All commands from `Integration/ar`, `PY=../.venv/bin/python`. After each take run `PY scripts/rec_video.py probe ../runs/<id>` and note the run id and the start time in `recording/plan.json`.

**Segment 1: map building (5 s, 1x2)**
1. Start the phone recording (landscape, the person and the laptop screen in view).
2. Run `PY april-tag-projection.py --config config-logitech-floor.json --mapping-mode build --map data/logitech-landmark-map-demo.npz --save-map --record-input --save-video --minimal-hud`. Use a new map name; never `--replace-map` an existing map.
3. Sync mark: cover the Logitech lens with a hand for about 1 s in view of the phone.
4. Move slowly through translated viewpoints (not only rotation) until the map saves; then `q`.
5. `probe` prints the darkest frames (the sync mark) in run seconds `t_run`. Find the same moment in the phone video (`t_phone`). For a segment that starts at run second `S`, the phone pane's `start` is `t_phone + (S - t_run)`.
- Panes: `run` annotated (built live: map building is timing-dependent, so the live view is the honest one), `file` phone video.

**Segment 2: RGB + depth with occlusion (7 s, 1x3)**
1. Camera on a tripod or held still, about 2.5-3 m from the array, tag in view. Live: `PY april-tag-projection.py --config config-array.json --no-occlusion --no-gestures --record-input`.
2. After 3 s, person A walks across between the camera and the array (about 1-1.5 m from the camera); then person B walks across behind the array (about 0.5-1 m behind its pivot). Slow walking, whole body in view. `q`.
3. Render: `PY april-tag-projection.py --config config-array.json --replay-run ../runs/<live id> --occlusion --save-video --depth-video --clean --headless`.
- Panes: `rgb`, `scene_depth`, `annotated` of the replay run. Occlusion with people has not been scored; check the render and pick the best take. If the array is hidden by the depth of the floor or a wall, note the run and frame for the depth work (Step 48).

**Segment 3: RGB + camera pose with covariance (5 s, 1x2)**
1. Live, handheld: `PY april-tag-projection.py --config config-array.json --no-occlusion --no-gestures --record-input`. Walk a slow arc of 1-2 m around the array at 1.5-3 m; for 1-2 s turn so the tag leaves the image (landmark pose, larger ellipse), then come back.
2. Render: `PY april-tag-projection.py --config config-array.json --replay-run ../runs/<live id> --no-occlusion --save-video --minimal-hud --headless` (or `--occlusion --clean` for a clean AR view).
- Panes: `annotated` of the replay, `posemap` of the replay. The map draws the raw pose; the ellipse is the logged conditional covariance (CONTEXT.md section 11), magnified by the stated factor (auto: the median ellipse is about 14 px). Step 51 found this covariance several times too large on the static run (NIS), so it shows relative, not calibrated, uncertainty; say so in the video caption or the paper.

**Segment 4: gestures (about 10 s, 3 rows of 1x2)**
1. Live, three takes (or one long take): `PY april-tag-projection.py --config config-array.json --no-occlusion --gesture-camera 0 --record-input`. Keep the gesture pane on (`g`) to see the classifier. Take A: turn hand clockwise, then counterclockwise. Take B: swipe right, then left. Take C: zoom in, then out. Wait about 3 s between gestures (3 s window, 4 of 5 votes, 1 s cooldown).
2. `probe` lists the gesture events with run times. Choose each row's `start` about 2 s before the first event, so the row shows the hand moving, the confirmation, and the array changing.
3. Render each take: `PY april-tag-projection.py --config config-array.json --replay-run ../runs/<live id> --replay-actions --no-occlusion --save-video --clean --headless` (`--occlusion` instead of `--no-occlusion` if the occlusion render looks right).
- Panes per row: `run` annotated of the replay, `gesture` of the LIVE run (the replay has no laptop-camera video). Both share the live run's timeline, so use the same `start`.

**Segment 5: RGB + game + video classification (10 s, 1x3)**
1. Live: `PY april-tag-projection.py --config config-array.json --no-occlusion --no-gestures --record-input`. Camera still or moving slowly, the whole array in view, the game playing (it plays from the start; `t` plays, `x` stops, `,`/`.` change game).
2. Render: `PY april-tag-projection.py --config config-array.json --replay-run ../runs/<live id> --replay-actions --occlusion --save-video --clean --headless` (or `--no-occlusion`).
- Panes: `annotated`, `panel` `video`, `panel` `topdown` of the replay run. The side panes read the frame index the panels showed, so they match the AR panels frame for frame.

**Assemble**
- Fill in `recording/plan.json`, then `PY scripts/rec_video.py segment recording/plan.json 03_pose` to check one segment, and `PY scripts/rec_video.py all recording/plan.json` for everything. Output: `runs/step52-video/<segment>.mp4` and `runs/step52-video/MOCam_demo.mp4` (1920x1080, 30 fps, 0.6 s transitions; the intro card is optional, delete `intro` to drop it).
- Check the final video: each segment's length, that the gesture rows show the event and the array change, that the side panes of segment 5 match the AR panels, and the sync of segment 1.

- Result: _(fill in: run ids per segment, starts, takes kept, anything that failed)_
