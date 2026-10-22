# Logitech tracking investigation and gated occlusion prototype

Use PowerShell from this folder. All normal tracking commands use the existing
`.venv`; no ML packages are required. `config-logitech.json` selects camera 0,
MSMF, the 640x480 Logitech calibration, the measured 0.17 m tag, and a separate
`data/logitech-landmark-map.npz`. The built-in-camera calibration/map and
`config.json` are preserved. Smoothing is disabled in this diagnostic config so
it cannot conceal raw tracking behavior.

On Windows both entry points disable MSMF hardware transforms before importing
OpenCV, avoiding the camera startup stall observed on this laptop. This is a
process-only default; an explicitly set
`OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS` environment variable takes precedence.
The console reports capture opening, first-frame acquisition, and tracking
startup separately so an interrupted startup is distinguishable from a run.

## 1. Build a compatible map

```powershell
.\.venv\Scripts\python.exe .\april-tag-projection.py --config config-logitech.json --mapping-mode build --save-map --record-input
```

Keep the tag and static textured background visible; translate the camera in
several directions and include the intended operating viewpoints. Keep hands
and people out of the mapped background. The target is 80 stable points from
at least three views, with capacity for 50 keyframes. These counts do not prove
useful coverage. Wait for `Frozen map saved`, then press Q. Existing maps are
protected; choose another `--map` path or explicitly use `--replace-map` when
rebuilding. Failed quality checks never force-save an incomplete map.

Live camera runs and visible previews now build/refine immutable map snapshots
in one background worker. Tracking and capture continue while it works; a map is
published only after the existing support/depth/parallax/reprojection checks pass.
The worker coalesces new keyframes into the next snapshot rather than queueing a
job per frame. A failed refinement needs a new accepted keyframe and at least
`mapping.refinement_retry_interval_s` (default 2 seconds) before another attempt.
`R` cancels the old worker and discards its result; Q/Esc cancels pending work.
Wait for `Frozen map saved` before exiting if you need a saved map.

`mapping.refinement_time_budget_s` defaults to 5 seconds per optimization attempt.
Cancellation/deadline checks occur between residual evaluations, so a native
solver call can briefly overrun that budget. The solver still has its independent
100-evaluation limit. Neither limit permits a failed or geometrically invalid
map to be saved. Convergence tolerances and tracking/quality thresholds are unchanged.
Headless file replay keeps sequential mapping for reproducibility and applies
the same optimization budget; it can therefore pause during bootstrap.

The GUI shows background mapping/refinement status. Frame logs and the run
summary's `bootstrap_at_last_frame` include `bootstrap_worker` and
`bootstrap_refinement`: outcome, optimizer status/message, nfev/njev, actual
residual-call count, elapsed time, cost and optimality when available. Distinct
outcomes include `solver_failed`, `time_budget_exceeded`, `tag_geometry_failed`
and `final_quality_failed`. Main-thread `mapping` timing excludes worker time.
Use the refinement's own `elapsed_ms` to measure worker cost.

Recording remains opt-in. Omit `--record-input` for an unrecorded performance
baseline, then repeat with recording for reproducible experiments. The Logitech
config selects Huffyuv (`logging.input_codec: "HFYU"`), tested pixel-exact and
faster than FFV1 on this laptop, with larger files. Existing configurations default
to FFV1; both formats replay through the same path. Video encoding now has its
own `recording` timing and is included in total processing time. AprilTag and ORB
times are also separated; ORB statistics include duplicate-filter timing.
Steady-state localization FPS/latency must still be measured separately from
bootstrap and cannot be inferred from a background worker or headless replay.

## 2. Record a controlled, tag-visible camera path

```powershell
.\.venv\Scripts\python.exe .\april-tag-projection.py --config config-logitech.json --mapping-mode localize --record-input
```

Move through the mapped area for 30-60 seconds with the tag visible. Include
translation and rotation, a stationary interval, and difficult but still mapped
views. Press Q. Note the `runs/<time-id>` printed in the console. Input is saved
losslessly as Huffyuv AVI with the Logitech config (FFV1 for older configs), with
one authoritative `source_time_s` per frame in JSONL.
The AVI's nominal FPS does not replace those acquisition timestamps.

The GUI now distinguishes LK survival, ORB identity reuse, inherited/fresh map
associations, RANSAC inliers/coverage, landmark rejection, and panel visibility.
A pose-prior timeout forces ORB and resets the pose smoother, but retains feature
tracks for flow/geometric validation. Invalid/nonincreasing timestamps still
invalidate the frame and reset the image tracker. Loss never displays a frozen
pose as current evidence.

## 3. Replay the same frames and timestamps in all tracking variants

Replace `runs/YOUR-VISIBLE-RUN` with the recorded run directory:

```powershell
.\.venv\Scripts\python.exe -m ar_projection.experiments compare runs/YOUR-VISIBLE-RUN --config config-logitech.json --output runs/logitech-comparison.json
```

This runs tag-only, LK/ORB-every-15, ORB-only/every-frame, LK/ORB-every-frame,
then joint using the best qualifying landmark variant (or the explicitly labelled
baseline if none qualifies). Landmark trials mask detected tag polygons and
withhold their observations from the solver; a separate tag-only fit supplies
reference panel corners. The reference is not independent physical ground truth.

The initial standalone gates stay at 12 inliers, four grid cells and 3 px.
Joint fitting currently requires landmarks to pass that standalone gate first;
this policy is reported, not silently weakened to make the HUD say `joint`.
Compare LK survival against map-match counts, RANSAC rejections and reference
errors before changing thresholds. The redetection spatial radius is now
independent of the LK forward/backward consistency threshold.

If a landmark configuration meets the availability, reference-error and processing
shortlist criteria, the comparison writes `logitech-comparison.recommended-config.json`.
Headless throughput is only a shortlist: it cannot pass live FPS or latency.

For a single explicit replay, use:

```powershell
.\.venv\Scripts\python.exe .\april-tag-projection.py --config config-logitech.json --replay-run runs/YOUR-VISIBLE-RUN --mapping-mode localize --pose-mode landmark_only --withhold-tag --headless
.\.venv\Scripts\python.exe -m ar_projection.experiments summarize runs/YOUR-REPLAY-RUN --output runs/replay-summary.json
```

`--replay-run` uses input.avi and frames.jsonl together and rejects mismatched frame
counts/indices. It automatically chooses a file decoder while retaining the live
camera backend in the recorded configuration. For a standalone file, use
`--video FILE --timestamps FRAMES.jsonl --backend auto`. Without a sidecar,
ordinary `--video` retains the legacy frame-index/FPS clock.

## 4. Physical acceptance before depth

Use the shortlisted configuration if one exists; otherwise diagnose the failing
comparison before continuing. All three acceptance runs must use the same camera
calibration, tag size, map, and dimensions.

Record a live localization run using `--record-input` with the tag fully covered
for at least 120 uninterrupted seconds while moving the camera through mapped
views. Keep the tag physically fixed. Uncover it and test recovery. Repeat for a
five-minute continuous tag-hidden interval. Keep enough static map texture in
view; also deliberately remove all usable evidence and restore it to check LOST
and reacquisition. The operator supplies these physical scenarios; detection
statistics alone cannot prove that the tag was physically covered.

```powershell
.\.venv\Scripts\python.exe -m ar_projection.experiments assess --controlled runs/YOUR-WITHHELD-REPLAY --physical runs/YOUR-TWO-MINUTE-LIVE-RUN --soak runs/YOUR-FIVE-MINUTE-LIVE-RUN --output runs/tracking-acceptance.json
```

This checks 95% supported tag-absent poses, continuous 120/300 s intervals,
at least 100 frames of reference corners, median/p95 reference disagreement
below 5/10 px, recovery within one second after usable evidence returns, and
live 20 FPS / p95 100 ms read-to-display submission latency. Capture-read time
and post-acquisition latency are also reported separately. Exposure time, driver
buffers, and monitor scan-out are not measured. A headless run or a short synthetic
test cannot pass the physical acceptance gate.

## 5. Optional depth prototype, only after acceptance passes

The depth code is prepared but disabled by default. It checks the acceptance
report against the selected calibration, measured tag, map and dimensions before
importing ML code. No model weights or ML dependencies were installed in the
tracking environment. On a separate environment:

```powershell
.\.venv\Scripts\python.exe -m venv .venv-depth
.\.venv-depth\Scripts\python.exe -m pip install -r requirements-depth.txt
git clone https://github.com/DepthAnything/Depth-Anything-V2 third_party/Depth-Anything-V2
```

For GPU execution, install the CUDA-enabled PyTorch build appropriate to the
driver using the [official selector](https://pytorch.org/get-started/locally/).
Download the **Small Hypersim metric** checkpoint from the
[author metric-depth page](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)
to `models/depth_anything_v2_metric_hypersim_vits.pth`. The relative-depth model
is not interchangeable. Author downloads/setup are deliberately outside normal
tracking startup; none run until you explicitly perform this step.

```powershell
.\.venv-depth\Scripts\python.exe .\april-tag-projection.py --config config-logitech.json --mapping-mode localize --tracking-report runs/tracking-acceptance.json --depth-repo third_party/Depth-Anything-V2 --depth-weights models/depth_anything_v2_metric_hypersim_vits.pth --depth-input-size 252 --occlusion-output runs/depth-252 --record-input
```

The screen becomes a diagnostic checker texture. Its per-pixel **camera-Z metres**
are computed by ray/plane intersection, not a constant depth or normalized colour
image. Partially visible panels are clipped by image rasterization. Scene depth
uses the same rectified image and frame. Median positive scale ratios from at
least 12 geometrically retained map anchors align the model; relative median
absolute deviation must be <=0.25. The scale uses the median of the last five
accepted estimates. If anchors disappear or disagree, occlusion is explicitly
unavailable and the diagnostic texture is unoccluded; stale masks are not reused.
The visibility test hides panel pixels where real depth is more than 0.02 m in
front. This margin is a prototype setting, not an accuracy guarantee.

Export includes frame-indexed visibility PNGs (white = virtual screen visible),
rectified images, virtual depth NPYs, and JSON with timestamps/intrinsics/poses,
scale validity, inference time and peak allocated GPU memory. Failed scale
validation frames are exported too, avoiding a success-only evaluation subset.
No video texture playback is included in this prototype.

## 6. Label, compare and select a depth resolution

Repeat the same recorded scene at input sizes 252, 392 and 518. Use
`--replay-run` for identical frames and timestamps, and separate live runs for
display performance. Label actual virtual-screen visibility for hands and opaque
objects both in front of and behind the screen. Preserve frame indices. Example
manifest (paths relative to its JSON file):

```json
{
  "method": "DA-V2-Small-Hypersim",
  "input_size": 252,
  "live_run": "../runs/YOUR-LIVE-DEPTH-RUN",
  "frames": [
    {"prediction": "../runs/depth-252/000120.png", "truth": "labels/000120.png", "sequence": "desk", "frame_index": 120, "condition": "front", "inference_ms": 25.0},
    {"prediction": "../runs/depth-252/000121.png", "truth": "labels/000121.png", "sequence": "desk", "frame_index": 121, "condition": "behind", "inference_ms": 25.0}
  ]
}
```

The timing numbers above illustrate the format; replace them with measured
sidecar values. Add representative annotated frames rather than only two. The
same frames and white-visible convention can evaluate masks from the
[author Implicit Depth implementation](https://github.com/nianticlabs/implicit-depth).
Use its pretrained Hypersim model and temporal prior, with the exported virtual
depth and posed rectified images as input. Its dataset/keyframe preparation must
be performed in its own author environment. It has not been run or benchmarked
on this scene yet.

```powershell
.\.venv\Scripts\python.exe -m ar_projection.occlusion evaluate annotations/depth-252.json --output runs/depth-252-metrics.json
.\.venv\Scripts\python.exe -m ar_projection.occlusion select runs/depth-252-metrics.json runs/depth-392-metrics.json runs/depth-518-metrics.json --output runs/depth-selection.json
```

Evaluation reports mean visibility IoU (target 0.85), boundary F1 at two pixels,
front/behind group scores, and transition disagreement against labelled changes.
The temporal score is not a pose-warped flicker metric. Resolution selection picks
the highest size passing labelled quality, both ordering cases, metric-scale
availability and measured live FPS/latency. If none passes, the result is null
and occlusion remains opt-in; no quality or real-time success is fabricated.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests exercise synthetic tracking, source transitions, reprojection/outlier gates,
real timestamp gaps, lossless replay/ablations, acceptance failures, partial panel
clipping and camera-Z depth ordering. Hardware mapping, physical tag occlusion,
neural inference performance and manually labelled mask quality remain separate
measurements to perform with the new recordings.
