# Anchor-referenced AR tracking, first pass

**Current Logitech workflow:** see [WORKFLOW.md](WORKFLOW.md) for the separate
17 cm / MSMF configuration, lossless timestamp-preserving replay, tracking
comparisons, physical acceptance, and the optional gated RGB-depth prototype.
Research references and their roles are in [RESEARCH.md](RESEARCH.md).

Python **3.10+**, NumPy, OpenCV contrib and SciPy. A calibrated RGB camera observes
AprilTag 36h11 ID 0 and a small frozen map of static natural landmarks. The app
renders the original opaque green panel using one jointly refined camera pose.
Tag-local X is right, Y is up, and Z is out of the printed front; the anchor defines
world W. `T_CW` maps world into camera; world camera position is `-R_CW.T @ t_CW`.

## Install

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Avoid an environment with incompatible NumPy/SciPy versions. A local `.venv` was
created for this implementation; the system Anaconda installation was untouched.
Generate a marker with a white margin, and print the outer black square at the
measured `tags.size_m` (170 mm for the current Logitech setup):

```powershell
python -c "import cv2; d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11); m=cv2.aruco.generateImageMarker(d,0,800); cv2.imwrite('tag36h11-id0.png',cv2.copyMakeBorder(m,80,80,80,80,cv2.BORDER_CONSTANT,value=255))"
```

## One configuration

Edit `config.json`; `--config PATH` selects a replacement full configuration with
the same schema. Unknown fields, wrong types, invalid combinations and unsupported
backends fail before opening a camera. CLI flags override only their corresponding
fields; the resolved effective config is saved with each logged run. Relative
config and CLI data paths resolve against the configuration file's directory.

The typed `calibration` section counts inner corner intersections, not printed
squares. The confirmed Logitech board has 7 by 9 inner corners with 0.020 m
squares; `config-logitech.json` requests 15 views. Change camera `width`/`height`
in the selected config for calibration and runtime.

Capture backends: `auto`, `dshow`, `msmf`, `v4l2`, `avfoundation`, `ffmpeg`.
`auto` uses DirectShow on Windows live capture and OpenCV auto selection elsewhere
and for files; explicit unavailable platform backends fail at capture opening.
Actual captured dimensions must match calibration exactly. A same-aspect mode
change may still change FOV, so recalibrate; the pipeline does not implicitly
resize or blindly scale intrinsics. Set camera focus/crop consistently.

With `camera.undistort=true`, cached remaps preserve full extent and original K;
all downstream observations use zero distortion. With it false, raw pixels and
the original distortion model are used, and triangulation normalizes observations.
Feature exclusion rectangles are integer `[x,y,width,height]` in the processed
image, clipped to its boundaries. Tag regions plus the configured margin are
excluded by default. These masks do not perform automatic hand segmentation.

## Calibration and the preserved panel baseline

```powershell
python .\calibrate-camera.py --camera 0 --columns 6 --rows 8 --square-size 0.02 --views 15 --output camera-calibration.npz
python .\april-tag-projection.py --camera 0 --calibration camera-calibration.npz --pose-mode tag_only --mapping-mode off
```

Calibration requires at least eight captured views. SPACE captures a detected board; Q/Esc
finishes. It uses raw images to estimate distortion, shares the configured source,
backend and requested dimensions, and retains the original NPZ keys
`camera_matrix`, `dist_coeffs`, `image_size`. Existing files are protected; explicit
`--replace-calibration` permits replacement.

The panel is 0.20 by 0.12 m, tilted 90 degrees around tag X, centred at
`[0,0.15,0.06]` m. Its +Y offset is **not general world vertical elevation**.
Its four corners at the default tilt have Z values `[0,0,0.12,0.12]`, preserving
the original geometry. `--tag-size`, `--width`, `--height`, `--elevation`, `--tilt`,
`--camera` and `--calibration` retain their original meanings. `D` toggles outlines
and axes; Q/Esc exits. Original calibrated invocation without mode flags now uses
the central default of joint tracking and bounded bootstrap.

```powershell
python .\april-tag-projection.py --demo-approx
python .\april-tag-projection.py --self-check
```

`--demo-approx` explicitly selects tag-only, mapping off and uncertainty disabled,
with a rough 60-degree-FOV camera. It is a labelled visual demo, unsuitable for
metric mapping/localisation or uncertainty claims. Explicit contradictory mode
flags are rejected. Panel projection remains floating point; only drawing rounds.
The panel hides on invalid tracking, nonpositive depth, or no image intersection.
Partially off-image panels are raster-clipped rather than hidden entirely.

## Bootstrap, save and load

Keep the anchor and textured static surroundings visible. Move slowly through
translated viewpoints, rather than rotating at a fixed camera centre:

```powershell
python .\april-tag-projection.py --mapping-mode build --pose-mode joint --save-map
python .\april-tag-projection.py --mapping-mode localize --pose-mode joint --map data/landmark-map.npz
python .\april-tag-projection.py --mapping-mode localize --pose-mode landmark_only --map data/landmark-map.npz
```

Bootstrap accepts unambiguous validated anchor poses, takes at most 50 useful
keyframes, reconstructs at most 500 landmarks, and requires at least three
confirming views with depth, baseline, parallax and reprojection gates. At least
80 stable landmarks are required by the current config to freeze the map. Optional sparse
bundle refinement jointly optimizes keyframe poses and points with the fixed
known-size anchor removing gauge freedom. The frozen map never grows at runtime.
The live window shows the configured stable-landmark and keyframe targets;
features with fewer than three confirming views do not count yet. Live/visible
runs build and refine snapshots in one cancellable background worker; headless
file replay stays sequential. Refinement has a five-second budget and can retry
after another accepted keyframe and a two-second cooldown. The GUI/logs report
worker status and optimizer failure details. If capacity is reached, the window prompts
you to press R. A prior saved 40-point map belongs to a different calibration and
10 cm tag; it is not compatible with the current Logitech setup.

`--save-map` explicitly saves once ready; a run ending before readiness fails with
a clear message. Alternatively press **S** to save. Existing maps are protected;
use a new `--map` path or explicitly pass `--replace-map`. **R** resets bootstrap,
tracks and motion/filter state. If capacity is reached without enough stable
points, inspect `bootstrap_reason` in JSONL and reset; thresholds are not lowered.

`localize` requires a ready compatible saved map and never silently starts a new
map. The map records schema, float64 world points, uint8 ORB descriptors, stable
IDs, support/quality, units, anchor and calibration/preprocessing identity. It is
loaded with `allow_pickle=False`. Calibration, tag size/ID/family or processed
image changes require rebuilding. The complete map is published atomically.

`pose.mode` supports `tag_only`, `landmark_only`, `joint`; actual per-frame source
is independently recorded as `tag`, `landmark`, `joint`, or `none`. Joint mode can
use tags during bootstrap or with mapping off. Landmark-only requires loaded-map
localisation. You may disable tags in the central config for landmark-only
localisation. Feature disabling also disables LK and landmark observations;
map building requires tags, features and actual calibration. Turning LK off keeps
ORB descriptor matching operational. Insufficient current evidence always loses
tracking; a previous pose is never displayed as fresh tracking.

## Replay and diagnostics

```powershell
python .\april-tag-projection.py --headless --video data/recording.avi --mapping-mode localize --pose-mode joint --map data/landmark-map.npz
python .\april-tag-projection.py --headless --video data/recording.avi --mapping-mode off --pose-mode tag_only
python .\april-tag-projection.py --headless --video data/recording.avi --mapping-mode localize --pose-mode landmark_only
```

Recordings must match the calibrated camera mode and dimensions. `--max-frames N`
limits a run. Video time is frame index/source FPS, independent of processing
wall time; invalid FPS is rejected. Live time is monotonic acquisition time, not
necessarily sensor exposure time. Nonincreasing times invalidate tracking; a
large source-time gap expires the pose prior and resets the smoother while
retaining LK identities for geometric checks and forcing ORB recovery.
`--record-input` saves lossless input; `--replay-run RUN` reuses the original
JSONL acquisition times rather than nominal FPS. NumPy/OpenCV are seeded and
frames run sequentially. Exact bitwise reproduction across versions/backends is
not promised.

Unique `runs/<UTC-time>-<id>/` directories contain `config.json`, `metadata.json`,
`frames.jsonl`, `summary.json` and optional videos. Every frame, including losses,
records states (`BOOTSTRAP`, `TRACKING`, `LOST`, recovered `RELOCALIZING`), raw and
render T_CW, world camera positions, inliers and group pixel RMS, tags and ambiguity,
map readiness/bootstrap reason, unrounded panel pixels, uncertainty, rejection
reasons, and detection/tracking/mapping/pose/covariance/rendering/total timings.
Summary includes tracking availability, source counts, failure durations and
median/p95 processing durations. Throughput is not capture-to-display latency.

Set `logging.save_input_video` or `save_annotated_video` in the same JSON to record
lossless FFV1/Huffyuv input AVI or MJPG annotated AVI. The Logitech config uses
Huffyuv for cheaper encoding; existing configs default to FFV1. Encoding has its
own frame timing and contributes to total processing time. Valid source FPS is used, with 30 FPS fallback for live devices lacking
FPS; frame JSONL timestamps remain authoritative. `logging.enabled=false` creates
no run directory and disables video writes. `display.enabled=false` or `--headless`
avoids all window calls. Debug/features/uncertainty overlays are separately
configurable; annotation happens on a copy after obtaining measurements.

## Conditional uncertainty and fixed smoothing

Pose refinement minimizes one whitened tag/landmark reprojection objective,
using PnP-RANSAC, independent group gates and at most one outlier refit. SciPy
robust losses are applied **per pixel coordinate**, with separate configured tag
and landmark noise. Spatial coverage, depth, convergence, residual and timestamp
motion gates apply; counts alone do not establish confidence. Planar ambiguous
tags are rejected unless independent mapped evidence resolves them.

`conditional_local_pose_covariance` is the inverse local robust-weighted
information under fixed calibration, fixed final map, assumed observation noise
and one selected pose mode. Its six-vector is `[dt_x,dt_y,dt_z,dr_x,dr_y,dr_z]` in a
**left** local retraction `Delta(dt,Rodrigues(dr)) @ T_CW`; this is not the exact
SE(3) exponential. Jacobians are finite differences in that same convention,
with scaled SVD rank/conditioning checks. Degenerate geometry produces `null`
and a reason, never fabricated zero variance or an unchecked pseudoinverse.

Calibration/map error, correlated observations, gross mismatches and alternative
planar modes are not fully represented. The map uses the same tag reference, so
this conditional estimate can be overconfident. Noise settings are assumed until
measured. Per-corner 2-by-2 pixel matrices use `G @ Sigma @ G.T`. Optional ellipses
show raw conditional approximately 2-sigma predictions; predicted uncertainty is
not jitter or registration accuracy. Disabling uncertainty makes it unavailable.

The optional fixed EMA is disabled in the Logitech diagnostic config. When enabled it smooths camera
position and geodesic orientation in T_WC, then inverts for rendering. It resets
on session changes/loss/recovery or time gaps and does not smooth panel corners
independently. Its fixed measurement weight depends on frame rate; compare at
the same source cadence. Raw covariance is never assigned to the filtered pose;
filtered covariance is unavailable. No adaptive gain policy is implemented.

## Tests and physical acceptance

```powershell
python .\april-tag-projection.py --self-check
python -m unittest discover -s tests -v
```

Tests require no camera or GUI and exercise calibrated synthetic observations,
noise/outliers, mapping/refinement, masks and tracking, loss/recovery, covariance,
persistence/configuration, and a generated recorded-video replay. See
[IMPLEMENTATION.md](IMPLEMENTATION.md) for check details and technical decisions.

1. Calibrate at runtime resolution with measured squares and intended focus/crop.
2. Run tag-only and verify the original panel remains attached to the printed tag.
3. Build/save with slow translation and static textured surroundings; restart and
   load for joint localisation.
4. Cover the tag while mapped features remain visible; verify landmark source.
5. Remove all usable evidence; verify LOST and a hidden panel. Restore anchor or
   mapped view; verify recovery without stale-filter interpolation.
6. Replay identical frames in all three modes and compare availability, held-out
   physical alignment and processing time. Select an explicitly stationary
   interval before computing jitter; do not call moving-camera spread jitter.
   Do not bridge missing frames or tracking resets in frame-step measurements.

No hardware test or real-camera accuracy is claimed. Residuals on fitted tag
corners and the fixed panel origin are not independent registration ground truth.
Tune noise, baseline/parallax, support, coverage, residual and speed thresholds
from recorded data; the supplied values are starting settings.

This pass supports temporary tag absence within previously mapped static coverage.
It does not promise an unrestricted 360-degree pan. The optional depth prototype
adds a diagnostic texture and scene-depth visibility after physical tracking
acceptance; it has not yet been validated on the new Logitech scene. Video
playback, badminton reconstruction, learned feature tracking, unlimited SLAM,
loop closure and adaptive filtering are outside this prototype.
