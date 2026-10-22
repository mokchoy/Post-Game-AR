# First-pass implementation and verification

These are the original first-pass notes. The current Logitech configuration,
tracking diagnostics/replay workflow, and acceptance-gated depth prototype are
documented in [WORKFLOW.md](WORKFLOW.md); research links are in [RESEARCH.md](RESEARCH.md).

Implemented in `ar_projection/`: strict typed configuration, camera/preprocessing,
geometry and shared records, tag candidates, grid ORB and forward/backward LK,
bounded map bootstrap and sparse joint bundle refinement, pose estimation,
conditional covariance, fixed pose EMA, rendering, diagnostics and orchestration.
The two original root scripts are thin CLI entry points. `config.json` is the
only tunable default source. Added `tests/test_tracking.py`, `.gitignore`, and
SciPy in `requirements.txt`; replaced the README with exact usage and limits.

## Checks actually run

In the project-local Python 3.12 environment, using NumPy 2.5.4,
OpenCV contrib 4.14.0 and SciPy 1.18.1:

```powershell
.\.venv\Scripts\python.exe april-tag-projection.py --self-check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q ar_projection tests april-tag-projection.py calibrate-camera.py
.\.venv\Scripts\python.exe -m pip check
```

The suite covers independent pinhole generating geometry, distorted projection
and triangulation, IPPE ordering and ambiguity, three-view support and bounded
map storage, sparse refinement, descriptor identity recovery, mixed noisy
observations with 30% gross feature mismatches, ablations, tag rejection,
motion gates, conditional noise scaling, rank failure, a 1,500-sample pixel
uncertainty experiment, safe map persistence and compatibility, invalid
configuration, ORB/LK and descriptor-only paths, disabled GUI/logging, loss and
recovery with filter reset, and synthetic MJPG video through the real capture,
detection, frame loop and logging. Replay checks compare estimates and source
times; processing durations are deliberately excluded from equality checks.

Synthetic covariance tests validate the conditional approximation only. They
do not establish statistically calibrated uncertainty for a real camera.
No physical calibration, webcam tracking, real scene map, held-out physical
registration measurement, or real recorded-camera replay was available.
Both entry points' `--help` output was checked. Dependency consistency and
compilation checks passed; the 13-test suite and legacy self-check passed.

## Technical decisions and deviations

- Added the optional typed `calibration` section to the central JSON for the
  supplied 6-by-8 inner-corner, 20 mm, 15-view checkerboard. Calibration necessarily observes raw
  images to estimate distortion; runtime optionally undistorts those images.
- Capture dimensions must exactly match calibration dimensions. There is no
  explicit resize step, so same-aspect capture changes are rejected rather
  than assumed to preserve FOV. This is stricter than the old prototype's
  automatic intrinsic scaling and meets the brief's known-resize requirement.
- Undistortion preserves original K and the full image extent with cached
  remaps. Video timestamps use frame index/source FPS; invalid FPS is an error.
- If any panel corner is outside the processed image or behind the camera,
  the whole panel and its pixel covariance are unavailable. This conservative
  clipping policy avoids partially visible polygons with ambiguous diagnostics.
- Fixed map descriptors are used directly for recovery; there is no persisted
  place-recognition database. Bootstrap keyframes remain bounded in memory.
- Calibration and map writes protect existing files and publish atomically.
  Replacing either requires its documented explicit CLI flag.
- Supported Python is 3.10+. The pre-existing Anaconda installation was unusable
  for SciPy optimisation due to incompatible NumPy/SciPy; it was left unchanged.

The default bootstrap keyframe bound is 20 rather than the brief's starting value
of 8. A logged camera run reached 8/8 by frame 85 with about 700 tracked features
but only 25/40 stable landmarks, then rejected subsequent keyframes at capacity.
The higher bounded capacity lets additional translated views support those points.
Other initial tracking thresholds are retained from the brief. Measure tag and
feature pixel noise, baseline/parallax, support requirements, image coverage,
residual limits and speed gates on recorded camera data before tuning them.
EMA strength is per-frame and changes behavior with source cadence.

## Design references

The solver follows joint reprojection rather than blending separate poses,
consistent with the scoped idea in [UcoSLAM sections 3.2, 4.1, 4.2 and 4.4](https://arxiv.org/html/1902.03729).
It uses explicit robust per-coordinate loss through
[SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
and the camera/PnP/triangulation contracts in the
[OpenCV calibration API](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
These references do not imply reproduced paper performance.

Deferred scope remains textured/video panels, badminton reconstruction, hand
segmentation/occlusion, adaptive filtering, learned features, unrestricted SLAM,
loop closure and room-scale/360-degree tracking. No switches pretend to enable
these capabilities.
