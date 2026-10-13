# Court-plane shuttle projection

This tool reuses the rally validation, homography validation, and floor transforms in `monotrack_test`. It reads the existing `rally_015_tracked_pixels.csv`; it does not run TrackNet.

## Calibrate

From the repository root, run:

```powershell
python .\court_projection_test\court_projection.py calibrate
```

The OpenCV preview shows the source clip and court lines. Press `1`–`4` to choose left-post bottom, left-post top, right-post bottom, and right-post top. Click the corresponding point in the preview. Use `a`/`d` to step one frame and `A`/`D` to step 25 frames. Press `s` to save after all four selections, or `q`/`Esc` to cancel. Clicks are saved in native 1920×1080 video coordinates. The tool records the four selections, video and homography metadata, scaling, and the homogeneous vertical vanishing point; parallel post projections are represented at infinity. Existing calibration files are never overwritten.

## Run

After saving the manual calibration, run:

```powershell
python .\court_projection_test\court_projection.py run
```

Defaults use the specified rally CSV, matching MP4, homography, and existing tracked-pixel CSV. The command writes these files in this directory and refuses to overwrite any of them:

- `rally_015_ground_trajectory.csv`
- `rally_015_diagnostics.json`
- `rally_015_court_projection.mp4`

Each stroke uses its own `hit_x/y` as A and `landing_x/y` as B. The first serve uses `player_location_x/y` as A because its hit coordinates are missing. Each interval includes its start hit and excludes the next hit; the final stroke runs through the end of the clip without estimating a landing time. Endpoint sources are recorded in the diagnostics. Every pixel and annotation coordinate is scaled consistently between 1920×1080 video/tracker space and 1280×720 annotation space. Only the inferred floor intersection is passed to the floor homography.

Projections are rejected at or behind the floor homography's horizon, more than 2 m outside the court, or when moving the tracked pixel by 1 native pixel along either image axis changes the court position by more than 1 m (or makes the intersection invalid). The existing 1° minimum intersection angle remains in place. These conservative quality limits are recorded in diagnostics; they do not measure actual reconstruction accuracy. Ordinary out-of-court results within the margin are retained without clamping. By default, missing detections and invalid projections are left unavailable, with no interpolation or smoothing.

The output video is 1440×720 H.264 at the source 25 fps, with source audio. Its left panel shows the tracked shuttle, current AB line, and inferred floor footprint. Its right panel shows court-metre positions and trails, with gaps at missing/invalid frames and stroke boundaries. Court drawing is clipped to the plotting viewport to protect the heading, footer, and source video. Player A and B are labeled from each hit row's player/opponent location annotations and held at those positions for that stroke. FFprobe checks the output frame count, frame rate, dimensions, and audio/video timing.

If the default output files already exist, render a fresh copy into a subfolder rather than overwriting them:

```powershell
python .\court_projection_test\court_projection.py run --output-dir .\player_locations
```

Relative `--output-dir` paths are resolved inside `court_projection_test`.

The top-down panel also shows only the current stroke's annotated floor endpoints as labeled diamonds: magenta `Hit`, green `Landing`, or `Serve fallback` for the first missing hit. They switch at the next hit, remain separate from A/B player and shuttle markers, and do not change reconstruction or imply a measured landing time.

## Optional short-gap experiment

To enable filling and held-out validation on a future render, use a fresh output directory:

```powershell
python .\court_projection_test\court_projection.py run --fill-short-gaps --output-dir .\short_gaps
```

This command creates video as usual. The implementation checks below do **not** regenerate video, run tracking, or change the cached tracking/calibration/endpoints.

Only consecutive `missing_detection` runs of at most **5 frames (0.2 s at 25 fps)** are eligible. Both immediate anchors must be original detections with valid tracked projections, and every gap/anchor frame must belong to the same stroke. Native-image x and y are separately interpolated using frame times and [`numpy.interp`](https://numpy.org/doc/stable/reference/generated/numpy.interp.html), then passed through the existing projection checks. Every candidate must pass; one failure leaves the entire gap unavailable. No extrapolation, hit-boundary crossing, replacement of detections/rejected projections, or use of filled points as anchors is allowed. Long and edge gaps remain unavailable.

The trajectory CSV appends `estimated_shuttle_x_native_px`, `estimated_shuttle_y_native_px`, and `position_source`. Sources are `tracked` for valid observed projections, `interpolated` for accepted estimates, and `unavailable` otherwise. Missing frames keep their observed-coordinate fields blank; estimated-coordinate fields are populated only for accepted fills, whose status is `available`. Rejected observations keep their original coordinates and rejection status. The tracked-pixel input CSV is never written.

Interpolated shuttle/floor markers and trail segments touching a filled frame are orange; the video includes a legend. Unavailable frames and stroke boundaries still break trails, and court markers/trails stay clipped to the plotting viewport. Diagnostics report tracked/interpolated counts globally and per stroke, the enabled flag, fixed five-frame limit, eligible/accepted/rejected gaps, and skipped ineligible missing runs. Rejected-gap counts cover eligible gaps failing projection checks; long, edge, boundary, and invalid-anchor runs count as skipped. Disabled runs have zero gap counts and no held-out validation.

With filling enabled, validation independently hides the first eligible interior window of **1, 3, and 5 consecutive valid detections** per stroke. Each trial starts again from the original observations and removes all hidden pixels before reconstruction. Window selection requires valid tracked projections throughout the window and at both immediate anchors. Strokes without an eligible window are reported separately. The final stroke is excluded because its landing time is unknown; this exclusion applies to validation, not short-gap filling.

Printed results and diagnostics include attempted/accepted windows, accepted frames, mean Euclidean error, and RMSE in native pixels and court metres. The court comparison uses an endpoint-only AB trajectory moving at constant speed **in court metres**, with A/B at the existing start/next-hit contact times (`frame / fps`). Both methods are scored against the same projected references on exactly the accepted hidden frames. RMSE is the square root of the mean squared Euclidean distance, rather than a per-axis average. No advantage over AB is required.

Current cached-rally check (2026-10-13): **10 accepted gaps, 21 recovered frames**, with 269 tracked projections retained. The 46-frame gap `[433, 479)` remains unavailable. Held-out results:

| Hidden frames | Attempted / accepted windows | Native px mean / RMSE | Interpolated court m mean / RMSE | AB court m mean / RMSE |
| --- | --- | --- | --- | --- |
| 1 | 16 / 16 | 5.7773 / 6.8761 | 0.1874 / 0.3255 | 4.2122 / 4.9687 |
| 3 | 15 / 15 | 8.4813 / 10.1012 | 0.1991 / 0.2775 | 4.4356 / 5.1629 |
| 5 | 13 / 13 | 16.5852 / 19.7020 | 0.2678 / 0.3595 | 4.2308 / 4.8273 |

These measurements favor interpolation for the selected windows in this rally. Selection favors already valid tracked runs, so it may not represent actual missing-detection gaps. Held-out TrackNet detections and their projected positions are reference measurements, **not independent ground truth for actual court footprints**; both projections also share the same calibration and AB-plane assumptions.

## Self-check

```powershell
python .\court_projection_test\self_check.py
```

The assert-based check exercises finite/infinite vanishing points, known footprints at four heights, scaling, hit/landing endpoints and the serve fallback, invalid endpoints, projection-quality rejection, legitimate out shots, A/B labels, and overwrite refusal. It also checks successful time-weighted filling, unchanged observations/input CSV, atomic candidate rejection, preserved long/edge/stroke gaps, tracked-only anchors, independent masking and AB error scoring, CSV provenance, orange markers/trails and viewport clipping, and the current rally's 21 recovered frames/46-frame preserved gap. It prints the measured held-out results. Run integration checks write temporary CSV/JSON files with video rendering mocked; all temporary outputs are removed. No inference or video encoding runs.

## Approximation limits

Hit/landing annotations are treated as approximate floor footprints, with the serving player's location used only for the first starting point. Their accuracy is not independently verified. Flights nearly aligned with the apparent vertical direction can remain unavailable even with these endpoints. The final net-fault flight is represented by one approximate vertical plane and assumes the shuttle drops roughly beneath its net-contact position. The method estimates a floor footprint from each visible or optionally interpolated pixel; it does not estimate shuttle height or landing time. Short-gap image interpolation assumes locally linear motion and can miss acceleration or curvature. Long-gap reconstruction and physics fitting are excluded.
