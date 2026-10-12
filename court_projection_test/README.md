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

Projections are rejected at or behind the floor homography's horizon, more than 2 m outside the court, or when moving the tracked pixel by 1 native pixel along either image axis changes the court position by more than 1 m (or makes the intersection invalid). The existing 1° minimum intersection angle remains in place. These conservative quality limits are recorded in diagnostics; they do not measure actual reconstruction accuracy. Ordinary out-of-court results within the margin are retained without clamping. Missing detections and invalid projections are left unavailable, with no interpolation or smoothing.

The output video is 1440×720 H.264 at the source 25 fps, with source audio. Its left panel shows the tracked shuttle, current AB line, and inferred floor footprint. Its right panel shows court-metre positions and trails, with gaps at missing/invalid frames and stroke boundaries. Court drawing is clipped to the plotting viewport to protect the heading, footer, and source video. Player A and B are labeled from each hit row's player/opponent location annotations and held at those positions for that stroke. FFprobe checks the output frame count, frame rate, dimensions, and audio/video timing.

If the default output files already exist, render a fresh copy into a subfolder rather than overwriting them:

```powershell
python .\court_projection_test\court_projection.py run --output-dir .\player_locations
```

Relative `--output-dir` paths are resolved inside `court_projection_test`.

## Self-check

```powershell
python .\court_projection_test\self_check.py
```

The assert-based check exercises finite and infinite vertical vanishing points, known floor footprints at four shuttle heights, coordinate scaling, hit/landing endpoints and the serve fallback, invalid endpoints, horizon/sensitivity/plausibility rejection, legitimate out shots, clipped trails/markers, A/B player labels, missing detections, degenerate intersections, and overwrite refusal. It reads clip metadata and existing CSV inputs, but does not render a video or run inference.

## Approximation limits

Hit/landing annotations are treated as approximate floor footprints, with the serving player's location used only for the first starting point. Their accuracy is not independently verified. Flights nearly aligned with the apparent vertical direction can remain unavailable even with these endpoints. The final net-fault flight is represented by one approximate vertical plane and assumes the shuttle drops roughly beneath its net-contact position. The method estimates a floor footprint from each visible pixel; it does not estimate shuttle height or landing time.
