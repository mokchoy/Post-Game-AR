# One-rally MonoTrack projection test

This Python 3.14 tool uses a selected cleaned rally, its annotation CSV, `set-1/homography.csv`, and the existing `TrackNetv2-pretrained/track.pt`. It requires 1920×1080 at 25 fps, reads the actual frame count, and validates increasing hit frames and frame/FPS timing before calibration or inference. Fuzhou remains the default input for `run_projection.py`.

## Commands

Run the noninteractive checks, including strict checkpoint loading and a synthetic CUDA forward pass:

```powershell
py -3.14 monotrack_test\self_check.py --repo-root . --checkpoint TrackNetv2-pretrained\track.pt
```

Start manual calibration in an interactive desktop session:

```powershell
py -3.14 -B monotrack_test\run_projection.py calibrate `
  --video cleaned_rallies\set-1\Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals\set1\rally_015.mp4 `
  --rally-csv cleaned_rallies\set-1\Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals\set1\rally_015.csv `
  --homography-csv set-1\homography.csv `
  --calibration monotrack_test\calibration.json `
  --output-dir monotrack_test
```

The preview is scaled to at most 1280×720. Select `1` and click the left post top tip; select `2` and click the right post top tip. Those are the only two extra camera-calibration points. For the final shot, select its exact first floor-contact frame, press `4`, and click the shuttle's floor-contact pixel. This supplies the landing time and a fixed metric endpoint at `z=0`; do not click an airborne shuttle or a later bounce. Press `3` and click a known net interruption; a marked interruption between the final hit and landing leaves the final trajectory unavailable because a direct flight cannot model it. Press `u` for an unidentified event. Without an identified floor contact the final trajectory remains unavailable. Use `a`/`d` for one-frame stepping, `A`/`D` for 25-frame steps, and `q` to save. Clicks are converted back to native 1920×1080 pixels.

After saving calibration, run the single rally:

```powershell
py -3.14 -B monotrack_test\run_projection.py run `
  --video cleaned_rallies\set-1\Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals\set1\rally_015.mp4 `
  --rally-csv cleaned_rallies\set-1\Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals\set1\rally_015.csv `
  --homography-csv set-1\homography.csv `
  --calibration monotrack_test\calibration.json `
  --output-dir monotrack_test\run_001
```

Outputs are tracked shuttle pixels, per-frame metric 3D coordinates, per-flight optimizer/reprojection diagnostics, and a 1440×720 H.264 projection MP4 with source audio. Output paths are restricted to this folder tree and existing files are refused; use a new run directory for another attempt.

The top-down panel shows only the current stroke's annotated floor endpoints as labeled diamonds: magenta `Hit`, green `Landing`, or `Serve fallback` for the first missing hit. Markers switch at each hit and stay clipped to the panel, including the final stroke. Shuttle endpoint annotations are optional display references: missing or malformed markers are omitted and never affect fitting. Drawing these markers does not add constraints or establish landing timing; the source-video panel is unchanged.

## Geometry and reconstruction

The existing four court corners are reused. The program verifies that their stored homography maps them to approximately `(25,150)`, `(325,150)`, `(25,810)`, `(325,810)` in dataset coordinates, then derives a dataset-to-metre transform for the 6.1×13.4 m court. Player and floor-contact pixels are scaled by 1.5 to native video pixels and back to annotation coordinates before applying the floor transforms. The floor homography is never applied to airborne shuttle pixels.

The camera matrix follows MonoTrack's six-point linear calibration using the four stored corners and the two manually selected 1.55 m post tips. Hit frames come from `clip_frame_num`; every `clip_time_seconds` is checked against frame/FPS timing. TrackNet visibility zero is stored as a missing pixel. Each hit-to-hit flight uses the current hitter's `player_location_x/y` as its soft horizontal start reference and the next hitter's `player_location_x/y` at the next hit as its receive reference. Both are transformed through the existing floor homography and validated as finite 2D positions before any flight fitting. A/B identity selects the hitter at each endpoint. Shuttle `hit_x/y` and `landing_x/y` do not enter initialization, bounds, or loss.

The fit retains free height, existing soft-reference penalty weights, optimizer bounds logic, actual 25 fps timing (`fr_adjust=1.0`), and gravity plus quadratic drag in SLSQP. Existing continuity is preserved: a previous fitted 3D endpoint with hit reprojection error below 40 px fixes the next start, taking precedence over its soft reference. Each adjacent hit pair is fit as one flight. Per-flight diagnostics record `start_reference_xy_m`, `receive_reference_xy_m`, `start_reference_source`, and `receive_reference_source`, alongside existing reference-error and `start_fixed` fields. Sources are `current_row_player_location` and `next_row_player_location`.

Both pixel losses also include a MonoTrack-style hypothetical-landing penalty. Each candidate shot is simulated from launch as though the receiver never hits it, using the same gravity/drag integrator until its first ground crossing. Ground contact is interpolated between integration substeps. Let `d_out` be the horizontal distance of that landing from the nearest point in the full 6.1 x 13.4 m court rectangle (zero inside or on its boundary). The added cost is `(0.15 * camera_scale * d_out)^2`, matching the existing player-prior scaling; this weight has not been tuned on real rallies. Diagnostics record `landing_outside_distance_m` and the weighted `landing_penalty`. Extrapolation is capped at 60 seconds; nonfinite or missing ground contact rejects the candidate, and an optimized result without a valid landing is reported failed. This continuation is used only in the objective and never adds trajectory/video frames or uses annotated shuttle endpoints.

Fitting defaults to pixel MSE. Select `--pixel-loss huber --huber-delta-px 10` on `run_projection.py run` to use the scaled Huber loss; the delta is in cleaned-video pixels. Both runners fit the final shot to an identified manual first floor-contact frame/pixel. The final shot starts at the previous fitted endpoint when available; otherwise it uses the final hitter's soft horizontal reference. The endpoint is fixed at the floor position transformed through the homography, with nonnegative sampled height before contact. A measured out-of-court landing is allowed and does not receive the hypothetical in-court landing penalty. At least three visible TrackNet detections and three frames are still required, and failed fits remain unavailable. The landing sample is included; later samples remain unavailable, with no invented floor hold.

Terminal diagnostics record `manual_landing_used_for_fitting`, the landing frame/world point, fit status, and failure reasons. These final-shot estimates are manually assisted. Existing calibration JSONs with manual floor contacts now supply fitting constraints; verify those clicks represent the first floor contact before reuse. Unknown contacts remain unavailable, and existing output files remain protected from overwrite.

## Ten selected clips

`projection_clips.txt` lists ten set-1 clips from held-out matches, with two rallies each from Korea 2019, Denmark 2019, Fuzhou 2019 semifinal, World Tour Finals, and Malaysia 2018. Export their smoothed player tracks using the explicit `--clips` commands in `player_keypoint_test/README.md`. This does not require retraining. Save a fresh calibration for each clip and use the corresponding smoothed player CSV in its original review directory.

For example, after exporting the player reviews, replace `REVIEW_DIRECTORY` below with the printed review directory and run from the repository root:

```powershell
$review = 'REVIEW_DIRECTORY'
$rally = 'cleaned_rallies\set-1\CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal\set1\rally_028'
$players = Join-Path $review '777a2c3a75eabc9a\player_positions_smoothed.csv'
$calibration = 'monotrack_test\denmark_028_landing_calibration_001.json'
py -3.14 -B monotrack_test\run_projection.py calibrate `
  --video "$rally.mp4" --rally-csv "$rally.csv" --calibration $calibration
py -3.14 -B monotrack_test\run_projection.py run `
  --video "$rally.mp4" --rally-csv "$rally.csv" --calibration $calibration `
  --player-positions-csv $players --pixel-loss huber --huber-delta-px 10 `
  --output-dir monotrack_test\denmark_028_with_landing_001
```

In calibration, mark both post tips and the first floor contact. Use a new calibration filename and output directory for each attempt. The result is `rally_028_projection.mp4` plus tracked pixels, metric trajectories, and diagnostics. Repeat with each clip and its matching review CSV; `render_summary.json` maps rally keys to review directories.

### Rally video with current-shot-only top-down panel

To render only the existing results in `ten_clips_001`, run from the repository root:

```powershell
py -3.14 -B monotrack_test\render_current_shot.py
```

The script reads saved trajectory CSVs, projection diagnostics, hit annotations, and the original smoothed player review files. It does not run tracking, smoothing, or trajectory fitting. Each 1440x720 H.264 MP4 preserves the original projection video's entire left panel, including the rally footage and its overlays, and replaces only the right court panel. The court shows moving A/B player markers, the current shuttle marker, and only the current shot's trail up to the displayed frame. The trail clears at each annotated hit, including the final hit; missing coordinates never join across a gap. Unavailable players or shuttle positions have no marker. These remain the existing reconstructed estimates.

Videos retain the saved projection's audio when present and each clip's saved FPS/frame count. Outputs save beside each original projection as `ten_clips_001/<rally-key>/<rally-stem>_projection_current_shot.mp4`. Existing outputs are refused, and failed partial outputs are removed. The original projection videos and CSVs remain unchanged. `ffmpeg`, `ffprobe`, and the existing projection Python dependencies are required. Saved player CSV hashes must still match the diagnostics.

Focused synthetic tests are in `test_render_current_shot.py`. They were added for later execution; only static checks were performed for this change.

## Saved smoothed keypoint tracks

Both `run_projection.py run` and `compare_losses.py` accept `--player-positions-csv PATH`. Without it, annotated player positions remain the default. With it, use a saved `player_positions_smoothed.csv` in its original review directory alongside `metrics_smoothed.json` and the parent `selection.json`; the adapter loads these files without running the player model or smoothing again.

The loader verifies the exact source clip/rally, dimensions, FPS, frame count, frame/player uniqueness and timing, stable A/B court-half mapping, smoothing/model provenance, and coordinate compatibility with the current court geometry. It consumes exported court metres directly, checking native-pixel consistency using the keypoint exporter's pixel-centre scaling. Exported reference-coordinate columns and annotated player coordinates never supply priors in this mode; the rally CSV still supplies hit frames, timing, hitter identities, and optional shuttle display labels.

Each flight uses the current hitter at the exact current hit frame and the next hitter at the exact next hit frame. An unavailable required hitter position stops the command before TrackNet, fitting, or output creation; there is no annotation fallback. Available interpolated/held positions are accepted and identified in each endpoint's diagnostics. Player markers follow the saved track every frame and disappear when unavailable; shuttle annotation markers still switch by stroke. The saved CSV path/hash, final model/checkpoint provenance, smoothing settings, and filled-hitter count appear under `player_positions` in projection diagnostics or `provenance.player_positions` in comparison diagnostics.

For the existing saved Korea review, run from the repository root with fresh output directories:

```powershell
$players = "player_keypoint_test\runs\final_cv_5f_s42_1e4a7705f081\selected_rally_reviews\review_test_20261016T015452Z\39b5e3329a5c474e\player_positions_smoothed.csv"
$rally = "cleaned_rallies\set-1\Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final\set1\rally_002"

py -3.14 -B monotrack_test\run_projection.py run `
  --video "$rally.mp4" --rally-csv "$rally.csv" `
  --calibration monotrack_test\korea_calibration.json `
  --player-positions-csv $players --pixel-loss huber --huber-delta-px 10 `
  --output-dir monotrack_test\korea_keypoint_smoothed_huber_001

py -3.14 -B monotrack_test\compare_losses.py `
  --video "$rally.mp4" --rally-csv "$rally.csv" `
  --calibration monotrack_test\korea_calibration.json `
  --player-positions-csv $players `
  --gt-csv Professional\match5\csv\1_01_01_ball.csv `
  --gt-frame-offset 28 --huber-delta-px 10 `
  --output-dir monotrack_test\korea_keypoint_smoothed_comparison_001
```

For a separate MSE video, change `--pixel-loss huber` to `--pixel-loss mse` and use a different output directory. The comparison fits both losses from the same saved player tracks and one fresh TrackNet pass. These remain annotation-assisted reconstructions with exploratory smoothed player priors; availability and plausible rendering do not establish improved 3D accuracy.

## Korea Open paired loss comparison

Create a separate Korea calibration with the existing picker:

```powershell
py -3.14 -B monotrack_test\run_projection.py calibrate `
  --video cleaned_rallies\set-1\Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final\set1\rally_002.mp4 `
  --rally-csv cleaned_rallies\set-1\Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final\set1\rally_002.csv `
  --homography-csv set-1\homography.csv `
  --calibration monotrack_test\korea_calibration.json `
  --output-dir monotrack_test
```

Select `1` and `2` and click the two net-post tips. Select `3` and press `u` to leave net contact unidentified. Use `D`/`A` to move 25 frames (one second at 25 fps) or `d`/`a` to move one frame; the displayed frame counter is zero-based. At first floor contact, select `4`, click the shuttle pixel, then press `q` to save. The calibration file is separate from the Fuzhou calibration and refuses overwrite.

Run both losses from the same TrackNet predictions, annotations, and calibration. GT is used only for evaluation; `28` maps `GT frame 0` to cleaned frame `28`, and GT coordinates are scaled once by `1.5`:

```powershell
py -3.14 -B monotrack_test\compare_losses.py `
  --video cleaned_rallies\set-1\Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final\set1\rally_002.mp4 `
  --rally-csv cleaned_rallies\set-1\Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final\set1\rally_002.csv `
  --homography-csv set-1\homography.csv `
  --calibration monotrack_test\korea_calibration.json `
  --gt-csv Professional\match5\csv\1_01_01_ball.csv `
  --gt-frame-offset 28 `
  --huber-delta-px 10 `
  --output-dir monotrack_test\korea_loss_comparison_001
```

The default Huber threshold is 10 cleaned-video pixels. To override it, use (for example) `--huber-delta-px 5` and choose a different new output directory such as `monotrack_test\korea_loss_comparison_delta_5`. Existing output directories are refused. The report includes per-loss available-frame metrics and paired metrics on the shared visible-GT/valid-trajectory intersection. It excludes pre-hit padding, frames at/after manually selected floor contact, and invisible GT. The manual floor contact also constrains final-shot fitting, so final-flight metrics and overall metrics containing that flight are manually assisted, not independent final-shot accuracy. Hit-to-hit baseline metrics include the final hit sample and remain separate. Fitting residuals against TrackNet are reported separately from GT errors.

Dataset player floor positions are approximate contact priors, not exact racket/shuttle contact locations. Shuttle endpoint CSV annotations remain available for visual comparison only; the manually clicked first floor contact constrains only the final shot. A plausible top-down path or successful SLSQP convergence does not verify reconstructed height or physical accuracy. The self-checks and synthetic tracker forward pass do not establish real-video tracking quality or physical trajectory accuracy. Reference-selection and label-independence checks use a fitter spy without real-rally fitting; the manual-landing checks use synthetic observations and trajectories.

## Attribution and adaptation

Camera calibration and the gravity-plus-drag/SLSQP reconstruction structure are adapted from [MonoTrack](https://github.com/jhwang7628/monotrack), inspected at commit `39e9227bd44fa2a579731b1cc97c7f529f391cd6` (`court.py` and `rally_reconstructor.py`). The upstream Adobe Research License is retained in [LICENSE](LICENSE); it permits use and redistribution for noncommercial research purposes only.

Adaptations: native-video scaling; the repository's supplied floor homography and derived metric conversion; explicit CFR frame timing; visibility-aware missing observations; consistent frame-indexed objective/simulation output; player-based endpoint priors; diagnostics and a bounded renderer. This runner does not import MonoTrack's TensorFlow, pose, detector, hit model, C++ detector, training data, or checkpoints. It reuses the repository's TrackNet video helper and `track.pt`.
