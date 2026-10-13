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

The preview is scaled to at most 1280×720. Select `1` and click the left post top tip; select `2` and click the right post top tip. Those are the only two extra camera-calibration points. Select `3` and click the net-contact frame/pixel, then `4` and click the first floor-contact frame/pixel. Press `u` for either terminal event when it cannot be identified; it will be recorded unavailable. Use `a`/`d` for one-frame stepping, `A`/`D` for 25-frame steps, and `q` to save. Clicks are converted back to native 1920×1080 pixels.

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

The top-down panel shows only the current stroke's annotated floor endpoints as labeled diamonds: magenta `Hit`, green `Landing`, or `Serve fallback` for the first missing hit. Markers switch at each hit and stay clipped to the panel, including the final stroke. Missing display markers are omitted; required fitting endpoints are validated as described below. Drawing these markers does not add constraints or establish landing timing; the source-video panel is unchanged.

## Geometry and reconstruction

The existing four court corners are reused. The program verifies that their stored homography maps them to approximately `(25,150)`, `(325,150)`, `(25,810)`, `(325,810)` in dataset coordinates, then derives a dataset-to-metre transform for the 6.1×13.4 m court. Player and floor-contact pixels are scaled by 1.5 to native video pixels and back to annotation coordinates before applying the floor transforms. The floor homography is never applied to airborne shuttle pixels.

The camera matrix follows MonoTrack's six-point linear calibration using the four stored corners and the two manually selected 1.55 m post tips. Hit frames come from `clip_frame_num`; every `clip_time_seconds` is checked against frame/FPS timing. TrackNet visibility zero is stored as a missing pixel. Completed hit-to-hit segments use the current row's `hit_x/y` and `landing_x/y`, transformed with the existing floor homography, as soft horizontal start/receive references. Only an absent first serve hit falls back to its player's location; any other missing, malformed, or nonfinite required endpoint raises a stroke-specific error before flight fitting. For the Fuzhou default rally, each completed-stroke landing coordinate equals the following hit coordinate, so it is used at the next hit time, not as a timed ground-contact measurement. A/B player identity is kept for display.

The fit retains free height, existing soft-reference penalty weights, optimizer bounds, actual 25 fps timing (`fr_adjust=1.0`), and gravity plus quadratic drag in SLSQP. Existing continuity is preserved: a previous fitted 3D endpoint with hit reprojection error below 40 px fixes the next start, taking precedence over its soft reference. Each adjacent hit pair is fit as one flight. Per-flight diagnostics record `start_reference_xy_m`, `receive_reference_xy_m`, `start_reference_source`, and `receive_reference_source`, alongside existing reference-error and `start_fixed` fields. Sources are `current_row_hit_x_y`, `first_row_player_location`, and `current_row_landing_x_y`.

Fitting defaults to pixel MSE. Select `--pixel-loss huber --huber-delta-px 10` on `run_projection.py run` to use the scaled Huber loss; the delta is in cleaned-video pixels. The attempted net-fault tail remains an extension to MonoTrack: last hit→net contact and net contact→floor contact are fit as separate flights and share the ray-derived net position. If net contact is unidentified but floor contact is selected, the last hit→floor contact is fit directly with the fixed floor endpoint. The floor point comes from the existing floor homography and may lie outside the court. Missing contacts, insufficient visible detections, invalid event order, and failed optimizer phases remain unavailable.

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

The default Huber threshold is 10 cleaned-video pixels. To override it, use (for example) `--huber-delta-px 5` and choose a different new output directory such as `monotrack_test\korea_loss_comparison_delta_5`. Existing output directories are refused. The report includes per-loss available-frame metrics and primary paired metrics on the shared visible-GT/valid-trajectory intersection. It excludes pre-hit padding, frames at/after manually selected floor contact, and invisible GT; missing estimates are reported as coverage gaps. Hit-to-hit baseline metrics and the final-flight extension metrics are separate. Fitting residuals against TrackNet are reported separately from GT errors.

Annotated endpoints (and the serve player fallback) are fit references, so agreement with them is not independent accuracy evidence. Improvement over player-based references remains unverified. A plausible top-down path does not verify reconstructed height. The self-checks and synthetic tracker forward pass do not establish real-video tracking quality or physical trajectory accuracy. Reference-selection checks use a fitter spy without real-rally fitting; run `python -B monotrack_test/self_check.py --repo-root .` without `--checkpoint` for checks without tracker inference.

## Attribution and adaptation

Camera calibration and the gravity-plus-drag/SLSQP reconstruction structure are adapted from [MonoTrack](https://github.com/jhwang7628/monotrack), inspected at commit `39e9227bd44fa2a579731b1cc97c7f529f391cd6` (`court.py` and `rally_reconstructor.py`). The upstream Adobe Research License is retained in [LICENSE](LICENSE); it permits use and redistribution for noncommercial research purposes only.

Adaptations: native-video scaling; the repository's supplied floor homography and derived metric conversion; explicit CFR frame timing; visibility-aware missing observations; consistent frame-indexed objective/simulation output; two piecewise terminal fits; diagnostics and a bounded renderer. This runner does not import MonoTrack's TensorFlow, pose, detector, hit model, C++ detector, training data, or checkpoints. It reuses the repository's TrackNet video helper and `track.pt`.
