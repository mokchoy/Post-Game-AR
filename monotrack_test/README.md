# One-rally MonoTrack projection test

This Python 3.14 test uses only the selected cleaned `rally_015.mp4`, its annotation CSV, `set-1/homography.csv`, and the existing `TrackNetv2-pretrained/track.pt`. It validates the expected 1920×1080, 25 fps, 524-frame video and 18 hit rows before calibration or inference.

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

## Geometry and reconstruction

The existing four court corners are reused. The program verifies that their stored homography maps them to approximately `(25,150)`, `(325,150)`, `(25,810)`, `(325,810)` in dataset coordinates, then derives a dataset-to-metre transform for the 6.1×13.4 m court. Player and floor-contact pixels are scaled by 1.5 to native video pixels and back to annotation coordinates before applying the floor transforms. The floor homography is never applied to airborne shuttle pixels.

The camera matrix follows MonoTrack's six-point linear calibration using the four stored corners and the two manually selected 1.55 m post tips. Hit frames come from `clip_frame_num`; every `clip_time_seconds` is checked against frame/FPS timing. TrackNet visibility zero is stored as a missing pixel. Each hit-to-hit segment uses the current hitter and next row's hitter as soft start/receive references; A/B player identity is kept from the annotation row. The fit uses actual 25 fps timing with `fr_adjust=1.0` and gravity plus quadratic drag in SLSQP.

The attempted net-fault tail is an extension to MonoTrack: last hit→net contact and net contact→floor contact are fit as separate flights, share the ray-derived net position, and may have different velocities. The net point comes from intersecting its native-pixel camera ray with `y=6.7 m`; the floor point comes from the existing floor homography. Missing manual events, insufficient visible detections, invalid event order, and failed optimizer phases remain unavailable.

Player locations are fit references, so agreement with them is not independent accuracy evidence. A plausible top-down path does not verify reconstructed height. The self-checks and synthetic tracker forward pass do not establish real-video tracking quality or physical trajectory accuracy.

## Attribution and adaptation

Camera calibration and the gravity-plus-drag/SLSQP reconstruction structure are adapted from [MonoTrack](https://github.com/jhwang7628/monotrack), inspected at commit `39e9227bd44fa2a579731b1cc97c7f529f391cd6` (`court.py` and `rally_reconstructor.py`). The upstream Adobe Research License is retained in [LICENSE](LICENSE); it permits use and redistribution for noncommercial research purposes only.

Adaptations: native-video scaling; the repository's supplied floor homography and derived metric conversion; explicit CFR frame timing; visibility-aware missing observations; consistent frame-indexed objective/simulation output; two piecewise terminal fits; diagnostics and a bounded renderer. This runner does not import MonoTrack's TensorFlow, pose, detector, hit model, C++ detector, training data, or checkpoints. It reuses the repository's TrackNet video helper and `track.pt`.
