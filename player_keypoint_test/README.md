# Point-only YOLO11n player localisation

This experiment uses only `cleaned_rallies/set-1`. It writes data, settings, checkpoints, and results inside this folder. It does not edit source footage, annotations, cleanup reports, baseline code, or files elsewhere in the repository.

The model uses YOLO11n layers 0–10 as a feature backbone and a small two-channel stride-4 heatmap decoder: far player in channel 0 and near player in channel 1. It predicts one proxy floor-reference point for each side of the court. A/B identities come from the saved rally court-half mapping. Heatmap peak scores are diagnostics, not presence probabilities. These annotations do not establish foot-midpoint labels or calibrated localisation accuracy.

## Python and dependencies

Run with the installed system Python. The experiment requires the pinned `ultralytics==8.4.175`, CUDA-enabled PyTorch when using the default CUDA device, OpenCV, NumPy, and PyYAML. The code does not create a virtual environment or install packages. `runner.py check` uses the existing official `yolo11n.pt` if present; otherwise it downloads only that checkpoint into `weights/`.

## Commands

Run in PowerShell from this folder:

~~~powershell
python runner.py check
python self_check.py
python runner.py prepare
python runner.py review
python runner.py split
python runner.py train-cv --folds 5
python runner.py train-final --run-id <CV_RUN_ID>
python runner.py evaluate --run-id <FINAL_RUN_ID> --split test
python runner.py overlay --run-id <FINAL_RUN_ID> --match <MATCH_FOLDER> --game 1 --rally 1
python runner.py review-test --run-id <FINAL_RUN_ID> --check
python runner.py review-test --run-id <FINAL_RUN_ID>
python runner.py review-test --run-id <FINAL_RUN_ID> --smooth --check
python runner.py review-test --run-id <FINAL_RUN_ID> --smooth
~~~

`check` verifies pretrained transfer, model output, independent initialization, checkpoint reload, gradient flow, and one synthetic optimization step. `self_check.py` adds source-filter, coordinate, fold, configuration, missing-prediction, and synthetic review-render checks. Neither command prepares the full dataset or trains on badminton footage.

`--config PATH` is supported by `check`, `prepare`, `split`, and `train-cv`. The YAML fold count is used unless `--folds N` is supplied to `split` or `train-cv`. Fold counts must be between 2 and the number of development matches. `train-final` reuses the selected CV run's saved settings.

## Data and review

`prepare` screens rally-level source quality, permanently omits the two specified CHOU matches, and extracts exactly the annotated `clip_frame_num` images. A/B are aligned using the hitter's `player` field. Coordinates are scaled from the 1280×720 reference image to native resolution and are never inverse-projected through a homography. The version-2 point manifest records source provenance, frame timing, dimensions, both identities and coordinates, and the far/near mapping. Heatmap targets are built when images are loaded; no detector, boxes, or pose labels are produced.

Every automatic source-quality failure excludes the complete rally and is written to `data/candidate/exclusions.csv`. The OpenCV reviewer samples up to five distinct rallies spread through each match, showing each rally's middle annotated frame and the two labelled reference points. Press A to accept, R to reject the rally, M to reject the match, S to save and quit, or Q to quit and resume later. Automatic checks and sampled-review status are stored separately; remaining frames do not require individual approval.

`split` freezes the accepted point manifest and reserves 20% of matches as the seed-42 held-out set. CV fold assignments keep whole matches together. Fold count and training settings do not change held-out membership. Each fold independently loads the same pretrained backbone and selects its checkpoint by validation mean court-metre error; any invalid projection makes that epoch's selection score infinite. OOF exports include combined, player, match, and fold metrics; unavailable points count as failures in coverage and distance-success denominators.

Final training uses all development matches for the rounded median selected epoch count and keeps the held-out matches out of training and checkpoint selection. `evaluate` runs the full held-out evaluation once. `review-test` selects up to three held-out rallies from different matches before inference, saves the selection, and writes fresh selected-rally diagnostics. It does not replace the full-test results. Its video shows the normal camera image beside a 600-pixel court panel, predicted A/B points, and reference crosses only at the annotated frames. It preserves source FPS and decoded frame count, exports two position rows per frame, records annotation-time errors, and verifies the encoded video by decoding it.

## Configuration

### Optional review smoothing

`review-test --smooth` keeps the raw review and exports intact and adds `review_smoothed.mp4`, `player_positions_smoothed.csv`, `shot_errors_smoothed.csv`, and `metrics_smoothed.json` in each rally directory. Inference runs once per frame; a second video read renders the smoothed results. Both views use the same filtered court trajectory, with inverse projection for the camera marker. Training, full-test evaluation, and `overlay` are unchanged.

Edit the separate `review_config.yaml` to tune `smoothing.median_window` (default 5) and `smoothing.savgol_window` (default 9). These are odd integer **frame counts**, not seconds: median must be at least 1, Savgol at least 3. Savgol uses a fixed quadratic polynomial. Median 1 bypasses median cleanup; Savgol 3 fits all three samples exactly, so use at least 5 for Savgol smoothing. Windows shorten for short available segments; Savgol is skipped below three frames. Larger windows smooth more but may soften genuine turns. `--smooth --check` validates these settings before inference. Each review saves the resolved settings in `review_smoothing.json`; saved training settings are never edited.

The filter rejects nonfinite court points, positions more than 3 m outside the court rectangle, positions more than 1 m into the opposite court half, and points more than 1 m from the local coordinate-wise median. These are exploratory rejection heuristics, not calibrated confidence thresholds. It applies median cleanup followed by Savgol and never clamps coordinates to court boundaries.

Set `smoothing.gap_policy` to either:

- `conservative` (default): interpolate gaps of at most 0.2 seconds with valid points on both sides. Longer and endpoint gaps remain unavailable and their markers disappear.
- `continuous`: interpolate every interior gap and hold the nearest valid position at endpoints. A completely unavailable trajectory remains unavailable.

Smoothed CSVs retain two rows per frame and include `filter_status`, `rejection_reason`, and `gap_filled`. Status distinguishes smoothing, short/long interpolation, endpoint holds, and unavailable points. The original heatmap score remains a raw-model diagnostic. Filled markers are labelled in the video. Smoothed metrics report annotation-time errors and coverage, all-frame position coverage, filled-point counts, and rejection/status counts. Unavailable annotation-time predictions remain failures in success-rate denominators; continuous-mode errors can include guessed positions. Compare coverage as well as error before claiming improvement. Smoothing can preserve sustained wrong predictions and cannot establish foot-position or movement-measurement accuracy. These selected held-out rallies are exploratory diagnostics, not an unbiased score after tuning.

### Training settings

Edit `config.yaml` without changing Python:

- `cv.folds`, `cv.seed`: grouped CV fold count and fold shuffle seed.
- `model.input_width`, `model.input_height`, `model.gaussian_sigma`: full-frame resize and normalized Gaussian target width in heatmap cells.
- `train.epochs`, `patience`, `batch`, `lr0`, and `weight_decay`: optimization budget and AdamW settings.
- `train.freeze_backbone_epochs`, `horizontal_flip_probability`, `seed`, `device`, `workers`, and `amp`: initialization/training controls.

Input dimensions must be multiples of four. The default image input is the full RGB frame resized to 960×540 and scaled to [0,1], without cropping or ImageNet normalization. Training uses spatial cross-entropy over each channel's heatmap cells.

Checkpoint and manifest loaders reject obsolete pose artifacts instead of treating them as point-only data. Run outputs save resolved configuration, dependency versions, model contract, transfer details, and checkpoint SHA256 hashes.
