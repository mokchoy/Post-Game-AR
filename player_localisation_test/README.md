# Pretrained player localisation pilot

This pilot runs `torchvision.models.detection.fasterrcnn_resnet50_fpn` with `FasterRCNN_ResNet50_FPN_Weights.COCO_V1` on the Korea Open 2019 `rally_002` clip. It uses the same Faster R-CNN detector family as MonoTrack, but a separate official COCO_V1 checkpoint; it does not load MonoTrack's checkpoint or use MonoTrack tracking.

## Local checkpoint

Place the file at:

`C:\Users\ILLEGEAR\Documents\Post-Game-AR\player_localisation_test\checkpoints\fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`

The runner does not download weights. It requires the SHA-256 to begin with `258fb6c6`, sets Torch Hub's directory to this pilot folder, then uses the official COCO_V1 model builder, normalization, and internal resize/preprocessing. Frames enter the model as RGB float tensors in `[0,1]`; the model applies its standard transform. If the file is absent, `check` still validates the inputs, geometry, timing, and compiled Torchvision operations, then reports the exact missing path. Real inference remains pending.

## Commands

From this folder:

```powershell
python runner.py check
python self_check.py
python runner.py run
```

`check` reads video metadata and annotations, validates each exact `clip_frame_num` against `clip_time_seconds` and 25 FPS, checks homography row 3, validates the checkpoint hash, and exercises NMS and ROIAlign. When the checkpoint is present, it also loads the official model and runs one frame. `run` repeats input/checkpoint validation, runs frame 0 first, then processes only this rally. It uses CUDA when available and otherwise CPU.

Each successful run creates a new timestamped directory under `outputs/`. Existing run folders are never overwritten. It contains:

- `player_positions.csv`: two A/B rows per decoded frame, including unavailable rows.
- `shot_errors.csv`: one A and one B evaluation row per annotated shot (58 total).
- `metrics.json`: combined and per-player coverage, pixel/metre error statistics, and within-distance rates.
- `review.mp4`: source boxes and bottom-centres beside the court panel; shot reference markers appear only on their annotated frames.

Predictions come only from person boxes with confidence at least 0.3. The box bottom-centre is mapped from native video pixels through the annotation-to-dataset homography and the existing dataset-to-metre transform. A is fixed to the far half (`y < 6.7 m`); B is fixed to the near half. The highest-confidence eligible detection is selected independently on each half. Candidates more than 1 m outside the court are rejected. There is no tracking, smoothing, interpolation, or clamping. Missing positions remain unavailable. Distance statistics use available matched predictions; coverage and within-0.5 m/within-1 m percentages use all 58 eligible references, with missing predictions counted as failures.

Pixel errors are measured in native 1920×1080 video pixels; annotation reference coordinates are scaled by 1.5 before comparison. The original 1280×720 reference coordinates are also preserved in `shot_errors.csv`.

## Matched comparison with the point-only model

Run these commands with system Python in PowerShell:

```powershell
Set-Location 'C:\Users\ILLEGEAR\Documents\Post-Game-AR\player_localisation_test'
$reviewDir = 'C:\Users\ILLEGEAR\Documents\Post-Game-AR\player_keypoint_test\runs\final_cv_5f_s42_1e4a7705f081\selected_rally_reviews\review_test_20261016T015452Z'
python compare_review.py --review-dir $reviewDir --check
```

`--check` validates the local checkpoint hashes, final-run/split/manifest identity, selected clips and video metadata, saved positions and annotation-time errors, A/B mappings, projections, and saved smoothing settings. It recomputes smoothing on the saved trajectories to check compatibility. It checks checkpoint loading and tiny Torchvision operations, without running either model's forward pass, rendering real clips, training, or downloading weights. Prepared JPEG images are not needed for this comparison.

For bounded synthetic validation:

```powershell
python self_check.py
python compare_self_check.py
```

The comparison self-check uses three eight-frame artificial clips and stub models. It retains PNG/PDF previews, synthetic reviews and a report under a fresh `outputs/run_comparison_self_check*` directory. Its timings and errors are **synthetic checks, not real model performance**.

To generate the full comparison yourself:

```powershell
python compare_review.py --review-dir $reviewDir
```

This uses the exact saved Korea `002`, Denmark `028`, and Fuzhou `015` clips: 1,529 source frames and 114 player references. It loads the saved final point-model checkpoint and the local Faster R-CNN checkpoint, benchmarking one model at a time on the device in the saved keypoint configuration. Each clip gets ten unmeasured warm-up calls before every frame is timed. The models use batch size 1, FP32 and no autocast; CUDA is synchronized around each timed inference pipeline. Timing includes preprocessing, transfers, forward pass, prediction decoding/selection and court projection, but excludes loading, video decoding, rendering and encoding. Smoothing computation, including inverse projection, is timed separately. The pipeline reports pooled and per-rally mean/median/p95 milliseconds per frame and FPS, rather than video-encoding speed.

Detector accuracy uses freshly computed positions; keypoint accuracy reuses validated saved raw/smoothed exports. The keypoint model reruns only for timing; its videos and accuracy artifacts are not regenerated. The three selected clips share frozen references and pixel-centre scaling, with A **near** for Denmark and Fuzhou. Detector confidence (0.3), person filtering, highest-score selection per court half, and the 1 m court margin remain unchanged. The original `runner.py run` preserves its original single-rally conventions.

The comparison uses the saved `review_smoothing.json`, not the current editable YAML. Both methods use the same rejection, median, Savgol and gap policy. With this review the windows are 5/9 frames and the gap policy is continuous. Filled positions are flagged and shown with hollow markers in the detector video and triangles in the trajectory figures. No trajectory is clamped to the court.

Each full comparison creates a fresh directory under `outputs/` containing:

- `comparison.md` with tables, embedded PNG graphs and links to matching vector PDFs; `comparison.json` with full pooled, per-player, per-rally and per-rally/player metrics.
- `provenance.json`, including source artifact/video hashes, both checkpoint hashes, runtime versions, detector settings, saved selection and smoothing settings; `references.json` snapshots the frozen references, mappings and geometry.
- `timings.csv` and `smoothing_times.csv` for reproducible timing summaries.
- `detector/<rally-id>/` with raw/smoothed position and shot-error CSVs, metrics and verified review videos.
- `graphs/` with pixel/metre error distributions, pooled/per-rally coverage and success rates, runtime plots, and per-rally court trajectories and annotation-time error plots, each as PNG and PDF.

Error statistics (mean, median, p95, Euclidean RMSE) use available predictions. Coverage and within-0.5 m/within-1 m rates include every eligible reference, so unavailable positions fail success tests. Error-distribution graphs use that same denominator and can plateau below 100%. Pooled statistics are calculated from individual error rows. All-frame position coverage and filled-position counts are reported separately. References are plotted as isolated crosses; there is no continuous ground-truth trajectory between annotation times.

Three selected rallies provide an exploratory comparison of an unadapted COCO detector against a badminton-trained heatmap model. They do not establish architecture superiority or comprehensive test accuracy. Heatmap scores are not presence probabilities, so score values are not compared with detector confidence. Continuous smoothing uses future samples, interpolates long gaps and holds endpoints; high smoothed coverage is not proof of observed positions or live tracking latency.

## Interpretation limits

The box bottom-centre is only an approximate floor reference, especially during lunges and jumps. The annotation's exact foot-midpoint definition is unconfirmed, and evaluation at shot timestamps does not validate positions between shots. The homography maps annotation image pixels to dataset court coordinates; those raw coordinates are not metres. The runner applies the existing dataset-to-metre transform before scoring or rendering positions.
