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

## Interpretation limits

The box bottom-centre is only an approximate floor reference, especially during lunges and jumps. The annotation's exact foot-midpoint definition is unconfirmed, and evaluation at shot timestamps does not validate positions between shots. The homography maps annotation image pixels to dataset court coordinates; those raw coordinates are not metres. The runner applies the existing dataset-to-metre transform before scoring or rendering positions.
