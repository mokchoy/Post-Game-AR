# Jester Data Pipeline with Three-Fold Cross-Validation

## 1. Goal and implementation boundary

Compare **GRU, LSTM, and DD-Net** using one shared Jester pipeline and **three stratified folds**.

Use 21 image-normalized `(x, y)` landmarks. Accept either hand and retain every compatible one-hand gesture, including both background classes.

Immediate implementation covers data preparation, landmark extraction commands, manifests, reports, and small verification checks. **Do not automatically launch bulk downloads, the full pilot, model training, or full-dataset extraction.**

## 2. Storage and dataset organization

| Data | Location |
|---|---|
| Original archives, official metadata, completed backups | Mounted Google Drive |
| Active archive staging and extracted frames | Local cache |
| Active landmark files and training inputs | Local cache |
| Completed landmark bundles, reports, checkpoints | Google Drive, with local working copies |

- Use a separate gesture project. Preserve Fire-Analysis-v2 and the reference TD-GCN checkout.
- Configure `cloud_root`, `local_cache_root`, and `cache_limit_gb`. Default the managed cache limit to **50 GB**.
- Check space before staging or extraction. Stop before exceeding the limit without automatically increasing it.
- Follow current official download instructions. Avoid a second full concatenated archive and initially extract only pilot clips.
- Preserve original downloads and record sizes and hashes.
- Bundle completed outputs for Drive. Avoid syncing millions of individual JPEGs.
- Delete only reproducible, project-owned staging files. Retain unique local outputs until cloud upload is confirmed.
- Monitor actual disk space because Drive’s streaming cache sits outside the managed limit.

Preserve original clip IDs, labels, frame filenames, and official splits. Use manifests rather than duplicate class folders.

Each clip record identifies its official split, class, frame location, integrity status, extraction status, and class-inclusion decision. Sort frames numerically. Record missing or corrupt frames without silently renumbering them.

Jester’s released timeline remains **12 FPS**, with **100-pixel image height**. Preserve source images and original timing. [Official dataset instructions](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/qualcomm-ai-research-jester-download-instructions-v2.pdf)

## 3. Cross-validation and final evaluation

**Replace the previous permanent 90%/10% training/development split with three-fold cross-validation.**

- Assign official training clips to three class-stratified folds using seed **42**.
- Save one fixed fold assignment per clip. All three architectures use identical assignments.
- For each run, train on two folds and validate on the remaining fold.
- Keep every frame, augmentation, and future window within its source clip’s fold.
- Store landmark features once. Folds are manifest views, not separate dataset copies.
- Keep official validation separate as the final labeled holdout.
- Keep the unlabeled official test set separate.
- Do not claim subject-independent evaluation without suitable actor identifiers.

For later training:

- Initialize model weights, optimizer state, and training-derived preprocessing independently for each fold.
- Fit normalization or other learned preprocessing using that fold’s training portion only.
- Use held-out folds for checkpoint selection and model-development decisions.
- Report each fold’s metrics, their mean and standard deviation, and pooled out-of-fold predictions.
- Save clip IDs, labels, predictions, probabilities, fold IDs, and run configuration for paired comparisons.
- Treat CV results as development evidence. Reserve official validation for final assessment.

The initial comparison uses **one training seed per configuration**:

- Three architectures × three folds = **nine CV fits**.
- Refit each finalized architecture on all official training clips = **three additional fits**.
- Total: **12 fits per selected set of configurations**, excluding any additional tuning.

Set final refit duration from each model’s CV results, using the median best-checkpoint epoch. Evaluate the three final models on official validation without feeding those results back into tuning.

## 4. Extraction pilot and shared artifacts

Select **200 unique clips per original class** from official training data, using seed 42. Save the **5,400 pilot IDs** before extraction.

Run **RTMlib Hand and MediaPipe Hand Landmarker on identical clips**.

- Process every frame in order and reset tracking between clips.
- Map both extractors to the same documented 21-joint ordering.
- Normalize coordinates against original image dimensions.
- Preserve wrist movement, apparent scale, and image-space direction.
- Do not mirror, wrist-center, scale-normalize, smooth, or interpolate the raw cache.
- Preserve all detected hand candidates and explicitly record frames without detections.
- Save compressed per-clip numeric outputs with frame indices, timestamps, dimensions, detection masks, and available scores.
- Record extractor versions, settings, runtime backend, and model-weight hashes.
- Resume only from complete outputs matching the current input and extraction configuration.
- Do not treat confidence scores from different extractors as equivalent.

Review representative training examples for one-hand compatibility. Always retain **“No gesture”** and **“Doing other things.”** Document ambiguous classes and extraction weaknesses rather than automatically removing difficult classes.

Generate a pilot report containing:

- Per-class detection coverage, missing-frame gaps, extraction failures, and multiple-hand counts.
- Overlays for five fixed random clips per class and flagged failures.
- Visual instability and hand-identity-switching observations.
- CPU extraction latency, including preprocessing and detection, with median and p95.
- Cache usage, processing time, and estimated full-processing requirements.
- Recommended extractor and class-inclusion table.

Pilot decisions use official training data only. Freeze the extractor and class list before model comparison. Numerical jitter remains a diagnostic rather than ground-truth landmark accuracy.

## 5. Later comparison, verification, and limits

After the pilot, extract retained classes in bounded batches using the frozen extractor configuration.

Compare:

- **GRU and LSTM:** identical input preprocessing and classifier heads, with parameter counts reported.
- **DD-Net:** preserve its joint-distance, two-scale motion, and temporal-convolution design. Document adaptations. [Published implementation](https://github.com/fandulu/DD-Net)

DD-Net uses derived features, so describe this as a comparison of complete classifiers. Final model sizes, temporal duration, and missing-data treatment belong to the subsequent training milestone.

Report accuracy, per-class precision/recall, macro F1, confusion matrices, model size, and CPU latency. Include both background classes and retain extraction failures in evaluation accounting.

Verify the pipeline with small fixtures and smoke checks covering:

- Reproducible, stratified, disjoint fold assignments.
- Identical folds and sample IDs across architectures.
- No frame or window leakage between roles.
- Correct frame ordering, timestamps, joint ordering, and normalization.
- Explicit missing detections and safe interrupted-run recovery.
- Cache limits and preservation of completed outputs.

Default to Python **3.14**, with **3.12** as the documented fallback if dependency checks fail. Training targets GPU; webcam inference targets CPU.

**Deferred:** model implementations, bulk processing, training, and webcam inference. No 3D, TD-GCN, or RGB baseline is included.

**Limits:** no implementation or training has run. Three-fold variability does not establish continuous webcam reliability; webcam testing remains qualitative.
