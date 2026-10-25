# Jester data pipeline

Small Python 3.14 scripts for Jester labels, fixed three-fold manifests, selected RGB frames, raw hand landmarks, and a later GRU/LSTM/DD-Net comparison. The local config points originals and completed bundles to `G:\My Drive\jester-data`; working files stay in `cache\` with a 50 GB managed limit. Google Drive streaming can use additional local disk outside that limit.

The current Qualcomm page names the three packages `20bnjester-v1-00`, `-01`, and `-02`; its instruction PDF describes split gzip/tar parts. The PDF's example archive name is older than the current page names. The pipeline streams the parts together without writing a second concatenated archive. Do not rename or edit the originals. Full package bytes are not available in this workspace, so the real archive has not yet been checked here.

## Setup

Run from this folder. Python 3.14.6 and the dependencies in `requirements.txt` are available in the implementation environment. If setting up another environment:

```powershell
py -3.14 -m pip install -r requirements.txt
```

Model weights are local inputs. The extractor never downloads them automatically. RTMLib's `Hand` solution uses the RTMDet-nano 320×320 detector and RTMPose-m hand model at 256×256. This adapter calls those supported components directly so an empty detector result stays a no-detection frame. Download and expand those two official archives only when you are ready to use that extractor:

```powershell
New-Item -ItemType Directory -Force .\models\rtmdet, .\models\rtmpose | Out-Null
if (-not (Test-Path .\models\rtmdet.zip)) { Invoke-WebRequest 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip' -OutFile .\models\rtmdet.zip }
if (-not (Test-Path .\models\rtmpose.zip)) { Invoke-WebRequest 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip' -OutFile .\models\rtmpose.zip }
Expand-Archive .\models\rtmdet.zip .\models\rtmdet
Expand-Archive .\models\rtmpose.zip .\models\rtmpose
Get-ChildItem .\models\rtmdet, .\models\rtmpose -Filter *.onnx -Recurse
```

MediaPipe's local `.task` model can be fetched explicitly:

```powershell
New-Item -ItemType Directory -Force .\models | Out-Null
if (-not (Test-Path .\models\hand_landmarker.task)) { Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile .\models\hand_landmarker.task }
```

## Archive and labels

If the packages are not already in Drive, these are the official download URLs and names. They are individual commands; none was run during implementation.

```powershell
$root = 'G:\My Drive\jester-data'
if (-not (Test-Path (Join-Path $root '20bnjester-v1-00'))) { Invoke-WebRequest 'https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester/20bnjester-v1-00' -OutFile (Join-Path $root '20bnjester-v1-00') }
if (-not (Test-Path (Join-Path $root '20bnjester-v1-01'))) { Invoke-WebRequest 'https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester/20bnjester-v1-01' -OutFile (Join-Path $root '20bnjester-v1-01') }
if (-not (Test-Path (Join-Path $root '20bnjester-v1-02'))) { Invoke-WebRequest 'https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester/20bnjester-v1-02' -OutFile (Join-Path $root '20bnjester-v1-02') }
if (-not (Test-Path (Join-Path $root 'jester-labels'))) { Invoke-WebRequest 'https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester/jester-labels' -OutFile (Join-Path $root 'jester-labels') }
```

Keep the downloaded originals. If you do not already have an extracted labels folder, extract into `G:\My Drive\jester-data\jester_labels` so `labels.csv`, `train.csv`, `validation.csv`, and `test.csv` are directly inside that folder. The pipeline intentionally does not read `test-answers.csv`.

The official labels endpoint returns a ZIP without a `.zip` suffix. Inspect its member names, then extract it while retaining the original download:

```powershell
py -3.14 -c "import zipfile; print(*zipfile.ZipFile(r'G:\My Drive\jester-data\jester-labels').namelist(), sep='\n')"
if (-not (Test-Path 'G:\My Drive\jester-data\jester_labels')) { py -3.14 -c "import zipfile; zipfile.ZipFile(r'G:\My Drive\jester-data\jester-labels').extractall(r'G:\My Drive\jester-data\jester_labels')" }
```

If the ZIP contains a subfolder, set `labels_dir` in `config.json` to the directory containing the four CSV files. Wait for Drive sync to finish before inventory or staging; the inventory detects files changing during hashing, but cannot prove that a paused upload is complete.

Check package presence, byte sizes, and SHA-256 values, then stage the parts locally:

```powershell
py -3.14 .\jester_pipeline.py --config .\config.json archive-inventory
py -3.14 .\jester_pipeline.py --config .\config.json stage-archives
```

Staging checks the 50 GB managed-cache limit and actual free disk space before copying. It retains the Drive originals and local staged parts. Extraction scans the multipart stream, then scans it again to write only selected clips; it does not concatenate or unpack every clip.

## Labels, folds, review, and pilot IDs

These commands write metadata only. Folds use all official training clips, seed 42, and stable per-class ranking. Each role is a manifest view of the same clips; official validation is reserved for final labeled evaluation and test remains unlabeled.

```powershell
py -3.14 .\jester_pipeline.py --config .\config.json validate-labels
py -3.14 .\jester_pipeline.py --config .\config.json folds --seed 42
py -3.14 .\jester_pipeline.py --config .\config.json review-template
```

Keep all 27 rows in `cache\metadata\class_review.csv`. This file controls selection for preparation and training. The current subset retains these 11 classes:

- Swiping Left, Swiping Right
- Turning Hand Clockwise, Turning Hand Counterclockwise
- Zooming In With Full Hand, Zooming Out With Full Hand
- Thumb Up, Thumb Down, Stop Sign
- No gesture, Doing other things

Set these rows to `retain`, and the other 16 to `exclude` with reason `Outside selected gesture set`. Preserve existing review notes. Both background classes must remain `retain`. Selection is not evidence that landmark quality has passed review. Preparation includes `pending` and `flag` classes unless excluded; training still blocks pending reviews. Official label files, complete manifests, and fold assignments continue to include all 27 classes. Existing folds do not need regeneration.

The review file is local cache metadata and is not tracked by Git. On a fresh cache, create it with `review-template` and apply the selection above before generating pilot IDs. Existing review tables are preserved rather than overwritten by that command.

```powershell
py -3.14 .\jester_pipeline.py --config .\config.json pilot --per-class 1000 --seed 42
```

This command saves 1,000 unique official training IDs per non-excluded class to `cache\metadata\pilot_ids.csv`: **11,000 clips** for this subset. Selection uses seed 42. The command's default remains 500 if `--per-class` is omitted. It reports insufficient selected-class counts instead of reducing the sample. Before regenerating a different pilot, move existing `pilot_ids.csv` and `pilot_ids.report.json` into a dated backup folder; generation refuses to overwrite a different ID list. Without exclusions, the original 27-class behavior remains available. Pilot membership is independent of CV fold assignment. Frame extraction validates the declared per-class count, including earlier 200-per-class pilots, after filtering excluded classes.

Write the combined clip manifest after generating folds, pilot IDs, or review decisions. Rerun it after frame extraction to refresh frame integrity and status:

```powershell
py -3.14 .\jester_pipeline.py --config .\config.json manifest
```

## Selected frames and landmarks

This is the opt-in pilot frame command. It reads only the previously saved pilot IDs and was not run here.

```powershell
py -3.14 .\jester_pipeline.py --config .\config.json extract-frames --pilot
```

Rerun the same extraction command after an interruption. Completed final clips are skipped. Staged clips with valid completion markers, matching archive identity and frame count, and verified frame hashes are promoted without re-extraction. Unmarked or invalid staged clips are re-extracted. Windows access/sharing errors during promotion get four attempts with brief delays; persistent failures preserve staging and appear in `cache\metadata\frame_extraction_failures.csv` while other clips continue. Resume still scans archive metadata to validate the selection. Run only one frame extraction process at a time.

These are separate, explicit later operations. The `--ids-file` command filters the CV assignment list to non-excluded training classes. Frame and landmark extraction, reports, and benchmarks apply the same filter to supplied ID lists. Empty filtered selections are rejected. While exclusions are active, `--all` and explicit unlabeled test IDs are rejected because test gestures are unknown. Without exclusions, `--all` retains its original behavior of extracting every official split.

```powershell
py -3.14 .\jester_pipeline.py --config .\config.json extract-frames --ids-file .\cache\metadata\fold_assignments.csv
```

The current comparison uses **1,000 training clips per class plus every official validation clip from the 11 retained classes**: **11,000 + 6,428 = 17,428 clips**. Export the combined selection after applying the class review and generating the 1,000-per-class pilot:

```powershell
$selectedClasses = @(Import-Csv .\cache\metadata\class_review.csv |
    Where-Object { $_.compatibility_decision -in @('retain', 'flag') } |
    Select-Object -ExpandProperty class_label)
$trainingIds = @(Import-Csv .\cache\metadata\pilot_ids.csv)
$validationIds = @(Import-Csv .\cache\metadata\clip_manifest.csv |
    Where-Object { $_.official_split -eq 'validation' -and $_.class_label -in $selectedClasses })
($trainingIds + $validationIds) |
    Select-Object clip_id, class_label, official_split |
    Sort-Object { [int]$_.clip_id } |
    Export-Csv .\cache\metadata\comparison_ids.csv -NoTypeInformation -Encoding utf8
```

Existing extracted frames can be reused. If selected frames are missing, use `extract-frames --ids-file .\cache\metadata\comparison_ids.csv`. Keep the original `labeled_ids.csv`; it still contains all 133,349 labeled clips and is not the comparison selection.

Use this command for the smaller MediaPipe extraction. Compatible completed landmark clips are skipped:

```powershell
py -3.14 .\jester_pipeline.py --config .\config.json extract-landmarks --extractor mediapipe --ids-file .\cache\metadata\comparison_ids.csv --model .\models\hand_landmarker.task
```

Refresh frame status only for the comparison IDs before preparing inputs. This retains the complete 27-class manifest and existing fold assignments, and preserves saved frame statuses outside the selection:

```powershell
py -3.14 .\jester_pipeline.py --config .\config.json manifest --ids-file .\cache\metadata\comparison_ids.csv
```

These are manual commands; changing metadata does not change an extraction process already running with the old list. Finish or stop that process before starting another cache writer. Reports and extractor benchmarks remain training-only; use `pilot_ids.csv` for them, not the combined train/validation selection. Omitting `manifest --ids-file` retains the full frame-refresh behavior.

Use the exact `.onnx` paths printed by `Get-ChildItem` above. The same pilot list is passed to either extractor:

```powershell
py -3.14 .\jester_pipeline.py --config .\config.json extract-landmarks --extractor rtmlib --ids-file .\cache\metadata\pilot_ids.csv --rtm-det-model '.\models\rtmdet\<detector-file>.onnx' --rtm-pose-model '.\models\rtmpose\<pose-file>.onnx'
py -3.14 .\jester_pipeline.py --config .\config.json extract-landmarks --extractor mediapipe --ids-file .\cache\metadata\pilot_ids.csv --model .\models\hand_landmarker.task
```

For an explicit full **training-feature** run after the pilot, first extract those training clips with the training `--ids-file` command above, then pass the same fold assignment CSV to reuse the exact CV clip IDs. These commands were not run:

```powershell
py -3.14 .\jester_pipeline.py --config .\config.json extract-landmarks --extractor rtmlib --ids-file .\cache\metadata\fold_assignments.csv --rtm-det-model '.\models\rtmdet\<detector-file>.onnx' --rtm-pose-model '.\models\rtmpose\<pose-file>.onnx'
py -3.14 .\jester_pipeline.py --config .\config.json extract-landmarks --extractor mediapipe --ids-file .\cache\metadata\fold_assignments.csv --model .\models\hand_landmarker.task
```

Both extractors retain all candidates and one row per source frame. Cache coordinates are image-normalized `(x, y)` in the documented wrist, thumb, index, middle, ring, pinky 21-joint order. They preserve translation, apparent scale, direction, frame indices, 12 FPS timestamps, dimensions, detection counts/masks, and supported scores or handedness. No mirroring, wrist centering, scale normalization, smoothing, or interpolation is applied. RTMLib's public detector returns boxes without detection scores; its joint scores are retained. MediaPipe's handedness score is stored separately. These score types are not comparable.

RTMLib extraction supports explicit NVIDIA GPU acceleration on Windows. Use `onnxruntime-gpu==1.24.4` instead of `onnxruntime`, plus the installed CUDA-enabled PyTorch to load compatible CUDA/cuDNN DLLs. Do not install both ONNX Runtime packages together. To configure another environment, first install the normal requirements, then run `py -3.14 -m pip uninstall -y onnxruntime` and `py -3.14 -m pip install onnxruntime-gpu==1.24.4`. This workspace has already been configured. Run:

```powershell
$detModel = (Get-ChildItem .\models\rtmdet -Filter *.onnx -Recurse | Select-Object -First 1).FullName
$poseModel = (Get-ChildItem .\models\rtmpose -Filter *.onnx -Recurse | Select-Object -First 1).FullName
py -3.14 .\jester_pipeline.py --config .\config.json extract-landmarks --extractor rtmlib --device cuda --ids-file .\cache\metadata\pilot_ids.csv --rtm-det-model "$detModel" --rtm-pose-model "$poseModel"
```

Omitting `--device` retains CPU extraction. CUDA requests fail if either model falls back to CPU. Backend and device are recorded in cache identities; incompatible existing RTMLib caches are preserved and reported rather than overwritten or reused. MediaPipe extraction and the `benchmark` command remain CPU-only. GPU extraction speed is separate from the CPU latency comparison.

RTMLib 0.0.16 declares the CPU `onnxruntime` distribution as a dependency, so `pip check` reports it missing when using the GPU replacement. Actual CUDA detector and pose inference were verified in this environment with GPU runtime 1.24.4. Do not install the CPU distribution alongside it to silence that metadata warning; reinstalling the normal `requirements.txt` also requires repeating the GPU replacement steps above.

By default the extractor uses `pilot_ids.csv`. A supplied CSV needs a `clip_id` header; official validation IDs can be selected explicitly, and each feature file records its official split. Unlabeled test IDs are accepted only when no classes are excluded. CV fold assignments remain training-only manifest metadata.

## Reports, benchmark, backup

```powershell
py -3.14 .\jester_pipeline.py --config .\config.json report --extractor rtmlib --ids-file .\cache\metadata\pilot_ids.csv
py -3.14 .\jester_pipeline.py --config .\config.json report --extractor mediapipe --ids-file .\cache\metadata\pilot_ids.csv
py -3.14 .\jester_pipeline.py --config .\config.json benchmark --ids-file .\cache\metadata\pilot_ids.csv --model .\models\hand_landmarker.task --rtm-det-model '.\models\rtmdet\<detector-file>.onnx' --rtm-pose-model '.\models\rtmpose\<pose-file>.onnx'
py -3.14 .\jester_pipeline.py --config .\config.json compare
py -3.14 .\jester_pipeline.py --config .\config.json bundle --extractor all --cloud-copy
```

Reports include per-class coverage, failed clips and missing-frame gaps, multiple-hand counts, five deterministic clip overlays per class plus flagged failures, class review fields, diagnostic wrist-step summaries, storage, elapsed time, and pilot-based full-data estimates. Low coverage on background clips is not an automatic failure. Wrist movement statistics are diagnostics, not landmark-accuracy evidence. CPU benchmarking separates local file read, image decode, and preprocessing/detection/inference, with median and p95, CPU/runtime/thread details, and an untimed warm-up. It does not estimate classifier accuracy or continuous-recognition reliability.

`bundle --extractor all --cloud-copy` packages completed frame clips, valid feature caches, manifests, and reports in one ZIP rather than syncing individual JPEGs. It keeps local results and verifies the mounted-path copy by size and SHA-256. Confirm that Google Drive finishes syncing before treating the backup as uploaded; Google Drive's own streaming cache is outside the managed limit.

## Verification

```powershell
py -3.14 -m unittest discover -s tests -v
py -3.14 .\jester_pipeline.py --config .\config.json validate-labels
```

The unit tests use tiny synthetic labels, JPEGs, cache files, multipart tar bytes, and CPU optimizer steps. Real archive extraction, extractor model inference, the comparison extraction, full extraction, and real-data classifier training are intentionally not part of verification. Python 3.14 works here, so no Python 3.12 fallback is needed.

## GRU, LSTM, and DD-Net comparison

Training uses Python 3.14 and a separate dependency file. The extractor environment stays unchanged:

```powershell
py -3.14 -m pip install -r .\training-requirements.txt
```

The current `training_config.json` selects `mediapipe` and sets `ids_file` to `cache/metadata/comparison_ids.csv`. Keep the same frozen extractor, class review, saved folds, and `sequence_length` for all models. Choosing this extractor and class set does not certify landmark quality. The default sequence length is 32; it must be even and at least 8.

`ids_file` is optional and resolves relative to the training configuration. When present, preparation and all three classifiers use exactly those retained train/validation IDs, and require each retained class in every CV fold and in official validation. Unknown, duplicate, empty, mismatched, excluded-class, and test IDs are rejected. Unselected raw frames and landmarks are not required. Omit the field or set it to null to use every retained official training and validation clip as before. The selection file's SHA-256 is part of the shared-cache identity; changing it requires a new input cache and experiment ID. The fixed existing fold assignments are reused; sampling does not guarantee equal selected counts within each fold.

Training uses four background data-loading workers (`"num_workers": 4` in `training_config.json`), reused across epochs. Each worker opens the shared feature cache read-only. Set this to `0` to load batches in the training process. Validation and model jobs remain sequential. Worker count is part of experiment settings, so changing it requires a new experiment ID. More workers are not guaranteed to improve throughput for these small landmark inputs.

After selected official train and validation clips have complete frame and landmark caches, build the single shared memory-mapped `(N,T,43)` input cache. This reads neither official test clips nor `test-answers.csv`:

```powershell
py -3.14 .\prepare_training_inputs.py --config .\training_config.json --check
py -3.14 .\prepare_training_inputs.py --config .\training_config.json --build
```

Input preparation verifies selected raw frames and landmarks before freezing the shared cache. Queue jobs validate that cache's hashes and its manifest, fold, class, selection, extractor, and sequence-length identities without reopening RGB frames or raw landmark files. Re-run input preparation to verify changed raw sources; training does not silently switch frozen inputs. Ambiguous matching caches block preflight.

After changing the class review, rebuild the shared inputs and use a new experiment ID. The review and retained class mapping change cache and experiment identities, so old checkpoints cannot resume as the new subset. Compatible raw frame and landmark caches can be reused; keep existing feature caches and results. All three classifiers derive their output count from the retained mapping, giving 11 outputs for this subset. Live-camera false triggers still require separate evaluation, including unwanted gestures and idle movement.

Landmark extraction counts managed-cache usage once, then accounts for committed outputs while checking free disk space per clip. Run one cache-writing operation at a time; parallel writers need shared space reservations.

Check queue readiness and the exact 12-job order. Check mode starts no notebook kernels:

```powershell
py -3.14 .\run_training_queue.py --config .\training_config.json --experiment-id jester-11class-1000-mediapipe-v1
```

Preflight also reserves managed cache and free disk for prediction exports, checkpoints, and reports.

Run once only after preflight is clear. Use resume after an error or interruption; it requires the same data, classes, settings, source files, and runtime:

```powershell
py -3.14 .\run_training_queue.py --config .\training_config.json --experiment-id jester-11class-1000-mediapipe-v1 --train
py -3.14 .\run_training_queue.py --config .\training_config.json --experiment-id jester-11class-1000-mediapipe-v1 --train --resume
```

The queue runs three CV folds per model, derives each final-fit epoch count from the median best fold epoch, then fits and evaluates three final models. It creates comparison reports and the fixed 100-clip CPU benchmark after all jobs complete. Regenerate those reports from saved outputs without training:

```powershell
py -3.14 .\run_training_queue.py --config .\training_config.json --experiment-id jester-11class-1000-mediapipe-v1 --compare
```

Inspect every saved class probability for one official validation clip:

```powershell
py -3.14 .\inspect_probabilities.py --experiment .\cache\training\experiments\jester-11class-1000-mediapipe-v1 --clip-id 12345
```

The queue creates a temporary `python3` kernelspec for the active Python interpreter, then removes it. Papermill needs local ZeroMQ sockets for each fresh kernel. Probabilities are softmax outputs, not calibrated confidence. DD-Net source and license are documented in `DDNET_REFERENCE.md` and `licenses\DD-Net-LICENSE.txt`.

## Live webcam

With a completed MediaPipe experiment and the local `models/hand_landmarker.task` file, run:

```powershell
py -3.14 .\webcam.py --experiment .\cache\training\experiments\jester-11class-1000-mediapipe-v1
```

Use `--model gru|lstm|ddnet`, `--camera 0`, or `--window-seconds 3` to set startup options. Press `1`, `2`, or `3` to switch models; `Q` or Escape exits. The preview and inference use unmirrored frames. Displayed probabilities are uncalibrated, and rolling windows can include frames from adjacent gestures.
