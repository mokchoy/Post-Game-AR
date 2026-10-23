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
py -3.14 .\jester_pipeline.py --config .\config.json pilot --per-class 200 --seed 42
```

The pilot command saves 200 unique official training IDs per original class to `cache\metadata\pilot_ids.csv` (expected 5,400). It reports insufficient class counts instead of reducing the sample. Pilot membership is independent of CV fold assignment.

Write the combined clip manifest after generating folds, pilot IDs, or review decisions. Rerun it after frame extraction to refresh frame integrity and status:

```powershell
py -3.14 .\jester_pipeline.py --config .\config.json manifest
```

## Selected frames and landmarks

This is the opt-in pilot frame command. It reads only the previously saved pilot IDs and was not run here.

```powershell
py -3.14 .\jester_pipeline.py --config .\config.json extract-frames --pilot
```

These are separate, explicit later operations. The `--ids-file` command extracts all official training clips from the CV assignment list. `--all` extracts every official split, including validation and unlabeled test.

```powershell
py -3.14 .\jester_pipeline.py --config .\config.json extract-frames --ids-file .\cache\metadata\fold_assignments.csv
py -3.14 .\jester_pipeline.py --config .\config.json extract-frames --all
```

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

By default the extractor uses `pilot_ids.csv`. A supplied CSV needs a `clip_id` header; official validation or test IDs can be selected explicitly when needed, and each feature file records its official split. CV fold assignments remain training-only manifest metadata.

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

The unit tests use tiny synthetic labels, JPEGs, cache files, multipart tar bytes, and CPU optimizer steps. Real archive extraction, model inference, the 5,400-clip pilot, full extraction, and real-data classifier training are intentionally not part of verification. Python 3.14 works here, so no Python 3.12 fallback is needed.

## GRU, LSTM, and DD-Net comparison

Training uses Python 3.14 and a separate dependency file. The extractor environment stays unchanged:

```powershell
py -3.14 -m pip install -r .\training-requirements.txt
```

Set `extractor` in `training_config.json` only after class review and extractor comparison. Keep the same frozen extractor, class review, saved folds, and `sequence_length` for all models. The default sequence length is 32; it must be even and at least 8. The current config intentionally leaves `extractor` unset.

Training uses four background data-loading workers (`"num_workers": 4` in `training_config.json`), reused across epochs. Each worker opens the shared feature cache read-only. Set this to `0` to load batches in the training process. Validation and model jobs remain sequential. Worker count is part of experiment settings, so changing it requires a new experiment ID. More workers are not guaranteed to improve throughput for these small landmark inputs.

After selected official train and validation clips have complete frame and landmark caches, build the single shared memory-mapped `(N,T,43)` input cache. This reads neither official test clips nor `test-answers.csv`:

```powershell
py -3.14 .\prepare_training_inputs.py --config .\training_config.json --check
py -3.14 .\prepare_training_inputs.py --config .\training_config.json --build
```

Input preparation verifies raw frames and landmarks before freezing the shared cache. Queue jobs validate that cache's hashes and its manifest, fold, class, extractor, and sequence-length identities without reopening RGB frames or raw landmark files. Re-run input preparation to verify changed raw sources; training does not silently switch frozen inputs. Ambiguous matching caches block preflight.

Landmark extraction counts managed-cache usage once, then accounts for committed outputs while checking free disk space per clip. Run one cache-writing operation at a time; parallel writers need shared space reservations.

Check queue readiness and the exact 12-job order. Check mode starts no notebook kernels:

```powershell
py -3.14 .\run_training_queue.py --config .\training_config.json --experiment-id jester-compare-v1
```

Preflight also reserves managed cache and free disk for prediction exports, checkpoints, and reports.

Run once only after preflight is clear. Use resume after an error or interruption; it requires the same data, classes, settings, source files, and runtime:

```powershell
py -3.14 .\run_training_queue.py --config .\training_config.json --experiment-id jester-compare-v1 --train
py -3.14 .\run_training_queue.py --config .\training_config.json --experiment-id jester-compare-v1 --train --resume
```

The queue runs three CV folds per model, derives each final-fit epoch count from the median best fold epoch, then fits and evaluates three final models. It creates comparison reports and the fixed 100-clip CPU benchmark after all jobs complete. Regenerate those reports from saved outputs without training:

```powershell
py -3.14 .\run_training_queue.py --config .\training_config.json --experiment-id jester-compare-v1 --compare
```

Inspect every saved class probability for one official validation clip:

```powershell
py -3.14 .\inspect_probabilities.py --experiment .\cache\training\experiments\jester-compare-v1 --clip-id 12345
```

The queue creates a temporary `python3` kernelspec for the active Python interpreter, then removes it. Papermill needs local ZeroMQ sockets for each fresh kernel. Probabilities are softmax outputs, not calibrated confidence. DD-Net source and license are documented in `DDNET_REFERENCE.md` and `licenses\DD-Net-LICENSE.txt`.
