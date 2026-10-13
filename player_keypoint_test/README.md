# YOLO11 player floor-reference experiment

This experiment uses only cleaned_rallies/set-1. It never edits the baseline player-localisation pilot, cleaned footage, annotations, or cleanup reports. Generated files, weights, review state, and training outputs stay in this folder.

The target is one proxy floor-reference point per player box. Annotation points are in the 1280×720 image reference system and are scaled to each clip's native dimensions. At inference, points are scaled back to annotation space and passed through the existing court geometry. These labels do not establish a foot-midpoint definition or calibration accuracy.

## Setup

The project venv reuses the installed CUDA PyTorch:

~~~powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

Ultralytics is pinned to 8.4.175. The check command downloads only the official yolo11n-pose.pt asset into weights/ if it is absent, checks one-keypoint model construction and weight transfer, and runs a one-epoch synthetic smoke training. It does not train on the badminton set.

## Workflow

Run from this folder:

~~~powershell
.\.venv\Scripts\python.exe runner.py check
.\.venv\Scripts\python.exe runner.py prepare
.\.venv\Scripts\python.exe runner.py review
.\.venv\Scripts\python.exe runner.py split
.\.venv\Scripts\python.exe runner.py train-cv
.\.venv\Scripts\python.exe runner.py train-final --run-id <CV_RUN_ID>
.\.venv\Scripts\python.exe runner.py evaluate --run-id <FINAL_RUN_ID> --split test
.\.venv\Scripts\python.exe runner.py overlay --run-id <FINAL_RUN_ID> --match <MATCH_FOLDER> --game 1 --rally 1
~~~

The config option selects another YAML file for check, prepare, split, or train-cv. The folds option explicitly overrides the YAML fold count for split or train-cv; the requested count is validated and never reduced. train-final uses the chosen CV run's saved resolved settings.

The prepare command checks all rally-level rules, then uses pretrained person boxes at confidence >=0.30 by default. One failing annotated frame excludes its whole rally. data/candidate/exclusions.csv records reasons, rallies.jsonl records each rally's automatic status, and manifest.jsonl contains one provenance row per extracted annotated frame. Preliminary counts are recomputed from these inputs before box association.

The OpenCV reviewer shows the normal frame, A/B proposal boxes, and A/B reference points. Press A to accept a sample, R to reject its rally, M to reject its whole match, S to save and quit, or Q to quit and resume later. A rally rejection excludes that rally from the frozen set. Unsampled frames that passed automatic checks need no manual approval. split refuses to freeze the dataset while any selected sample is pending.

split reserves a deterministic 20% of accepted matches with seed 42 and records the immutable held-out set. Development fold assignments are grouped by whole match and saved per fold-count/CV-seed combination. Training and hyperparameter changes do not alter the held-out set.

train-cv initializes a fresh nano model from the same pose checkpoint for each fold, saves the resolved settings and transfer report, and evaluates each fold's validation-selected best.pt on its held-out matches. train-final trains on all development matches for the rounded median selected epoch count and uses last.pt; the test set is not used for training or selection. evaluate --split test records the final test evaluation once.

Inference assigns A/B only from each rally's validated fixed court-half mapping. A/B are not image classes, and predictions are never matched to ground truth by proximity. Multiple eligible detections on one half are treated as ambiguous/missing. Error summaries use available predictions; coverage and distance-success denominators include missing predictions as failures.

## Outputs

- data/candidate/: exact annotated-frame images, YOLO pose labels, automatic manifest, exclusion ledger, and five review samples per retained match.
- splits/split.json: frozen match test reservation and exact grouped fold assignments.
- runs/: one directory per CV/final run, with settings, versions, weight hashes, reports, player positions, and overlay exports.
- smoke/: the tiny synthetic training check, separate from experiment data.
