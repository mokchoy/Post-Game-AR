"""Sequential, resumable Papermill queue for the 12 settled training jobs."""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import papermill as pm
import torch

import training
import training_data as td
import training_reports


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "train_classifier.ipynb"


def job_plan() -> list[dict]:
    return [
        {"model": model, "kind": "cv", "fold": fold}
        for model in training.MODELS for fold in (1, 2, 3)
    ] + [{"model": model, "kind": "final"} for model in training.MODELS]


def safe_experiment_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError("experiment ID must use 1-64 letters, digits, dots, underscores, or hyphens")
    return value


def _lock_path(experiment_dir: Path) -> Path:
    return experiment_dir.with_name(experiment_dir.name + ".lock")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _sha256(path: Path) -> str:
    return td._sha256(path)


def _source_fingerprints() -> dict[str, str]:
    files = (NOTEBOOK, HERE / "run_training_queue.py", HERE / "training.py", HERE / "training_data.py", HERE / "classifiers.py", HERE / "training_reports.py")
    return {path.name: _sha256(path) for path in files}


def _runtime(config: dict) -> dict:
    return {
        "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
        "papermill": pm.__version__, "device": str(training._resolve_device(config)),
    }


def _estimated_training_output_bytes(cache: dict) -> int:
    rows = cache["index"]
    train_count = sum(row["official_split"] == "train" for row in rows)
    validation_count = sum(row["official_split"] == "validation" for row in rows)
    class_count = len(cache["metadata"]["class_order"])
    prediction_bytes = 6 * (train_count + validation_count) * (2048 + 64 * class_count)
    with torch.random.fork_rng(devices=[]):
        parameter_bytes = sum(
            parameter.numel() * 4
            for name in training.MODELS
            for parameter in training.classifiers.build_model(name, class_count).parameters()
        )
    return prediction_bytes + 48 * parameter_bytes + 128 * 1024**2


def _reserve_training_space(inspected: dict, existing_output_bytes: int = 0) -> int:
    config = inspected["config"]
    estimate = _estimated_training_output_bytes(inspected["feature_cache"])
    additional = max(0, estimate - existing_output_bytes)
    td.jp.ensure_capacity(config["_paths"]["root"], config["_pipeline"]["cache_limit_gb"], additional)
    return additional


def _experiment_snapshot(config: dict, source: dict, cache: dict, experiment_id: str) -> dict:
    user_config = {key: value for key, value in config.items() if not key.startswith("_")}
    pipeline_path = config["_pipeline_config_path"]
    identity_payload = {
        "experiment_id": experiment_id,
        "training_config": user_config,
        "pipeline_config_sha256": _sha256(pipeline_path),
        "data_identity": cache["identity"],
        "class_order": cache["metadata"]["class_order"],
        "source_fingerprints": _source_fingerprints(),
        "runtime": _runtime(config),
        "fold_assignments_sha256": _sha256(config["_paths"]["metadata"] / "fold_assignments.csv"),
        "class_review_sha256": _sha256(config["_paths"]["metadata"] / "class_review.csv"),
    }
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "identity": identity,
        "data_identity": cache["identity"],
        "class_order": cache["metadata"]["class_order"],
        "settings": user_config,
        "runtime": identity_payload["runtime"],
        "source_fingerprints": identity_payload["source_fingerprints"],
        "fold_assignments_sha256": identity_payload["fold_assignments_sha256"],
        "class_review_sha256": identity_payload["class_review_sha256"],
        "final_fit_epoch_budgets": {},
    }


def _check_experiment(existing_path: Path, expected: dict) -> dict:
    actual = json.loads(existing_path.read_text(encoding="utf-8"))
    if actual.get("identity") != expected["identity"]:
        raise ValueError("resume rejected: data, classes, settings, source code, or runtime fingerprint changed")
    return actual


def _write_training_log(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


@contextmanager
def temporary_python_kernel():
    """Expose the current interpreter as a temporary python3 kernelspec."""
    with tempfile.TemporaryDirectory(prefix="jester-kernel-", dir=HERE) as temporary:
        root = Path(temporary)
        spec = root / "kernels" / "python3"
        spec.mkdir(parents=True)
        runtime = root / "runtime"
        runtime.mkdir()
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python 3.14 (Jester queue)", "language": "python",
        }), encoding="utf-8")
        previous = os.environ.get("JUPYTER_PATH")
        previous_runtime = os.environ.get("JUPYTER_RUNTIME_DIR")
        os.environ["JUPYTER_PATH"] = temporary + (os.pathsep + previous if previous else "")
        os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous
            if previous_runtime is None:
                os.environ.pop("JUPYTER_RUNTIME_DIR", None)
            else:
                os.environ["JUPYTER_RUNTIME_DIR"] = previous_runtime


def run_jobs(jobs: list[dict], experiment_dir: Path, experiment: dict, execute) -> None:
    for job in jobs:
        if training.validate_job_complete(experiment_dir, experiment, job):
            print(f"SKIP {training.job_name(job)}: validated completion")
            continue
        print(f"RUN {training.job_name(job)}")
        execute(job)


def _execute_job(experiment_dir: Path, config_path: Path, identity: str, job: dict) -> None:
    name = training.job_name(job)
    attempts = experiment_dir / "attempts" / name
    attempts.mkdir(parents=True, exist_ok=True)
    prior = [int(m.group(1)) for p in attempts.glob("attempt-*.ipynb") if (m := re.fullmatch(r"attempt-(\d+)\.ipynb", p.name))]
    attempt = max(prior, default=0) + 1
    notebook_output = attempts / f"attempt-{attempt:04d}.ipynb"
    stdout_path = attempts / f"attempt-{attempt:04d}.stdout.log"
    stderr_path = attempts / f"attempt-{attempt:04d}.stderr.log"
    parameters = {
        "RUN_TRAINING": True,
        "CONFIG_PATH": str(config_path.resolve()),
        "EXPERIMENT_DIR": str(experiment_dir.resolve()),
        "EXPERIMENT_IDENTITY": identity,
        "JOB_JSON": json.dumps(job, separators=(",", ":")),
    }
    try:
        # Papermill starts and shuts down one isolated kernel per notebook invocation.
        with temporary_python_kernel(), stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            pm.execute_notebook(
                str(NOTEBOOK), str(notebook_output), parameters=parameters,
                kernel_name="python3", cwd=str(HERE), progress_bar=False, log_output=True,
                stdout_file=stdout, stderr_file=stderr,
            )
    except BaseException as exc:
        _write_training_log(attempts / f"attempt-{attempt:04d}.error.txt", f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
        raise
    if not training.validate_job_complete(experiment_dir, json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8")), job):
        raise RuntimeError(f"notebook returned without a valid completion marker: {name}")


def _cv_budgets(experiment_dir: Path, experiment: dict, inspected: dict) -> dict:
    budgets = {}
    index = inspected["feature_cache"]["index"]
    expected_train = {row["clip_id"] for row in index if row["official_split"] == "train"}
    for model in training.MODELS:
        best = []
        oof_ids = []
        for fold in (1, 2, 3):
            job = {"model": model, "kind": "cv", "fold": fold}
            if not training.validate_job_complete(experiment_dir, experiment, job):
                raise ValueError(f"cannot derive final budget; CV job incomplete: {training.job_name(job)}")
            job_dir = experiment_dir / "jobs" / training.job_name(job)
            marker = json.loads((job_dir / "training_complete.json").read_text(encoding="utf-8"))
            epoch = int(marker["best_epoch"])
            if not 1 <= epoch <= 100:
                raise ValueError(f"invalid best CV epoch for {training.job_name(job)}: {epoch}")
            best.append(epoch)
            with (job_dir / "predictions.csv").open("r", encoding="utf-8-sig", newline="") as stream:
                oof_ids.extend(row["clip_id"] for row in csv.DictReader(stream))
        if len(oof_ids) != len(set(oof_ids)) or set(oof_ids) != expected_train:
            raise ValueError(f"pooled OOF membership is incomplete for {model}")
        budgets[model] = {"median_best_cv_epoch": median_epoch_budget(best), "fold_best_epochs": best, "epochs": median_epoch_budget(best)}
    return budgets


def median_epoch_budget(best_epochs: list[int]) -> int:
    if len(best_epochs) != 3 or any(not 1 <= int(epoch) <= 100 for epoch in best_epochs):
        raise ValueError("final-fit budget needs three valid one-based CV epochs")
    return sorted(map(int, best_epochs))[1]


def _comparison_done(experiment_dir: Path, identity: str) -> bool:
    path = experiment_dir / "reports" / "comparison_complete.json"
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
        if marker.get("status") != "complete" or marker.get("identity") != identity:
            return False
        return all((experiment_dir / "reports" / name).is_file() and _sha256(experiment_dir / "reports" / name) == digest for name, digest in marker.get("sha256", {}).items())
    except Exception:
        return False


def _write_comparison_done(experiment_dir: Path, identity: str) -> None:
    report_dir = experiment_dir / "reports"
    files = [p for p in report_dir.iterdir() if p.is_file() and p.name != "comparison_complete.json"]
    _atomic_json(report_dir / "comparison_complete.json", {
        "status": "complete", "identity": identity,
        "sha256": {path.name: _sha256(path) for path in files},
    })


def _run(config_path: Path, experiment_id: str, resume: bool) -> int:
    experiment_id = safe_experiment_id(experiment_id)
    inspected = td.inspect_inputs(config_path, require_features=True)
    if inspected["blockers"]:
        raise RuntimeError("training preflight blocked:\n- " + "\n- ".join(inspected["blockers"]))
    config, cache, source = inspected["config"], inspected["feature_cache"], inspected["source"]
    experiment_dir = config["_paths"]["root"] / "training" / "experiments" / experiment_id
    snapshot = _experiment_snapshot(config, source, cache, experiment_id)
    lock = _lock_path(experiment_dir)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "identity": snapshot["identity"]}))
    except FileExistsError as exc:
        raise RuntimeError(f"another queue holds the experiment lock: {lock}") from exc
    try:
        experiment_file = experiment_dir / "experiment.json"
        if resume:
            if not experiment_file.is_file():
                raise ValueError(f"resume requires an existing experiment: {experiment_dir}")
            experiment = _check_experiment(experiment_file, snapshot)
            _reserve_training_space(inspected, td.jp._cache_bytes(experiment_dir))
        else:
            if experiment_dir.exists():
                raise FileExistsError(f"experiment already exists; use --train --resume: {experiment_dir}")
            _reserve_training_space(inspected)
            experiment_dir.mkdir(parents=True)
            experiment = snapshot
            _atomic_json(experiment_file, experiment)

        jobs = job_plan()
        run_jobs(jobs[:9], experiment_dir, experiment, lambda job: _execute_job(experiment_dir, config_path, experiment["identity"], job))

        budgets = _cv_budgets(experiment_dir, experiment, inspected)
        expected_budgets = {name: row for name, row in budgets.items()}
        if experiment.get("final_fit_epoch_budgets") and experiment["final_fit_epoch_budgets"] != expected_budgets:
            raise ValueError("saved final-fit epoch budgets differ from validated CV results")
        if not experiment.get("final_fit_epoch_budgets"):
            experiment["final_fit_epoch_budgets"] = expected_budgets
            _atomic_json(experiment_file, experiment)
        budget_path = experiment_dir / "final_fit_epoch_budgets.json"
        budget_record = {"models": expected_budgets, "identity": experiment["identity"]}
        if budget_path.exists():
            if json.loads(budget_path.read_text(encoding="utf-8")) != budget_record:
                raise ValueError("saved final-fit budget artifact conflicts with validated CV results")
        else:
            _atomic_json(budget_path, budget_record)

        run_jobs(jobs[9:], experiment_dir, experiment, lambda job: _execute_job(experiment_dir, config_path, experiment["identity"], job))

        comparison_marker = experiment_dir / "reports" / "comparison_complete.json"
        if _comparison_done(experiment_dir, experiment["identity"]):
            print("SKIP comparison reports: validated completion")
        elif comparison_marker.exists():
            raise ValueError("comparison completion marker or a completed report artifact is incompatible")
        else:
            training_reports.generate_reports(config_path, experiment_dir)
            _write_comparison_done(experiment_dir, experiment["identity"])
        print(f"COMPLETE {experiment_id}")
        return 0
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _compare_existing(config_path: Path, experiment_id: str) -> int:
    experiment_id = safe_experiment_id(experiment_id)
    inspected = td.inspect_inputs(config_path, require_features=True)
    if inspected["blockers"]:
        raise RuntimeError("comparison preflight blocked:\n- " + "\n- ".join(inspected["blockers"]))
    config, cache, source = inspected["config"], inspected["feature_cache"], inspected["source"]
    experiment_dir = config["_paths"]["root"] / "training" / "experiments" / experiment_id
    experiment_file = experiment_dir / "experiment.json"
    if not experiment_file.is_file():
        raise FileNotFoundError(f"experiment does not exist: {experiment_dir}")
    expected = _experiment_snapshot(config, source, cache, experiment_id)
    experiment = _check_experiment(experiment_file, expected)
    lock = _lock_path(experiment_dir)
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "identity": experiment["identity"], "mode": "compare"}))
    except FileExistsError as exc:
        raise RuntimeError(f"another queue holds the experiment lock: {lock}") from exc
    try:
        budgets = _cv_budgets(experiment_dir, experiment, inspected)
        if experiment.get("final_fit_epoch_budgets") != budgets:
            raise ValueError("saved final-fit epoch budgets differ from validated CV results")
        for job in job_plan():
            if not training.validate_job_complete(experiment_dir, experiment, job):
                raise ValueError(f"comparison requires completed, validated job: {training.job_name(job)}")
        marker = experiment_dir / "reports" / "comparison_complete.json"
        if _comparison_done(experiment_dir, experiment["identity"]):
            print("Comparison reports already complete.")
            return 0
        if marker.exists():
            raise ValueError("comparison marker or report artifact is incompatible; preserving saved reports")
        training_reports.generate_reports(config_path, experiment_dir)
        _write_comparison_done(experiment_dir, experiment["identity"])
        print(f"Comparison reports ready: {experiment_dir / 'reports'}")
        return 0
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check or run the sequential 12-job GRU/LSTM/DD-Net queue.")
    parser.add_argument("--config", type=Path, required=True, help="explicit training_config.json path")
    parser.add_argument("--experiment-id", required=True, help="explicit frozen run identity")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="read-only preflight; this is the default")
    mode.add_argument("--train", action="store_true", help="execute the queue")
    mode.add_argument("--compare", action="store_true", help="validate saved results and regenerate reports/CPU benchmarks only")
    parser.add_argument("--resume", action="store_true", help="resume the same compatible experiment; requires --train")
    args = parser.parse_args(argv)
    try:
        experiment_id = safe_experiment_id(args.experiment_id)
        if args.resume and not args.train:
            parser.error("--resume requires --train")
        if args.compare:
            return _compare_existing(args.config, experiment_id)
        inspected = td.inspect_inputs(args.config, require_features=True)
        print(f"Experiment: {experiment_id}")
        print("Jobs in order:")
        for index, job in enumerate(job_plan(), 1):
            print(f"  {index:02d}. {training.job_name(job)}")
        if not args.train:
            print("Mode: check only; no notebook kernel starts")
            print(f"Data clips ready: {len((inspected.get('feature_cache') or {}).get('index', []))}")
            if inspected["blockers"]:
                print("Blockers:")
                for blocker in inspected["blockers"]:
                    print(f"  - {blocker}")
                return 2
            try:
                reserve = _reserve_training_space(inspected)
                print(f"Estimated training-output reserve: {reserve / 1024**3:.2f} GiB")
            except RuntimeError as exc:
                print(f"Blocker: {exc}")
                return 2
            print("Preflight: ready")
            return 0
        if inspected["blockers"]:
            raise RuntimeError("training preflight blocked:\n- " + "\n- ".join(inspected["blockers"]))
        return _run(args.config, experiment_id, args.resume)
    except KeyboardInterrupt:
        print("Interrupted; active Papermill attempt and completed-epoch checkpoints are preserved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
