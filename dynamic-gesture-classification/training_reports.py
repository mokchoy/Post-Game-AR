"""Comparison exports and fixed CPU inference benchmark."""

from __future__ import annotations

import csv
import json
import platform
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import classifiers
import training
import training_data as td


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_prediction(path: Path, class_order: list[str]) -> list[dict]:
    rows = _read_csv(path)
    expected_probability_fields = ["prob__" + label for label in class_order]
    if not rows:
        raise ValueError(f"prediction export is empty: {path}")
    if [name for name in rows[0] if name.startswith("prob__")] != expected_probability_fields:
        raise ValueError(f"probability columns or order differ from frozen classes: {path}")
    seen = set()
    for row in rows:
        cid = row.get("clip_id", "")
        if not cid or cid in seen:
            raise ValueError(f"missing or duplicate prediction clip ID in {path}: {cid}")
        seen.add(cid)
        probs = np.asarray([float(row[name]) for name in expected_probability_fields], dtype=np.float64)
        if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1) or not np.isclose(probs.sum(), 1.0, atol=1e-5):
            raise ValueError(f"invalid probability vector for clip {cid} in {path}")
        best = int(np.argmax(probs))
        if row["predicted_class"] != class_order[best] or row["correct"] != str(row["true_class"] == row["predicted_class"]).lower():
            raise ValueError(f"prediction export labels disagree with saved probabilities: {cid}")
    return rows


def _score_rows(rows: list[dict], class_order: list[str]) -> dict:
    index = {name: i for i, name in enumerate(class_order)}
    if any(row["true_class"] not in index or row["predicted_class"] not in index for row in rows):
        raise ValueError("prediction export has a class outside the frozen mapping")
    return training._metrics(
        np.asarray([index[row["true_class"]] for row in rows]),
        np.asarray([index[row["predicted_class"]] for row in rows]), class_order,
    )


def _write_confusion(path: Path, matrix: list[list[float]], class_order: list[str], normalize: bool) -> None:
    _write_csv(path, ["true_class", *class_order], [
        {"true_class": class_order[i], **{class_order[j]: matrix[i][j] for j in range(len(class_order))}}
        for i in range(len(class_order))
    ])


def _curve(experiment: Path, model: str) -> None:
    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(11, 4))
    for fold in (1, 2, 3):
        history = json.loads((experiment / "jobs" / f"{model}_cv_fold_{fold}" / "history.json").read_text(encoding="utf-8"))["epochs"]
        epochs = [row["epoch"] for row in history]
        loss_ax.plot(epochs, [row["train_loss"] for row in history], label=f"fold {fold} train")
        loss_ax.plot(epochs, [row["development_loss"] for row in history], linestyle="--", label=f"fold {fold} development")
        f1_ax.plot(epochs, [row["development_macro_f1"] for row in history], label=f"fold {fold}")
    loss_ax.set(xlabel="Epoch", ylabel="Cross-entropy loss")
    f1_ax.set(xlabel="Epoch", ylabel="Development macro F1")
    loss_ax.legend(fontsize=7)
    f1_ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(experiment / "reports" / f"learning_curves_{model}.png", dpi=150)
    plt.close(fig)


def _confusion_figure(path: Path, raw: list[list[int]], normalized: list[list[float]], class_order: list[str], title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, data, caption in zip(axes, (raw, normalized), ("Counts", "Row-normalized"), strict=True):
        image = ax.imshow(data, interpolation="nearest", cmap="Blues", aspect="auto")
        ax.set_title(caption)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticks(range(len(class_order)), class_order, rotation=90, fontsize=6)
        ax.set_yticks(range(len(class_order)), class_order, fontsize=6)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def benchmark_cpu(config_path: Path, experiment_dir: Path, inspected: dict) -> dict:
    config, cache = inspected["config"], inspected["feature_cache"]
    index = cache["index"]
    rows = sorted((row for row in index if row["official_split"] == "train"), key=lambda row: (int(row["clip_id"]), row["clip_id"]))
    if len(rows) < 100:
        raise ValueError("CPU benchmark needs 100 retained official training clips")
    rows = rows[:100]
    arrays = [np.asarray(cache["features"][int(row["feature_index"])], dtype=np.float32).copy() for row in rows]
    cpu_inputs = [torch.from_numpy(array).unsqueeze(0) for array in arrays]
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    output = []
    try:
        for model_name in training.MODELS:
            checkpoint = torch.load(experiment_dir / "jobs" / f"{model_name}_final" / "final.pt", map_location="cpu", weights_only=False)
            model = classifiers.build_model(model_name, len(cache["metadata"]["class_order"]))
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            if model_name == "ddnet":
                prepared = [classifiers.ddnet_features(x) for x in cpu_inputs]
                model_call = lambda i: model.forward_prepared(*prepared[i])
            else:
                model_call = lambda i: model(cpu_inputs[i])

            def combined_call(i: int):
                x = torch.from_numpy(arrays[i]).unsqueeze(0)
                return model(x)

            for fn in (model_call, combined_call):
                with torch.no_grad():
                    for i in range(20):
                        fn(i % 100)
            model_times, combined_times = [], []
            with torch.no_grad():
                for i in range(100):
                    start = time.perf_counter()
                    model_call(i)
                    model_times.append((time.perf_counter() - start) * 1000)
                    start = time.perf_counter()
                    combined_call(i)
                    combined_times.append((time.perf_counter() - start) * 1000)
            output.append({
                "model": model_name,
                "clip_ids": [row["clip_id"] for row in rows],
                "model_only_median_ms": statistics.median(model_times),
                "model_only_p95_ms": _percentile(model_times, 95),
                "feature_plus_inference_median_ms": statistics.median(combined_times),
                "feature_plus_inference_p95_ms": _percentile(combined_times, 95),
                "model_only_boundary": "CPU forward on shared float32 (T,43); DD-Net derived streams prepared before timer",
                "combined_boundary": "CPU forward on shared float32 (T,43); DD-Net distance/motion derivation included",
            })
    finally:
        torch.set_num_threads(old_threads)
    report = {
        "models": output,
        "clip_ids": [row["clip_id"] for row in rows],
        "batch_size": 1,
        "cpu_threads": 1,
        "warmup_calls_per_boundary": 20,
        "hardware": {"processor": platform.processor(), "platform": platform.platform(), "python": platform.python_version()},
        "runtime": {"torch": torch.__version__, "numpy": np.__version__},
        "runtime_backend": "PyTorch CPU",
        "thread_settings": {"torch_num_threads": 1, "torch_num_interop_threads": torch.get_num_interop_threads()},
        "note": "Landmark extraction, cloud transfer, and continuous recognition are excluded. Probabilities are not calibrated confidence.",
    }
    (experiment_dir / "reports").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "reports" / "cpu_benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_csv(experiment_dir / "reports" / "cpu_benchmark.csv", list(output[0]), output)
    return report


def generate_reports(config_path: Path, experiment_dir: Path) -> dict:
    inspected = td.inspect_inputs(config_path, require_features=True)
    if inspected["blockers"]:
        raise RuntimeError("reporting preflight blocked:\n- " + "\n- ".join(inspected["blockers"]))
    cache = inspected["feature_cache"]
    class_order = cache["metadata"]["class_order"]
    experiment = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    if experiment.get("data_identity") != cache["identity"] or experiment.get("class_order") != class_order:
        raise ValueError("comparison input identity or class mapping changed")
    reports = experiment_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    index = cache["index"]
    expected_val = {row["clip_id"] for row in index if row["official_split"] == "validation"}
    expected_train = {row["clip_id"] for row in index if row["official_split"] == "train"}
    true_label_by_id = {row["clip_id"]: row["class_label"] for row in index}
    summary = []
    all_cv = {}
    final_rows = {}
    per_class_rows = []
    for model in training.MODELS:
        folds, cv_rows = [], []
        for fold in (1, 2, 3):
            job_dir = experiment_dir / "jobs" / f"{model}_cv_fold_{fold}"
            metrics = json.loads((job_dir / "metrics.json").read_text(encoding="utf-8"))
            rows = _read_prediction(job_dir / "predictions.csv", class_order)
            if any(int(row.get("fold", 0)) != fold for row in rows):
                raise ValueError(f"OOF fold identity mismatch for {model} fold {fold}")
            expected_fold = {row["clip_id"] for row in index if row["official_split"] == "train" and row["fold"] == str(fold)}
            if {row["clip_id"] for row in rows} != expected_fold:
                raise ValueError(f"OOF prediction clip IDs differ from saved assignments: {model} fold {fold}")
            if any(row["true_class"] != true_label_by_id[row["clip_id"]] for row in rows):
                raise ValueError(f"OOF labels differ from saved manifest: {model} fold {fold}")
            folds.append(metrics["macro_f1"])
            cv_rows.extend(rows)
            per_class_rows.extend({
                "model": model, "role": f"cv_fold_{fold}", "class": name, **values,
                "detection_coverage": metrics["per_class_detection_coverage"][name]["coverage"],
                "extraction_failures": metrics["per_class_extraction_failures"][name],
            } for name, values in metrics["per_class"].items())
            _write_confusion(reports / f"cv_confusion_{model}_fold_{fold}.csv", metrics["confusion_raw"], class_order, False)
            _write_confusion(reports / f"cv_confusion_normalized_{model}_fold_{fold}.csv", metrics["confusion_row_normalized"], class_order, True)
        if {row["clip_id"] for row in cv_rows} != expected_train or len(cv_rows) != len(expected_train):
            raise ValueError(f"pooled OOF predictions are incomplete or duplicated: {model}")
        oof_metrics = _score_rows(cv_rows, class_order)
        all_cv[model] = cv_rows
        final_job = experiment_dir / "jobs" / f"{model}_final"
        holdout = _read_prediction(final_job / "predictions.csv", class_order)
        if {row["clip_id"] for row in holdout} != expected_val:
            raise ValueError(f"holdout prediction IDs differ from official validation: {model}")
        if any(row["true_class"] != true_label_by_id[row["clip_id"]] for row in holdout):
            raise ValueError(f"holdout labels differ from saved manifest: {model}")
        holdout_metrics = json.loads((final_job / "metrics.json").read_text(encoding="utf-8"))
        final_rows[model] = holdout
        per_class_rows.extend({
            "model": model, "role": "official_validation_holdout", "class": name, **values,
            "detection_coverage": holdout_metrics["per_class_detection_coverage"][name]["coverage"],
            "extraction_failures": holdout_metrics["per_class_extraction_failures"][name],
        } for name, values in holdout_metrics["per_class"].items())
        params = holdout_metrics["trainable_parameters"]
        weight_bytes = holdout_metrics["inference_weight_bytes"]
        summary.append({
            "model": model,
            "cv_fold_1_macro_f1": folds[0], "cv_fold_2_macro_f1": folds[1], "cv_fold_3_macro_f1": folds[2],
            "cv_mean_macro_f1": statistics.mean(folds), "cv_sample_std_macro_f1": statistics.stdev(folds),
            "pooled_oof_macro_f1": oof_metrics["macro_f1"], "holdout_macro_f1": holdout_metrics["macro_f1"],
            "holdout_top1_accuracy": holdout_metrics["top1_accuracy"], "holdout_macro_precision": holdout_metrics["macro_precision"],
            "holdout_macro_recall": holdout_metrics["macro_recall"], "trainable_parameters": params,
            "inference_weight_bytes": weight_bytes,
            "holdout_detection_coverage": holdout_metrics["detection_coverage"],
            "holdout_extraction_failures": holdout_metrics["extraction_failures"],
            "holdout_background_to_gesture_numerator": holdout_metrics["background_to_gesture"]["numerator"],
            "holdout_background_to_gesture_denominator": holdout_metrics["background_to_gesture"]["denominator"],
            "holdout_gesture_to_background_numerator": holdout_metrics["gesture_to_background"]["numerator"],
            "holdout_gesture_to_background_denominator": holdout_metrics["gesture_to_background"]["denominator"],
        })
        _write_csv(reports / f"oof_predictions_{model}.csv", list(cv_rows[0]), sorted(cv_rows, key=lambda row: (int(row["clip_id"]), row["clip_id"])))
        _write_csv(reports / f"holdout_predictions_{model}.csv", list(holdout[0]), sorted(holdout, key=lambda row: (int(row["clip_id"]), row["clip_id"])))
        oof_metrics_full = oof_metrics | {"class_order": class_order, "role": "pooled_out_of_fold", "model": model}
        (reports / f"oof_metrics_{model}.json").write_text(json.dumps(oof_metrics_full, indent=2) + "\n", encoding="utf-8")
        _write_confusion(reports / f"oof_confusion_{model}.csv", oof_metrics["confusion_raw"], class_order, False)
        _write_confusion(reports / f"holdout_confusion_{model}.csv", holdout_metrics["confusion_raw"], class_order, False)
        _write_confusion(reports / f"holdout_confusion_normalized_{model}.csv", holdout_metrics["confusion_row_normalized"], class_order, True)
        _confusion_figure(reports / f"holdout_confusion_{model}.png", holdout_metrics["confusion_raw"], holdout_metrics["confusion_row_normalized"], class_order, f"{model.upper()} official validation")
        _curve(experiment_dir, model)
    _write_csv(reports / "comparison_summary.csv", list(summary[0]), summary)
    _write_csv(reports / "per_class_metrics.csv", ["model", "role", "class", "precision", "recall", "f1", "support", "detection_coverage", "extraction_failures"], per_class_rows)
    (reports / "comparison_summary.md").write_text(
        "# GRU / LSTM / DD-Net comparison\n\n"
        "CV fold averages, pooled OOF scores, and official validation holdout scores are separate. Softmax outputs are not calibrated confidence.\n\n"
        "| Model | CV mean macro F1 | CV sample SD | Pooled OOF macro F1 | Holdout macro F1 | Holdout accuracy | Parameters | Weight bytes |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(
            f"| {row['model'].upper()} | {row['cv_mean_macro_f1']:.4f} | {row['cv_sample_std_macro_f1']:.4f} | {row['pooled_oof_macro_f1']:.4f} | {row['holdout_macro_f1']:.4f} | {row['holdout_top1_accuracy']:.4f} | {row['trainable_parameters']} | {row['inference_weight_bytes']} |"
            for row in summary
        ) + "\n",
        encoding="utf-8",
    )
    benchmark = benchmark_cpu(config_path, experiment_dir, inspected)
    return {"summary": summary, "cpu_benchmark": benchmark, "report_dir": str(reports)}
