"""Shared training, checkpoint recovery, evaluation, and prediction export."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import classifiers
import training_data as td


MODELS = ("gru", "lstm", "ddnet")


class LandmarkDataset(torch.utils.data.Dataset):
    """Open the frozen feature map inside each worker rather than copying it."""

    def __init__(self, features, positions, labels):
        self.path = str(features.filename) if isinstance(features, np.memmap) else None
        self.features = None if self.path else features
        self.positions = positions
        self.labels = labels

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        if self.features is None:
            self.features = np.load(self.path, mmap_mode="r", allow_pickle=False)
        return np.array(self.features[self.positions[index]], dtype=np.float32, copy=True), self.labels[index]


def _training_loader(features, positions, labels, batches, num_workers):
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    return torch.utils.data.DataLoader(
        LandmarkDataset(features, positions, labels), batch_sampler=batches,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        # Worker seeds must not consume model or checkpointed shuffle RNG state.
        generator=torch.Generator().manual_seed(0),
        **({"multiprocessing_context": "spawn"} if num_workers else {}),
    )


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


def _atomic_torch_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
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


def _hash_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def batch_sizes(sample_count: int, batch_size: int = 64) -> list[int]:
    if sample_count < 1 or batch_size < 2:
        raise ValueError("training requires at least one sample and batch_size >= 2")
    sizes = [batch_size] * (sample_count // batch_size)
    remainder = sample_count % batch_size
    if remainder:
        sizes.append(remainder)
    if sizes[-1] == 1:
        if len(sizes) == 1:
            raise ValueError("a singleton training set cannot train DD-Net batch normalization")
        sizes[-2] -= 1
        sizes[-1] = 2
    return sizes


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, class_order: list[str]) -> dict:
    count = len(class_order)
    confusion = np.zeros((count, count), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred, strict=True):
        confusion[int(actual), int(predicted)] += 1
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    tp = confusion.diagonal()
    precision = np.divide(tp, predicted_count, out=np.zeros(count, dtype=np.float64), where=predicted_count > 0)
    recall = np.divide(tp, support, out=np.zeros(count, dtype=np.float64), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(count, dtype=np.float64), where=(precision + recall) > 0)
    normalized = np.divide(confusion, support[:, None], out=np.zeros_like(confusion, dtype=np.float64), where=support[:, None] > 0)
    backgrounds = {class_order.index(name) for name in ("No gesture", "Doing other things") if name in class_order}
    is_bg = np.isin(y_true, list(backgrounds))
    pred_bg = np.isin(y_pred, list(backgrounds))
    return {
        "sample_count": int(len(y_true)),
        "macro_f1": float(f1.mean()),
        "top1_accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(class_order)
        },
        "confusion_raw": confusion.tolist(),
        "confusion_row_normalized": normalized.tolist(),
        "background_to_gesture": {"numerator": int((is_bg & ~pred_bg).sum()), "denominator": int(is_bg.sum())},
        "gesture_to_background": {"numerator": int((~is_bg & pred_bg).sum()), "denominator": int((~is_bg).sum())},
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _marker_valid(path: Path, identity: str, expected_files: tuple[str, ...]) -> dict | None:
    try:
        marker = _read_json(path)
        if marker.get("status") != "complete" or marker.get("identity") != identity:
            return None
        for name in expected_files:
            file = path.parent / name
            if not file.is_file() or marker.get("sha256", {}).get(name) != _sha256(file):
                return None
        return marker
    except Exception:
        return None


def _write_marker(path: Path, identity: str, file_names: tuple[str, ...], details: dict | None = None) -> dict:
    marker = {
        "status": "complete", "identity": identity,
        "sha256": {name: _sha256(path.parent / name) for name in file_names},
    } | (details or {})
    _atomic_json(path, marker)
    return marker


def _seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    return generator


def _rng_state(shuffle: torch.Generator) -> dict:
    value = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "shuffle": shuffle.get_state(),
    }
    if torch.cuda.is_available():
        value["torch_cuda"] = torch.cuda.get_rng_state_all()
    return value


def _restore_rng(state: dict, shuffle: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    shuffle.set_state(state["shuffle"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _cpu_state(model: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _resolve_device(config: dict) -> torch.device:
    value = config.get("device", "auto")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _load_features(config_path: Path) -> tuple[dict, dict, np.ndarray, list[str]]:
    inspected = td.inspect_inputs(config_path, require_features=True)
    if inspected["blockers"]:
        raise RuntimeError("training preflight blocked:\n- " + "\n- ".join(inspected["blockers"]))
    cache = inspected["feature_cache"]
    class_order = cache["metadata"]["class_order"]
    return inspected["config"], inspected["source"], cache, class_order


def _role_rows(cache: dict, kind: str, fold: int | None = None) -> tuple[list[int], np.ndarray, list[str]]:
    selected = []
    for row in cache["index"]:
        if kind == "train" and row["official_split"] == "train":
            selected.append(row)
        elif kind == "validation" and row["official_split"] == "validation":
            selected.append(row)
    if kind == "cv_train":
        selected = [row for row in cache["index"] if row["official_split"] == "train" and row["fold"] != str(fold)]
    elif kind == "cv_development":
        selected = [row for row in cache["index"] if row["official_split"] == "train" and row["fold"] == str(fold)]
    selected.sort(key=lambda row: (int(row["clip_id"]), row["clip_id"]))
    positions = [int(row["feature_index"]) for row in selected]
    return positions, np.asarray([row["class_label"] for row in selected], dtype=object), [row["clip_id"] for row in selected]


def _class_ids(labels: np.ndarray, class_order: list[str]) -> np.ndarray:
    mapping = {name: i for i, name in enumerate(class_order)}
    unknown = sorted(set(labels.tolist()) - set(mapping))
    if unknown:
        raise ValueError(f"input labels are outside frozen class mapping: {unknown}")
    return np.asarray([mapping[name] for name in labels], dtype=np.int64)


def _predict(model: nn.Module, positions: list[int], cache: dict, labels: np.ndarray, class_order: list[str], device: torch.device, batch_size: int = 256) -> tuple[dict, dict]:
    model.eval()
    truth, predictions, probabilities = [], [], []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start:start + batch_size]
            x = np.asarray(cache["features"][batch_positions], dtype=np.float32)
            logits = model(torch.from_numpy(x).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
                raise ValueError("model exported invalid softmax probabilities")
            probabilities.append(probs)
        p = np.concatenate(probabilities) if probabilities else np.empty((0, len(class_order)), dtype=np.float32)
    y_true = _class_ids(labels, class_order)
    y_pred = p.argmax(axis=1) if len(p) else np.empty(0, dtype=np.int64)
    prediction = {"true": y_true, "predicted": y_pred, "probabilities": p}
    return prediction, _metrics(y_true, y_pred, class_order)


def _write_predictions(path: Path, ids: list[str], labels: np.ndarray, prediction: dict, class_order: list[str], model_id: str, config_id: str, fold: int | None = None) -> None:
    probability_columns = ["prob__" + name for name in class_order]
    fields = ["clip_id", "true_class", "predicted_class", "correct", "model_id", "config_id"]
    if fold is not None:
        fields.append("fold")
    fields.extend(probability_columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i, cid in enumerate(ids):
                true_name = str(labels[i])
                pred_name = class_order[int(prediction["predicted"][i])]
                row = {
                    "clip_id": cid, "true_class": true_name, "predicted_class": pred_name,
                    "correct": str(true_name == pred_name).lower(), "model_id": model_id, "config_id": config_id,
                }
                if fold is not None:
                    row["fold"] = fold
                row.update({probability_columns[j]: repr(float(prediction["probabilities"][i, j])) for j in range(len(class_order))})
                writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


def _validate_predictions(path: Path, ids: list[str], labels: np.ndarray, class_order: list[str], model_id: str, config_id: str, fold: int | None = None) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames or []
    probability_columns = ["prob__" + name for name in class_order]
    if [field for field in fields if field.startswith("prob__")] != probability_columns:
        raise ValueError("prediction export probability columns do not match saved class order")
    if [row.get("clip_id") for row in rows] != ids or [row.get("true_class") for row in rows] != labels.tolist():
        raise ValueError("prediction export is missing, duplicating, or reordering expected clips")
    for row in rows:
        if row.get("model_id") != model_id or row.get("config_id") != config_id:
            raise ValueError("prediction export model or configuration identity mismatch")
        if fold is not None and row.get("fold") != str(fold):
            raise ValueError("prediction export fold identity mismatch")
        probs = np.asarray([float(row[name]) for name in probability_columns], dtype=np.float64)
        if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1) or not np.isclose(probs.sum(), 1.0, atol=1e-5):
            raise ValueError(f"invalid probability vector for clip {row.get('clip_id')}")
        predicted = class_order[int(np.argmax(probs))]
        if row.get("predicted_class") != predicted or row.get("correct") != str(row["true_class"] == predicted).lower():
            raise ValueError(f"prediction export disagrees with probabilities for clip {row.get('clip_id')}")


def _save_metrics(path: Path, metrics: dict, class_order: list[str], config_id: str, model_name: str, model: nn.Module, weights_path: Path, cache: dict, ids: list[str]) -> None:
    index_by_id = {row["clip_id"]: row for row in cache["index"]}
    rows = [index_by_id[cid] for cid in ids]
    per_class_coverage = {}
    per_class_failures = {}
    for label in class_order:
        class_rows = [row for row in rows if row["class_label"] == label]
        frames = sum(int(row["source_frame_count"]) for row in class_rows)
        valid_frames = sum(round(float(row["detection_coverage"]) * int(row["source_frame_count"])) for row in class_rows)
        per_class_coverage[label] = {"valid_frames": valid_frames, "denominator_frames": frames, "coverage": valid_frames / frames if frames else None}
        per_class_failures[label] = sum(row.get("raw_extraction_status", "complete") != "complete" for row in class_rows)
    metrics.update({
        "class_order": class_order,
        "model_id": model_name,
        "config_id": config_id,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "inference_weight_bytes": weights_path.stat().st_size,
        "detection_coverage": float(np.mean([float(row["detection_coverage"]) for row in rows])) if rows else 0.0,
        "extraction_failures": int(sum(row.get("raw_extraction_status", "complete") != "complete" for row in rows)),
        "detection_coverage_denominator_frames": int(sum(int(row["source_frame_count"]) for row in rows)),
        "per_class_detection_coverage": per_class_coverage,
        "per_class_extraction_failures": per_class_failures,
    })
    _atomic_json(path, metrics)


def _save_epoch_history(path: Path, history: list[dict]) -> None:
    _atomic_json(path, {"epochs": history})


def _checkpoint_payload(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_state: dict, best_score: float, best_epoch: int, no_improvement: int, history: list[dict], shuffle: torch.Generator, identity: str, job: dict, elapsed_seconds: float) -> dict:
    return {
        "schema_version": 1,
        "identity": identity,
        "job": job,
        "model_state": _cpu_state(model),
        "best_model_state": best_state,
        "optimizer_state": optimizer.state_dict(),
        "completed_epoch": epoch,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "no_improvement": no_improvement,
        "history": history,
        "rng": _rng_state(shuffle),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "elapsed_seconds": elapsed_seconds,
    }


def _fit(model_name: str, train_positions: list[int], train_labels: np.ndarray, cache: dict, class_order: list[str], config: dict, job: dict, job_identity: str, job_dir: Path, final_epochs: int | None = None) -> dict:
    is_cv = job["kind"] == "cv"
    if is_cv and final_epochs is not None:
        raise ValueError("CV epochs are selected by early stopping")
    if not is_cv and (final_epochs is None or final_epochs < 1 or final_epochs > 100):
        raise ValueError("final-fit epoch budget must be in [1, 100]")
    training_marker = job_dir / "training_complete.json"
    required = ("last.pt", "history.json", "best.pt") if is_cv else ("last.pt", "history.json", "final.pt")
    existing = _marker_valid(training_marker, job_identity, required)
    if existing:
        checkpoint_path = job_dir / ("best.pt" if is_cv else "final.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return {"training": existing, "checkpoint": checkpoint_path, "best_epoch": int(existing["best_epoch"]), "history": _read_json(job_dir / "history.json")["epochs"]}
    if training_marker.exists():
        raise ValueError(f"incompatible or corrupt completed training marker: {training_marker}")

    device = _resolve_device(config)
    seed = int(config.get("seed", 42))
    shuffle = _seed(seed)
    model = classifiers.build_model(model_name, len(class_order)).to(device).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.001)))
    y = _class_ids(train_labels, class_order)
    batch_plan = batch_sizes(len(train_positions), int(config.get("batch_size", 64)))
    last_path = job_dir / "last.pt"
    best_path = job_dir / "best.pt"
    final_path = job_dir / "final.pt"
    history: list[dict] = []
    start_epoch = 1
    elapsed_base = 0.0
    best_score = -1.0
    best_epoch = 0
    best_state = _cpu_state(model)
    no_improvement = 0
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("identity") != job_identity or state.get("job") != job:
            raise ValueError(f"checkpoint is incompatible with this experiment: {last_path}")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        history = state["history"]
        completed = int(state["completed_epoch"])
        if len(history) != completed or any(int(item["epoch"]) != i + 1 for i, item in enumerate(history)):
            raise ValueError("checkpoint history does not match completed epochs")
        best_state = state["best_model_state"]
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        no_improvement = int(state["no_improvement"])
        elapsed_base = float(state.get("elapsed_seconds", 0.0))
        start_epoch = completed + 1
        _save_epoch_history(job_dir / "history.json", history)
        _restore_rng(state["rng"], shuffle)
        if is_cv:
            _atomic_torch_save(best_path, {"identity": job_identity, "model_state": best_state, "best_epoch": best_epoch, "best_score": best_score})
    else:
        _save_epoch_history(job_dir / "history.json", history)

    target_epochs = int(config.get("epochs", 100)) if is_cv else int(final_epochs)
    eval_positions, eval_labels, eval_ids = ([], np.empty(0, dtype=object), [])
    if is_cv:
        eval_positions, eval_labels, eval_ids = _role_rows(cache, "cv_development", int(job["fold"]))
        if not eval_positions:
            raise ValueError("CV development fold is empty")

    batches = []
    loader = _training_loader(cache["features"], train_positions, y, batches, config.get("num_workers", 0))
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, target_epochs + 1):
            if is_cv and no_improvement >= int(config.get("patience", 10)):
                break
            model.train()
            permutation = torch.randperm(len(train_positions), generator=shuffle).tolist()
            cursor = 0
            batches.clear()
            for size in batch_plan:
                batches.append(permutation[cursor:cursor + size])
                cursor += size
            losses, samples = 0.0, 0
            for xb, yb in loader:
                size = len(yb)
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = nn.functional.cross_entropy(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip_norm", 1.0)), error_if_nonfinite=True)
                optimizer.step()
                losses += float(loss.detach()) * size
                samples += size
            item = {"epoch": epoch, "train_loss": losses / max(samples, 1), "development_loss": None, "development_macro_f1": None}
            if is_cv:
                predicted, metrics = _predict(model, eval_positions, cache, eval_labels, class_order, device)
                item["development_loss"] = _evaluation_loss(model, eval_positions, eval_labels, cache, class_order, device)
                item["development_macro_f1"] = metrics["macro_f1"]
                if metrics["macro_f1"] > best_score:
                    best_score = metrics["macro_f1"]
                    best_epoch = epoch
                    best_state = _cpu_state(model)
                    no_improvement = 0
                    _atomic_torch_save(best_path, {"identity": job_identity, "model_state": best_state, "best_epoch": best_epoch, "best_score": best_score})
                else:
                    no_improvement += 1
            else:
                best_epoch = epoch
                best_score = float("nan")
                best_state = _cpu_state(model)
            history.append(item)
            _save_epoch_history(job_dir / "history.json", history)
            elapsed = elapsed_base + time.perf_counter() - started
            _atomic_torch_save(last_path, _checkpoint_payload(model, optimizer, epoch, best_state, best_score, best_epoch, no_improvement, history, shuffle, job_identity, job, elapsed))
            if is_cv and no_improvement >= int(config.get("patience", 10)):
                break

    finally:
        del loader

    if not is_cv:
        _atomic_torch_save(final_path, {"identity": job_identity, "model_state": _cpu_state(model), "epochs": target_epochs})
    if not is_cv:
        best_epoch = target_epochs
    reached_cap = bool(is_cv and len(history) == int(config.get("epochs", 100)) and history[-1]["epoch"] == int(config.get("epochs", 100)))
    _save_epoch_history(job_dir / "history.json", history)
    files = required
    marker = _write_marker(training_marker, job_identity, files, {
        "best_epoch": best_epoch, "best_macro_f1": best_score if is_cv else None,
        "completed_epochs": len(history), "reached_epoch_cap": reached_cap,
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "weight_bytes": (best_path if is_cv else final_path).stat().st_size,
        "elapsed_seconds": elapsed_base + time.perf_counter() - started,
    })
    checkpoint_path = best_path if is_cv else final_path
    return {"training": marker, "checkpoint": checkpoint_path, "best_epoch": best_epoch, "history": history}


def _evaluation_loss(model: nn.Module, positions: list[int], labels: np.ndarray, cache: dict, class_order: list[str], device: torch.device) -> float:
    model.eval()
    y = _class_ids(labels, class_order)
    loss_sum, seen = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(positions), 256):
            batch_positions = positions[start:start + 256]
            x = torch.from_numpy(np.asarray(cache["features"][batch_positions], dtype=np.float32)).to(device)
            target = torch.from_numpy(y[start:start + len(batch_positions)]).to(device)
            loss_sum += float(nn.functional.cross_entropy(model(x), target, reduction="sum"))
            seen += len(batch_positions)
    return loss_sum / max(seen, 1)


def _job_identity(experiment: dict, job: dict) -> str:
    return _hash_json({"experiment_identity": experiment["identity"], "job": job})


def job_name(job: dict) -> str:
    if job["kind"] == "cv":
        return f"{job['model']}_cv_fold_{int(job['fold'])}"
    return f"{job['model']}_final"


def _prediction_eval(model: nn.Module, job: dict, cache: dict, class_order: list[str], config: dict, experiment: dict, job_dir: Path, identity: str, weight_file: Path) -> dict:
    marker_path = job_dir / "evaluation_complete.json"
    required = ("metrics.json", "predictions.csv")
    existing = _marker_valid(marker_path, identity, required)
    if existing:
        return existing
    if marker_path.exists():
        raise ValueError(f"incompatible or corrupt completed evaluation: {marker_path}")
    if job["kind"] == "cv":
        positions, labels, ids = _role_rows(cache, "cv_development", int(job["fold"]))
    else:
        positions, labels, ids = _role_rows(cache, "validation")
    if not positions:
        raise ValueError("evaluation role is empty")
    device = _resolve_device(config)
    prediction, metrics = _predict(model, positions, cache, labels, class_order, device)
    metrics.update({
        "role": "cv_development" if job["kind"] == "cv" else "official_validation_holdout",
        "fold": job.get("fold"),
        "elapsed_training_seconds": _read_json(job_dir / "training_complete.json").get("elapsed_seconds"),
        "epochs": _read_json(job_dir / "training_complete.json").get("completed_epochs"),
    })
    _save_metrics(job_dir / "metrics.json", metrics, class_order, experiment["identity"], job["model"], model, weight_file, cache, ids)
    _write_predictions(job_dir / "predictions.csv", ids, labels, prediction, class_order, job["model"], experiment["identity"], job.get("fold"))
    _validate_predictions(job_dir / "predictions.csv", ids, labels, class_order, job["model"], experiment["identity"], job.get("fold"))
    return _write_marker(marker_path, identity, required, {"clip_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()})


def run_notebook_job(config_path: str, experiment_dir: str, job_json: str, experiment_identity: str) -> dict:
    config_path, experiment_dir = Path(config_path).resolve(), Path(experiment_dir).resolve()
    job = json.loads(job_json) if isinstance(job_json, str) else job_json
    config, source, cache, class_order = _load_features(config_path)
    experiment = _read_json(experiment_dir / "experiment.json")
    if experiment.get("identity") != experiment_identity or experiment.get("data_identity") != cache["identity"] or experiment.get("class_order") != class_order:
        raise ValueError("experiment identity, data cache, or class mapping changed")
    if job.get("model") not in MODELS or job.get("kind") not in {"cv", "final"} or (job["kind"] == "cv" and job.get("fold") not in (1, 2, 3)):
        raise ValueError(f"invalid queue job: {job}")
    if job["kind"] == "cv":
        train_positions, train_labels, _ = _role_rows(cache, "cv_train", int(job["fold"]))
    else:
        train_positions, train_labels, _ = _role_rows(cache, "train")
    if not train_positions:
        raise ValueError("training role is empty")
    name = job_name(job)
    job_dir = experiment_dir / "jobs" / name
    job_dir.mkdir(parents=True, exist_ok=True)
    identity = _job_identity(experiment, job)
    complete = _marker_valid(job_dir / "evaluation_complete.json", identity, ("metrics.json", "predictions.csv"))
    if complete:
        return {"job": name, "status": "already_complete"}
    final_epochs = experiment["final_fit_epoch_budgets"][job["model"]]["epochs"] if job["kind"] == "final" else None
    trained = _fit(job["model"], train_positions, train_labels, cache, class_order, config, job, identity, job_dir, final_epochs)
    payload = torch.load(trained["checkpoint"], map_location="cpu", weights_only=False)
    model = classifiers.build_model(job["model"], len(class_order))
    model.load_state_dict(payload.get("model_state", payload))
    model.to(_resolve_device(config))
    _prediction_eval(model, job, cache, class_order, config, experiment, job_dir, identity, trained["checkpoint"])
    return {"job": name, "status": "complete", "best_epoch": trained["best_epoch"]}


def validate_job_complete(experiment_dir: Path, experiment: dict, job: dict) -> bool:
    identity = _job_identity(experiment, job)
    job_dir = experiment_dir / "jobs" / job_name(job)
    weights = "best.pt" if job["kind"] == "cv" else "final.pt"
    if _marker_valid(job_dir / "training_complete.json", identity, ("last.pt", "history.json", weights)) is None:
        return False
    return _marker_valid(job_dir / "evaluation_complete.json", identity, ("metrics.json", "predictions.csv")) is not None
