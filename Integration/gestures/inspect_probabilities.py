"""Inspect saved predictions only; this command never loads models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def inspect(experiment: Path, clip_id: str, role: str = "holdout") -> list[dict]:
    experiment = experiment.resolve()
    frozen = json.loads((experiment / "experiment.json").read_text(encoding="utf-8"))
    classes = frozen["class_order"]
    reports = experiment / "reports"
    values = []
    for model in ("gru", "lstm", "ddnet"):
        path = reports / f"{role}_predictions_{model}.csv" if role == "oof" else experiment / "jobs" / f"{model}_final" / "predictions.csv"
        rows = _read(path)
        found = [row for row in rows if row.get("clip_id") == clip_id]
        if len(found) != 1:
            raise ValueError(f"expected one saved {role} prediction for clip {clip_id} in {path}, found {len(found)}")
        row = found[0]
        expected_columns = ["prob__" + name for name in classes]
        if [key for key in row if key.startswith("prob__")] != expected_columns:
            raise ValueError(f"probability columns do not match saved class order in {path}")
        probabilities = [float(row[key]) for key in expected_columns]
        if not np.isfinite(probabilities).all() or any(p < 0 or p > 1 for p in probabilities) or not np.isclose(sum(probabilities), 1.0, atol=1e-5):
            raise ValueError(f"invalid probability vector in {path}")
        if row.get("model_id") != model or row.get("config_id") != frozen["identity"]:
            raise ValueError(f"model or experiment identity mismatch in {path}")
        values.append({"model": model.upper(), "true": row["true_class"], "predicted": row["predicted_class"], "probabilities": probabilities})
    if len({item["true"] for item in values}) != 1:
        raise ValueError("the three saved predictions disagree on the true class")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, help="experiment directory")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--role", choices=("holdout", "oof"), default="holdout")
    args = parser.parse_args(argv)
    try:
        frozen = json.loads((args.experiment.resolve() / "experiment.json").read_text(encoding="utf-8"))
        classes = frozen["class_order"]
        for result in inspect(args.experiment, args.clip_id, args.role):
            print(f"{result['model']}: true={result['true']} predicted={result['predicted']}")
            order = sorted(range(len(classes)), key=lambda index: (-result["probabilities"][index], index))
            for index in order:
                print(f"  {classes[index]}: {result['probabilities'][index] * 100:.6f}%")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
