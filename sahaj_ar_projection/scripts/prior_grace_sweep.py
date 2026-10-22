"""Step 15 evaluation: Step 13 protocol (joint mode, guided on, same windows) at several prior_grace_s values."""
import json
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar_projection.config import load_config
from ar_projection.experiments import window_metrics, window_sources
from ar_projection.pipeline import run

cfg = load_config("config-exp-handoff.json")
windows = [tuple(w) for w in json.load(open("runs/step13-handoff.json"))["windows"]]
out = {}
for grace in (0.25, 0.5, 1.0):  # grace 0 = runs/step13-handoff.json joint/guided
    variant = replace(cfg, pose=replace(cfg.pose, mode="joint", prior_grace_s=grace),
                      features=replace(cfg.features, guided_matching=replace(cfg.features.guided_matching, enabled=True)))
    path = run(variant, withhold_tag_windows=windows)
    out[str(grace)] = {"run": str(path), "sources": window_sources(path, windows), "windows": window_metrics(path, windows)}
json.dump(out, open("runs/step15-grace.json", "w"), indent=1)
