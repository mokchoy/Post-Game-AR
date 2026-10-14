from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

MODEL_KIND = "yolo11n_two_player_heatmap_v1"
BACKBONE_LAYERS = 11
HEATMAP_STRIDE = 4
CHANNELS = {"far": 0, "near": 1}


def pixel_center_scale(point, source_width: int, source_height: int,
                       target_width: int, target_height: int) -> tuple[float, float]:
    x, y = point
    return ((float(x) + 0.5) * target_width / source_width - 0.5,
            (float(y) + 0.5) * target_height / source_height - 0.5)


def heatmap_target(points, height: int, width: int, sigma: float):
    import torch

    if len(points) != 2 or height < 1 or width < 1 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Expected two points, positive heatmap dimensions, and sigma > 0")
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    targets = []
    for x_input, y_input in points:
        x = (float(x_input) + 0.5) / HEATMAP_STRIDE - 0.5
        y = (float(y_input) + 0.5) / HEATMAP_STRIDE - 0.5
        target = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        targets.append(target / target.sum())
    return torch.stack(targets).float()


class PointHeatmapModel(nn.Module):
    """Two-channel far/near decoder over YOLO11n layers 0 through 10."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.shallow_project = nn.Conv2d(64, 32, 1)
        self.deep_project = nn.Conv2d(256, 32, 1)
        self.refine = nn.Conv2d(32, 32, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.heatmaps = nn.Conv2d(32, 2, 1)

    def forward(self, image):
        values, x, shallow = [], image, None
        for layer in self.backbone:
            source = layer.f
            if isinstance(source, int) and source != -1:
                x = values[source]
            elif isinstance(source, list):
                x = [x if index == -1 else values[index] for index in source]
            x = layer(x)
            values.append(x)
            if layer.i == 2:
                shallow = x
        if shallow is None:
            raise RuntimeError("YOLO11n backbone did not expose layer-2 features")
        deep = torch.nn.functional.interpolate(
            self.deep_project(x), size=shallow.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.heatmaps(self.relu(self.refine(self.shallow_project(shallow) + deep)))


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_hash(state_dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize_from_yolo11n(weights: Path, seed: int):
    import torch
    from ultralytics import YOLO

    seed_everything(seed)
    loaded = YOLO(str(weights), task="detect")
    source = loaded.model.model[:BACKBONE_LAYERS]
    model = PointHeatmapModel(source)
    source_state, target_state = source.state_dict(), model.backbone.state_dict()
    if source_state.keys() != target_state.keys():
        raise RuntimeError("YOLO11n backbone parameter names differ from the custom backbone")
    if any(source_state[key].shape != target_state[key].shape for key in source_state):
        raise RuntimeError("YOLO11n backbone parameter shapes differ from the custom backbone")
    model.backbone.load_state_dict(source_state, strict=True)
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(True)
    transferred = [key for key in source_state
                   if torch.equal(source_state[key].detach().cpu(), target_state[key].detach().cpu())]
    if len(transferred) != len(source_state):
        raise RuntimeError(f"Incomplete pretrained transfer: {len(transferred)}/{len(source_state)} tensors")
    report = {
        "model_kind": MODEL_KIND,
        "source_checkpoint": str(weights.resolve()),
        "source_checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "backbone_layers": list(range(BACKBONE_LAYERS)),
        "source_state_tensors": len(source_state),
        "target_state_tensors": len(target_state),
        "transferred_parameters_and_buffers": len(transferred),
        "complete_transfer": True,
        "transferred_key_sha256": hashlib.sha256("\n".join(sorted(transferred)).encode()).hexdigest(),
        "head_initialization": {
            "method": "PyTorch default Conv2d initialization",
            "seed": seed,
            "state_tensors": len(head_state_dict(model)),
            "state_sha256": tensor_hash(head_state_dict(model)),
            "tensor_sha256": {
                key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for key, value in sorted(head_state_dict(model).items())
            },
            "parameters": {key: list(value.shape) for key, value in head_state_dict(model).items()},
        },
        "contract": model_contract(),
    }
    del loaded
    return model, report


def architecture_backbone():
    from ultralytics import YOLO

    return YOLO("yolo11n.yaml", task="detect").model.model[:BACKBONE_LAYERS]


def model_contract() -> dict:
    return {
        "kind": MODEL_KIND, "backbone": "YOLO11n layers 0-10",
        "channels": {"far": 0, "near": 1}, "classes": 1, "keypoints_per_person": 1,
        "input": "full RGB frame resized, float32 [0,1], no crop or ImageNet normalization",
        "output": "two stride-4 heatmap logits", "decoder": "one spatial argmax per channel",
    }


def head_state_dict(model):
    return {
        key: value for key, value in model.state_dict().items()
        if not key.startswith("backbone.")
    }


def image_tensor(frame, input_width: int, input_height: int):
    import torch

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)


def decode_heatmaps(logits, input_width: int, input_height: int,
                    native_width: int, native_height: int) -> list[dict]:
    import torch

    if logits.ndim != 3 or logits.shape[0] != 2:
        raise ValueError(f"Expected two heatmaps, received shape {tuple(logits.shape)}")
    height, width = logits.shape[-2:]
    probabilities = torch.softmax(logits.float().flatten(1), dim=1)
    decoded = []
    for channel in range(2):
        index = int(torch.argmax(probabilities[channel]).item())
        x_heat, y_heat = index % width, index // width
        x_input = (x_heat + 0.5) * HEATMAP_STRIDE - 0.5
        y_input = (y_heat + 0.5) * HEATMAP_STRIDE - 0.5
        x_native, y_native = pixel_center_scale(
            (x_input, y_input), input_width, input_height, native_width, native_height
        )
        decoded.append({
            "x_native": x_native, "y_native": y_native,
            "heatmap_score": float(probabilities[channel, index].item()),
            "heatmap_x": x_heat, "heatmap_y": y_heat,
        })
    return decoded


class FrameDataset:
    def __new__(cls, rows, input_width: int, input_height: int, sigma: float,
                horizontal_flip_probability: float = 0.0):
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __init__(self):
                self.rows = rows
                self.input_width, self.input_height = input_width, input_height
                self.sigma = sigma
                self.horizontal_flip_probability = horizontal_flip_probability

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                row = self.rows[index]
                image = cv2.imread(str(row["image_path"]))
                if image is None:
                    raise RuntimeError(f"Could not read prepared image: {row['image_path']}")
                tensor = image_tensor(image, self.input_width, self.input_height)
                native_width, native_height = int(row["width"]), int(row["height"])
                points = [None, None]
                for player in ("A", "B"):
                    half = row[f"{player}_half"]
                    if half not in CHANNELS or points[CHANNELS[half]] is not None:
                        raise ValueError(f"Invalid far/near mapping in {stable_key(row)}")
                    points[CHANNELS[half]] = pixel_center_scale(
                        (float(row[f"{player}_native_x"]), float(row[f"{player}_native_y"])),
                        native_width, native_height, self.input_width, self.input_height,
                    )
                if self.horizontal_flip_probability and random.random() < self.horizontal_flip_probability:
                    tensor = torch.flip(tensor, dims=(2,))
                    points = [(self.input_width - 1 - point[0], point[1]) for point in points]
                targets = heatmap_target(
                    points, (self.input_height + 3) // 4, (self.input_width + 3) // 4, self.sigma
                )
                return tensor, targets

        return _Dataset()


def spatial_cross_entropy(logits, targets):
    import torch.nn.functional as F

    if logits.shape != targets.shape:
        raise ValueError(f"Model/target heatmap shapes differ: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    log_prob = F.log_softmax(logits.flatten(2), dim=2)
    return -(targets.flatten(2) * log_prob).sum(dim=2).mean()


def save_checkpoint(path: Path, model, settings: dict, metadata: dict) -> str:
    import os
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model_kind": MODEL_KIND,
        "model_contract": model_contract(),
        "state_dict": model.state_dict(),
        "resolved_settings": settings,
        "metadata": metadata,
    }, temp)
    os.replace(temp, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def load_checkpoint(path: Path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != MODEL_KIND or checkpoint.get("model_contract") != model_contract():
        raise ValueError(f"Incompatible or obsolete pose checkpoint: {path}")
    model = PointHeatmapModel(architecture_backbone())
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint


def one_keypoint_output(model, input_width: int, input_height: int, device="cpu") -> dict:
    import torch

    model = model.to(device).eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, input_height, input_width, device=device))
    expected = ((input_height + 3) // 4, (input_width + 3) // 4)
    if tuple(logits.shape) != (1, 2, *expected):
        raise RuntimeError(f"Expected one point per far/near channel, output was {tuple(logits.shape)}")
    return {"output_shape": list(logits.shape), "points_per_image": 2,
            "channel_mapping": dict(CHANNELS)}
