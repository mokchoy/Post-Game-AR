"""GRU, LSTM, and the adapted DD-Net lite classifier."""

from __future__ import annotations

from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F


def _validate_input(x: torch.Tensor) -> None:
    if x.ndim != 3 or x.shape[-1] != 43:
        raise ValueError(f"expected (batch, T, 43), got {tuple(x.shape)}")
    if x.shape[1] < 8 or x.shape[1] % 2:
        raise ValueError("T must be even and at least 8")


class RecurrentClassifier(nn.Module):
    def __init__(self, classes: int, kind: str):
        super().__init__()
        if kind not in {"gru", "lstm"} or classes < 2:
            raise ValueError("kind must be gru/lstm and classes must be at least two")
        recurrent = nn.GRU if kind == "gru" else nn.LSTM
        self.recurrent = recurrent(43, 64, num_layers=1, bias=True, batch_first=True, dropout=0.0)
        self.head = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_input(x)
        _, state = self.recurrent(x)
        hidden = state[0] if isinstance(state, tuple) else state
        return self.head(hidden[-1])


def ddnet_features(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return static distances, full-rate motion, and half-rate motion."""
    _validate_input(x)
    batch, t, _ = x.shape
    poses = x[..., :42].reshape(batch, t, 21, 2)
    pose_valid = x[..., 42] > 0.5
    pairs = list(combinations(range(21), 2))
    distances = torch.stack([
        torch.linalg.vector_norm(poses[:, :, left] - poses[:, :, right], dim=-1)
        for left, right in pairs
    ], dim=-1)
    distances = distances * pose_valid.unsqueeze(-1)
    static = torch.cat((distances, pose_valid.to(x.dtype).unsqueeze(-1)), dim=-1)

    def motion_stream(source: torch.Tensor, valid: torch.Tensor, output_length: int) -> torch.Tensor:
        pair_valid = valid[:, 1:] & valid[:, :-1]
        delta = source[:, 1:] - source[:, :-1]
        delta = delta * pair_valid[:, :, None, None]
        flat = delta.reshape(batch, delta.shape[1], 42).transpose(1, 2)
        resized = F.interpolate(flat, size=output_length, mode="linear", align_corners=False)
        mask = F.interpolate(pair_valid.to(x.dtype).unsqueeze(1), size=output_length, mode="nearest")[:, 0] > 0.5
        resized = resized * mask.unsqueeze(1)
        return torch.cat((resized.transpose(1, 2), mask.to(x.dtype).unsqueeze(-1)), dim=-1)

    slow = motion_stream(poses, pose_valid, t)
    fast = motion_stream(poses[:, ::2], pose_valid[:, ::2], t // 2)
    return static, slow, fast


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            self._conv(input_channels, output_channels, 3),
            self._conv(output_channels, output_channels, 3),
        )

    @staticmethod
    def _conv(in_channels: int, out_channels: int, kernel: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding="same", bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DistanceMotionBranch(nn.Module):
    def __init__(self, features: int, base: int, pool: bool):
        super().__init__()
        layers: list[nn.Module] = [
            ConvBlock._conv(features, base * 2, 1), nn.Dropout1d(0.1),
            ConvBlock._conv(base * 2, base, 3), nn.Dropout1d(0.1),
            ConvBlock._conv(base, base, 1),
        ]
        if pool:
            layers.append(nn.MaxPool1d(2))
        layers.append(nn.Dropout1d(0.1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.transpose(1, 2))


class DDNetLite(nn.Module):
    """PyTorch port of DD-Net SHREC coarse 1D lite, adapted to Jester 2D hands."""

    def __init__(self, classes: int, base: int = 16):
        super().__init__()
        self.distance_branch = DistanceMotionBranch(211, base, pool=True)
        self.slow_branch = DistanceMotionBranch(43, base, pool=True)
        self.fast_branch = DistanceMotionBranch(43, base, pool=False)
        self.block_1 = ConvBlock(base * 3, base * 2)
        self.pool_1 = nn.MaxPool1d(2)
        self.drop_1 = nn.Dropout1d(0.1)
        self.block_2 = ConvBlock(base * 2, base * 4)
        self.pool_2 = nn.MaxPool1d(2)
        self.drop_2 = nn.Dropout1d(0.1)
        self.block_3 = ConvBlock(base * 4, base * 8)
        self.drop_3 = nn.Dropout1d(0.1)
        self.dense_1 = nn.Linear(base * 8, 128, bias=False)
        self.bn_1 = nn.BatchNorm1d(128)
        self.drop_dense_1 = nn.Dropout(0.5)
        self.dense_2 = nn.Linear(128, 128, bias=False)
        self.bn_2 = nn.BatchNorm1d(128)
        self.drop_dense_2 = nn.Dropout(0.3)
        self.output = nn.Linear(128, classes)

    @staticmethod
    def _channels(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)

    def forward_prepared(self, static: torch.Tensor, slow: torch.Tensor, fast: torch.Tensor) -> torch.Tensor:
        distance = self.distance_branch(static)
        slow_motion = self.slow_branch(slow)
        fast_motion = self.fast_branch(fast)
        x = torch.cat((distance, slow_motion, fast_motion), dim=1)
        x = self.drop_1(self.pool_1(self.block_1(x)))
        x = self.drop_2(self.pool_2(self.block_2(x)))
        x = self.drop_3(self.block_3(x))
        x = torch.amax(x, dim=2)
        x = self.drop_dense_1(F.leaky_relu(self.bn_1(self.dense_1(x)), 0.2))
        x = self.drop_dense_2(F.leaky_relu(self.bn_2(self.dense_2(x)), 0.2))
        return self.output(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_prepared(*ddnet_features(x))


def build_model(name: str, classes: int) -> nn.Module:
    if name == "gru":
        return RecurrentClassifier(classes, "gru")
    if name == "lstm":
        return RecurrentClassifier(classes, "lstm")
    if name == "ddnet":
        return DDNetLite(classes, base=16)
    raise ValueError(f"unknown classifier: {name}")
