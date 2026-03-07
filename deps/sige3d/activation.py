from __future__ import annotations

import torch


def activation(x: torch.Tensor, activation_name: str) -> torch.Tensor:
    if activation_name == "relu":
        return torch.relu(x)
    if activation_name == "sigmoid":
        return torch.sigmoid(x)
    if activation_name == "tanh":
        return torch.tanh(x)
    if activation_name in ("swish", "silu"):
        return x * torch.sigmoid(x)
    if activation_name == "identity":
        return x
    raise ValueError(f"Unknown activation: {activation_name}")
