from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .contracts import ActionRange
from .model import FurnitureRepairNetwork


def checkpoint_payload(
    model: FurnitureRepairNetwork,
    action_range: ActionRange,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract": model.checkpoint_contract(),
        "action_range": action_range.to_dict(),
        "model_state_dict": model.state_dict(),
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def save_checkpoint(
    path: str | Path,
    model: FurnitureRepairNetwork,
    action_range: ActionRange,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model, action_range, optimizer=optimizer, extra=extra
        ),
        destination,
    )


def load_checkpoint(
    path: str | Path,
    model: FurnitureRepairNetwork,
    expected_action_range: ActionRange,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    expected_contract = model.checkpoint_contract()
    if payload.get("contract") != expected_contract:
        raise ValueError("checkpoint model/graph/vocabulary contract mismatch")
    if payload.get("action_range") != expected_action_range.to_dict():
        raise ValueError("checkpoint action range mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        state = payload.get("optimizer_state_dict")
        if state is None:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(state)
    return dict(payload.get("extra") or {})
