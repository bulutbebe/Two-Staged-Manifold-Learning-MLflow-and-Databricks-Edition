"""Explicit, reproducible saving and loading of model artifacts."""
from pathlib import Path
import torch


def save_checkpoint(path: str | Path, model, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, path)


def load_checkpoint(path: str | Path, model, map_location="cpu", strict: bool = True):
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    return checkpoint.get("metadata", {})
