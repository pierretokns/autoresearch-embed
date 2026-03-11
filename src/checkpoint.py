"""
Checkpoint save/load for MLX embedding models.
Saves weights as safetensors, config as YAML, metadata as JSON.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import yaml
from mlx.utils import tree_flatten, tree_unflatten


def save_checkpoint(
    model,
    config: dict,
    checkpoint_dir: str | Path,
    metadata: dict | None = None,
):
    """Save model weights, config, and metadata."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save weights as npz
    weights = dict(tree_flatten(model.parameters()))
    mx.savez(str(checkpoint_dir / "weights.npz"), **weights)

    # Save config
    with open(checkpoint_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save metadata
    meta = metadata or {}
    meta["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(checkpoint_dir: str | Path):
    """Load model config, weights dict, and metadata from a checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)

    with open(checkpoint_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    weights = dict(mx.load(str(checkpoint_dir / "weights.npz")))

    metadata = {}
    meta_path = checkpoint_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return config, weights, metadata


def get_latest_checkpoint(checkpoints_root: str | Path) -> Path | None:
    """Find the most recent checkpoint directory."""
    root = Path(checkpoints_root)
    if not root.exists():
        return None
    dirs = [d for d in root.iterdir() if d.is_dir() and (d / "weights.npz").exists()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)
