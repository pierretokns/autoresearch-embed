"""
Multi-stage contrastive embedding training loop on MLX.
This is the main file the agent modifies during experiments.

Usage: uv run src/train.py [--config configs/training_stages.yaml]

This file is AGENT-MUTABLE: architecture, optimizer, hyperparameters, stages,
batch size, model size — everything is fair game.
"""

import gc
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import yaml
from mlx.utils import tree_flatten, tree_map
from transformers import AutoTokenizer, AutoModel

# ---- Configuration ----

DEFAULT_CONFIG = "configs/training_stages.yaml"


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---- Model Setup ----

def load_base_model(config: dict):
    """Load pretrained encoder and tokenizer."""
    model_name = config.get("base_model", "answerdotai/ModernBERT-base")
    print(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # For MLX, we'll use the transformers model and convert outputs
    # The agent can replace this with a native MLX implementation
    encoder = AutoModel.from_pretrained(model_name)
    hidden_size = encoder.config.hidden_size

    return encoder, tokenizer, hidden_size


# ---- Training Stages ----

def train_stage(
    model,
    optimizer,
    dataloader,
    duration_minutes: float,
    stage_name: str,
    loss_fn,
    loss_kwargs: dict,
):
    """Run a single training stage for a fixed wall-clock duration."""
    print(f"\n=== Stage: {stage_name} ({duration_minutes} min) ===")
    start_time = time.time()
    budget_seconds = duration_minutes * 60
    step = 0
    total_loss = 0.0

    for batch in dataloader:
        if time.time() - start_time > budget_seconds:
            break

        # Forward pass
        query_emb = model(batch["query_ids"], batch["query_mask"])
        positive_emb = model(batch["positive_ids"], batch["positive_mask"])

        negative_emb = None
        if batch.get("negative_ids") is not None:
            negative_emb = model(batch["negative_ids"], batch["negative_mask"])

        # Compute loss
        loss = loss_fn(query_emb, positive_emb, negative_emb, **loss_kwargs)

        # Backward pass
        loss.backward()
        optimizer.step()
        mx.eval(model.parameters())

        step += 1
        total_loss += loss.item()

        if step % 10 == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / step
            print(f"  Step {step} | loss={avg_loss:.4f} | {elapsed:.0f}s/{budget_seconds:.0f}s")

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(step, 1)
    print(f"  Completed {step} steps in {elapsed:.1f}s | avg_loss={avg_loss:.4f}")
    return avg_loss


# ---- Main ----

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    total_start = time.time()

    # Load model
    encoder, tokenizer, hidden_size = load_base_model(config)

    # TODO: The agent will fill in the full training pipeline here.
    # This scaffold shows the structure; the agent modifies it each experiment.

    from src.model import EmbeddingModel
    from src.losses import infonce_loss
    from src.data.loader import ContrastiveDataLoader
    from src.checkpoint import save_checkpoint

    model = EmbeddingModel(
        encoder=encoder,
        hidden_size=hidden_size,
        projection_dim=config.get("projection_dim"),
        pooling=config.get("pooling", "mean"),
        normalize=config.get("normalize_embeddings", True),
    )

    # Report model size
    num_params = sum(p.size for p in tree_flatten(model.parameters()) if hasattr(p, 'size'))
    if isinstance(num_params, int):
        pass
    else:
        num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model parameters: {num_params / 1e6:.1f}M")

    # TODO: Load training data
    # TODO: Run stages
    # TODO: Evaluate

    total_time = time.time() - total_start
    peak_memory = 0
    try:
        peak_memory = mx.metal.get_peak_memory() / 1024 / 1024 / 1024  # GB
    except Exception:
        pass

    # Print structured output
    print("\n---")
    print(f"primary_score:     0.0000")
    print(f"sts_avg:           0.0000")
    print(f"pair_class_avg:    0.0000")
    print(f"cluster_avg:       0.0000")
    print(f"retrieval_avg:     0.0000")
    print(f"training_minutes:  {total_time / 60:.1f}")
    print(f"peak_memory_gb:    {peak_memory:.1f}")
    print(f"num_params_M:      {num_params / 1e6:.1f}")
    print(f"base_model:        {config.get('base_model', 'unknown')}")
    print(f"training_stage:    scaffold")
    print(f"total_train_pairs: 0")


if __name__ == "__main__":
    main()
