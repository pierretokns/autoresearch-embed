"""
Multi-stage contrastive embedding training loop.
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

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- Configuration ----

DEFAULT_CONFIG = "configs/training_stages.yaml"


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---- Model ----

class EmbeddingModel(torch.nn.Module):
    """Wraps a pretrained encoder with mean pooling and optional projection."""

    def __init__(self, encoder, hidden_size: int, projection_dim: int = None, pooling: str = "mean"):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        if projection_dim and projection_dim != hidden_size:
            self.projection = torch.nn.Linear(hidden_size, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projection = None
            self.output_dim = hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T, H)

        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:  # mean
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        if self.projection is not None:
            pooled = self.projection(pooled)

        # L2 normalize
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled


def infonce_loss(query_emb, positive_emb, temperature=0.05):
    """InfoNCE loss with in-batch negatives."""
    sim = torch.matmul(query_emb, positive_emb.T) / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return torch.nn.functional.cross_entropy(sim, labels)


# ---- Main ----

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    total_start = time.time()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load model
    model_name = config.get("base_model", "answerdotai/ModernBERT-base")
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    hidden_size = encoder.config.hidden_size

    model = EmbeddingModel(
        encoder=encoder,
        hidden_size=hidden_size,
        projection_dim=config.get("projection_dim"),
        pooling=config.get("pooling", "mean"),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.1f}M")
    print(f"Hidden size: {hidden_size}, Output dim: {model.output_dim}")

    # Load a small training dataset for baseline
    print("Loading training data...")
    try:
        from datasets import load_dataset
        ds = load_dataset("sentence-transformers/all-nli", "triplet", split="train[:10000]")
        triplets = [{"query": row["anchor"], "positive": row["positive"]} for row in ds]
        print(f"Loaded {len(triplets)} training pairs")
    except Exception as e:
        print(f"Could not load dataset: {e}")
        triplets = []

    if not triplets:
        print("No training data available. Reporting scaffold baseline.")
        total_time = time.time() - total_start
        print("\n---")
        print(f"primary_score:     0.0000")
        print(f"sts_avg:           0.0000")
        print(f"pair_class_avg:    0.0000")
        print(f"cluster_avg:       0.0000")
        print(f"retrieval_avg:     0.0000")
        print(f"training_minutes:  {total_time / 60:.1f}")
        print(f"peak_memory_gb:    0.0")
        print(f"num_params_M:      {num_params / 1e6:.1f}")
        print(f"base_model:        {model_name}")
        print(f"training_stage:    scaffold")
        print(f"total_train_pairs: 0")
        return

    # Training
    stages = config.get("stages", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stages.get("warmup", {}).get("learning_rate", 1e-4)),
        weight_decay=float(config.get("optimizer", {}).get("weight_decay", 0.01)),
    )
    batch_size = int(stages.get("warmup", {}).get("batch_size", 64))
    temperature = float(stages.get("warmup", {}).get("temperature", 0.1))
    max_seq_len = int(config.get("max_seq_length", 512))

    # Simple training loop — agent will make this multi-stage
    total_duration = sum(
        float(s.get("duration_minutes", 0))
        for s in stages.values()
        if isinstance(s, dict) and "duration_minutes" in s
    )
    budget_seconds = total_duration * 60
    print(f"Training budget: {total_duration:.0f} minutes ({budget_seconds:.0f}s)")

    model.train()
    train_start = time.time()
    step = 0
    total_loss = 0.0
    import random
    random.shuffle(triplets)

    while time.time() - train_start < budget_seconds:
        for batch_start in range(0, len(triplets), batch_size):
            if time.time() - train_start > budget_seconds:
                break

            batch = triplets[batch_start:batch_start + batch_size]
            if len(batch) < 2:
                continue

            queries = [t["query"] for t in batch]
            positives = [t["positive"] for t in batch]

            q_enc = tokenizer(queries, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt").to(device)
            p_enc = tokenizer(positives, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt").to(device)

            q_emb = model(q_enc["input_ids"], q_enc["attention_mask"])
            p_emb = model(p_enc["input_ids"], p_enc["attention_mask"])

            loss = infonce_loss(q_emb, p_emb, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            total_loss += loss.item()

            if step % 20 == 0:
                elapsed = time.time() - train_start
                avg_loss = total_loss / step
                print(f"  Step {step} | loss={avg_loss:.4f} | {elapsed:.0f}s/{budget_seconds:.0f}s")

        # Reshuffle for next epoch
        random.shuffle(triplets)

    train_time = time.time() - train_start
    avg_loss = total_loss / max(step, 1)
    print(f"Training done: {step} steps, avg_loss={avg_loss:.4f}, {train_time:.0f}s")

    # Save checkpoint
    from src.checkpoint import save_checkpoint
    ckpt_dir = Path("checkpoints") / f"exp_{time.strftime('%Y%m%d_%H%M%S')}"
    # Save just the state dict as a torch checkpoint for now
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    # Symlink as latest
    latest = Path("checkpoints/latest")
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(ckpt_dir.name)
    print(f"Checkpoint saved: {ckpt_dir}")

    # Quick eval using sentence-transformers + MTEB
    print("Running quick evaluation...")
    model.eval()

    # Wrap for MTEB
    from src.eval.mteb_runner import MTEBModelWrapper, evaluate as mteb_evaluate
    class TorchModelForMTEB:
        def __init__(self, model, tokenizer, device, max_length=512):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.max_length = max_length

        def encode(self, sentences, batch_size=64, **kwargs):
            all_embs = []
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    emb = self.model(enc["input_ids"], enc["attention_mask"])
                all_embs.append(emb.cpu().numpy())
            return np.concatenate(all_embs, axis=0)

    mteb_model = TorchModelForMTEB(model, tokenizer, device)

    # Run quick eval (3 tasks)
    import mteb
    quick_tasks = ["STSBenchmark", "SICK-R", "TwitterURLCorpus"]
    try:
        evaluation = mteb.MTEB(tasks=quick_tasks)
        results = evaluation.run(mteb_model, output_folder="mteb_results_quick")
        print("Quick eval complete.")
    except Exception as e:
        print(f"Quick eval failed: {e}")
        results = []

    # Parse results
    scores = {}
    for task_name in quick_tasks:
        result_files = list(Path("mteb_results_quick").rglob(f"*{task_name}*.json"))
        if result_files:
            import json
            with open(result_files[0]) as f:
                task_result = json.load(f)
            for split_name in ["test", "validation", "dev"]:
                if split_name in task_result:
                    split_data = task_result[split_name]
                    if isinstance(split_data, dict):
                        score = split_data.get("main_score", 0)
                        scores[task_name] = float(score) * 100
                    break

    sts_avg = np.mean([scores.get("STSBenchmark", 0), scores.get("SICK-R", 0)])
    pair_class_avg = scores.get("TwitterURLCorpus", 0)
    primary = 0.3 * sts_avg + 0.2 * pair_class_avg  # partial (missing clustering + retrieval)

    total_time = time.time() - total_start
    peak_mem = torch.mps.driver_allocated_memory() / 1024**3 if torch.backends.mps.is_available() else 0

    print("\n---")
    print(f"primary_score:     {primary:.4f}")
    print(f"sts_avg:           {sts_avg:.4f}")
    print(f"pair_class_avg:    {pair_class_avg:.4f}")
    print(f"cluster_avg:       0.0000")
    print(f"retrieval_avg:     0.0000")
    print(f"training_minutes:  {train_time / 60:.1f}")
    print(f"peak_memory_gb:    {peak_mem:.1f}")
    print(f"num_params_M:      {num_params / 1e6:.1f}")
    print(f"base_model:        {model_name}")
    print(f"training_stage:    warmup_only")
    print(f"total_train_pairs: {len(triplets)}")


if __name__ == "__main__":
    main()
