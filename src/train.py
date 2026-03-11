"""
Multi-stage contrastive embedding training loop.
This is the main file the agent modifies during experiments.

Usage: uv run src/train.py [--config configs/training_stages.yaml]

This file is AGENT-MUTABLE: architecture, optimizer, hyperparameters, stages,
batch size, model size — everything is fair game.
"""

import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force line-buffered stdout so run.log streams in real-time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ---- Configuration ----

DEFAULT_CONFIG = "configs/training_stages.yaml"


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---- Model ----

class EmbeddingModel(torch.nn.Module):
    """Wraps a pretrained encoder with mean/cls pooling and optional projection."""

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

    def encode(self, sentences, tokenizer, batch_size=64, max_length=512):
        """MTEB-compatible encode method."""
        all_embs = []
        self.eval()
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
                enc = {k: v.to(next(self.parameters()).device) for k, v in enc.items()}
                emb = self.forward(enc["input_ids"], enc["attention_mask"])
                all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)


def infonce_loss(query_emb, positive_emb, temperature=0.05):
    """InfoNCE loss with in-batch negatives."""
    sim = torch.matmul(query_emb, positive_emb.T) / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return torch.nn.functional.cross_entropy(sim, labels)


def infonce_loss_with_hard_negs(query_emb, positive_emb, hard_neg_emb, temperature=0.05, hard_neg_weight=2.0):
    """InfoNCE loss with in-batch negatives plus explicit hard negatives."""
    B = query_emb.size(0)
    # In-batch similarity: (B, B)
    sim_inbatch = torch.matmul(query_emb, positive_emb.T) / temperature
    # Hard neg similarity: (B, 1) -> squeeze
    sim_hardneg = torch.sum(query_emb * hard_neg_emb, dim=-1, keepdim=True) / temperature * hard_neg_weight
    # Concatenate: (B, B+1)
    logits = torch.cat([sim_inbatch, sim_hardneg], dim=1)
    labels = torch.arange(B, device=query_emb.device)
    return torch.nn.functional.cross_entropy(logits, labels)


# ---- Data Loading ----

def load_training_data(datasets_to_load: list[str], max_rows_per_dataset: int = 50000) -> list[dict]:
    """Load and combine multiple training datasets."""
    from transformers import AutoTokenizer
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets library not available")
        return []

    all_triplets = []

    for ds_spec in datasets_to_load:
        ds_id = ds_spec.get("id")
        config = ds_spec.get("config")
        fmt = ds_spec.get("format", "triplet")
        split = ds_spec.get("split", "train")
        try:
            print(f"  Loading {ds_id} ({fmt})...")
            if config:
                ds = load_dataset(ds_id, config, split=f"{split}[:{max_rows_per_dataset}]")
            else:
                ds = load_dataset(ds_id, split=f"{split}[:{max_rows_per_dataset}]")

            if fmt == "nli":
                for row in ds:
                    label = row.get("label", -1)
                    premise = row.get("premise", row.get("sentence1", ""))
                    hypothesis = row.get("hypothesis", row.get("sentence2", ""))
                    if label == 0:  # entailment
                        all_triplets.append({"query": premise, "positive": hypothesis, "source": ds_id})
            elif fmt == "quora_pairs":
                # Quora: questions column has list of 2 questions, is_duplicate field
                for row in ds:
                    qs = row.get("questions", {})
                    if isinstance(qs, dict):
                        texts = qs.get("text", [])
                        is_dup = row.get("is_duplicate", 0)
                        if is_dup and len(texts) >= 2:
                            all_triplets.append({"query": texts[0], "positive": texts[1], "source": ds_id})
                    elif isinstance(qs, list) and len(qs) >= 2:
                        is_dup = row.get("is_duplicate", 0)
                        if is_dup:
                            all_triplets.append({"query": qs[0], "positive": qs[1], "source": ds_id})
            elif fmt == "ms_marco":
                # MS MARCO: query + positive passage from passages
                for row in ds:
                    query = row.get("query", "")
                    passages = row.get("passages", {})
                    if isinstance(passages, dict):
                        texts = passages.get("passage_text", [])
                        labels = passages.get("is_selected", [])
                        for i, (text, label) in enumerate(zip(texts, labels)):
                            if label == 1 and text:
                                all_triplets.append({"query": query, "positive": text, "source": ds_id})
                                break  # one positive per query
            elif fmt == "triplet":
                cols = ds.column_names
                anchor_col = next((c for c in cols if c in ("anchor", "query", "sentence1", "text1")), cols[0])
                pos_col = next((c for c in cols if c in ("positive", "pos", "sentence2", "text2")), cols[1] if len(cols) > 1 else cols[0])
                neg_col = next((c for c in cols if c in ("negative", "neg", "sentence3", "text3")), None)
                for row in ds:
                    t = {"query": str(row[anchor_col]), "positive": str(row[pos_col]), "source": ds_id}
                    if neg_col and row.get(neg_col):
                        t["negatives"] = [str(row[neg_col])]
                    all_triplets.append(t)
            elif fmt == "pair_score":
                cols = ds.column_names
                s1_col = next((c for c in cols if c in ("sentence1", "text1", "anchor")), cols[0])
                s2_col = next((c for c in cols if c in ("sentence2", "text2", "positive")), cols[1])
                score_col = next((c for c in cols if c in ("score", "label", "similarity")), None)
                for row in ds:
                    score = float(row[score_col]) if score_col else 1.0
                    if score >= 3.5:  # high similarity pairs only
                        all_triplets.append({"query": str(row[s1_col]), "positive": str(row[s2_col]), "source": ds_id})
            print(f"    -> {len(all_triplets)} total pairs so far")
        except Exception as e:
            print(f"    -> FAILED: {e}")

    return all_triplets


# ---- Training Stage ----

def run_training_stage(
    model,
    tokenizer,
    triplets: list[dict],
    stage_cfg: dict,
    device,
    optimizer,
    stage_name: str,
):
    """Run one training stage for the configured duration."""
    duration_s = float(stage_cfg.get("duration_minutes", 10)) * 60
    batch_size = int(stage_cfg.get("batch_size", 128))
    temperature = float(stage_cfg.get("temperature", 0.05))
    max_seq_len = 256  # keep sequences short for speed
    hard_neg_weight = float(stage_cfg.get("hard_neg_weight", 1.0))

    model.train()
    stage_start = time.time()
    step = 0
    total_loss = 0.0

    print(f"\n=== Stage: {stage_name} ({stage_cfg.get('duration_minutes', 10)} min) ===")

    data = list(triplets)
    random.shuffle(data)

    while time.time() - stage_start < duration_s:
        for batch_start in range(0, len(data), batch_size):
            if time.time() - stage_start >= duration_s:
                break
            batch = data[batch_start:batch_start + batch_size]
            if len(batch) < 2:
                continue

            queries = [t["query"][:500] for t in batch]
            positives = [t["positive"][:500] for t in batch]

            q_enc = tokenizer(queries, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt").to(device)
            p_enc = tokenizer(positives, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt").to(device)

            q_emb = model(q_enc["input_ids"], q_enc["attention_mask"])
            p_emb = model(p_enc["input_ids"], p_enc["attention_mask"])

            # Check for hard negatives
            has_hard_negs = any(t.get("negatives") for t in batch)
            if has_hard_negs and hard_neg_weight > 1.0:
                negs_text = []
                for t in batch:
                    negs = t.get("negatives", [])
                    negs_text.append(negs[0][:500] if negs else t["positive"][:500])
                n_enc = tokenizer(negs_text, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt").to(device)
                n_emb = model(n_enc["input_ids"], n_enc["attention_mask"])
                loss = infonce_loss_with_hard_negs(q_emb, p_emb, n_emb, temperature=temperature, hard_neg_weight=hard_neg_weight)
            else:
                loss = infonce_loss(q_emb, p_emb, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            total_loss += loss.item()

            if step % 50 == 0:
                elapsed = time.time() - stage_start
                avg_loss = total_loss / step
                print(f"  [{stage_name}] Step {step} | loss={avg_loss:.4f} | {elapsed:.0f}s/{duration_s:.0f}s")

        random.shuffle(data)

    stage_time = time.time() - stage_start
    avg_loss = total_loss / max(step, 1)
    print(f"  [{stage_name}] Done: {step} steps, avg_loss={avg_loss:.4f}, {stage_time:.0f}s")
    return step


def mine_hard_negatives(model, tokenizer, triplets: list[dict], device, top_k: int = 7, batch_size: int = 64):
    """Mine hard negatives: embed all positives, find top-k nearest non-positives for each query."""
    print("\n=== Hard Negative Mining ===")
    model.eval()
    max_len = 128  # shorter seqs for faster mining

    # Free MPS memory from training before mining
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    gc.collect()

    # Embed all positives in small batches, keep on CPU
    all_texts = [t["positive"][:400] for t in triplets]
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            emb = model(enc["input_ids"], enc["attention_mask"])
            all_embeddings.append(emb.cpu().float())
            del enc, emb
            if i % 1000 == 0 and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
    all_emb = torch.cat(all_embeddings, dim=0)  # (N, D) on CPU
    del all_embeddings

    # For each query, find top-k hard negatives
    query_texts = [t["query"][:400] for t in triplets]
    mined = list(triplets)

    with torch.no_grad():
        for i in range(0, len(query_texts), batch_size):
            batch_q = query_texts[i:i+batch_size]
            enc = tokenizer(batch_q, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            q_emb = model(enc["input_ids"], enc["attention_mask"]).cpu().float()  # (B, D) on CPU
            del enc

            # Similarity to all positives
            sims = torch.matmul(q_emb, all_emb.T)  # (B, N)

            for j in range(len(batch_q)):
                idx = i + j
                sims[j, idx] = -1.0  # exclude self
                # Also exclude nearby positives
                top_sims, top_indices = sims[j].topk(top_k + 5)
                hard_negs = []
                for sim_val, neg_idx in zip(top_sims.tolist(), top_indices.tolist()):
                    if 0.3 <= sim_val <= 0.95 and neg_idx != idx:
                        hard_negs.append(all_texts[neg_idx])
                    if len(hard_negs) >= top_k:
                        break
                if hard_negs:
                    mined[idx] = dict(mined[idx])
                    mined[idx]["negatives"] = hard_negs

    neg_count = sum(1 for t in mined if t.get("negatives"))
    print(f"  Mined hard negatives for {neg_count}/{len(mined)} triplets")
    return mined


# ---- MTEB Evaluation ----

def run_mteb_eval(model, tokenizer, device, tasks: list[str], output_dir: str = "mteb_results") -> dict:
    """Run MTEB evaluation via the official mteb library."""
    import mteb
    import warnings
    from mteb.models.abs_encoder import AbsEncoder

    class ModelWrapper(AbsEncoder):
        """MTEB AbsEncoder wrapper around our PyTorch model."""
        mteb_model_meta = None

        def __init__(self, m, tok):
            self.model = m
            self.tokenizer = tok

        def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs):
            """inputs is a DataLoader yielding BatchedInput dicts with 'text' key."""
            all_embs = []
            for batch in inputs:
                sentences = batch.get("text", batch.get("sentence", []))
                if not sentences and batch:
                    sentences = list(batch.values())[0]
                if sentences:
                    emb = self.model.encode(sentences, self.tokenizer, batch_size=64)
                    all_embs.append(emb)
            if all_embs:
                return np.concatenate(all_embs, axis=0)
            return np.zeros((0, self.model.output_dim))

    wrapper = ModelWrapper(model, tokenizer)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Use deprecated MTEB class with task objects
    task_objects = mteb.get_tasks(tasks=tasks, languages=["eng"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ev = mteb.MTEB(tasks=task_objects)
        results = ev.run(wrapper, output_folder=output_dir, overwrite_results=True)

    # Parse scores from TaskResult objects
    scores = {}
    for task_result in results:
        task_name = getattr(task_result, 'task_name', None)
        if task_name is None:
            continue
        if hasattr(task_result, 'scores'):
            for split_name in ["test", "validation", "dev"]:
                if split_name in task_result.scores:
                    split_scores = task_result.scores[split_name]
                    if isinstance(split_scores, list) and split_scores:
                        score = split_scores[0].get("main_score", 0)
                    elif isinstance(split_scores, dict):
                        score = split_scores.get("main_score", 0)
                    else:
                        score = 0
                    scores[task_name] = float(score) * 100
                    break

    # Fallback: parse JSON files from output_folder
    if not scores:
        for task_name in tasks:
            result_files = list(Path(output_dir).rglob(f"*{task_name}*.json"))
            if result_files:
                with open(sorted(result_files)[-1]) as f:
                    task_result = json.load(f)
                for split_name in ["test", "validation", "dev"]:
                    if split_name in task_result:
                        split_data = task_result[split_name]
                        if isinstance(split_data, dict):
                            score = split_data.get("main_score", 0)
                            if isinstance(score, dict):
                                score = score.get("spearman", score.get("ap", 0))
                            scores[task_name] = float(score) * 100
                        break

    return scores


# ---- Main ----

FULL_TASKS = [
    "STSBenchmark", "SICK-R",
    "TwitterURLCorpus", "SprintDuplicateQuestions",
    "TwentyNewsgroupsClustering", "RedditClustering",
    "SciFact", "NFCorpus",
]

QUICK_TASKS = ["STSBenchmark", "SICK-R", "TwitterURLCorpus"]

DATASETS = [
    # all-nli triplets: safe for training (pre-curated, no STS test overlap)
    {"id": "sentence-transformers/all-nli", "config": "triplet", "format": "triplet"},
    # Quora question pairs: duplicate questions, no STS benchmark overlap
    {"id": "quora", "config": None, "format": "quora_pairs"},
    # MS MARCO passages: retrieval pairs, diverse domain
    {"id": "ms_marco", "config": "v2.1", "format": "ms_marco"},
    # NOTE: snli and multi_nli REMOVED — overlap with SICK-R and STSBenchmark test sets
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--quick-eval-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    total_start = time.time()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load model
    model_name = config.get("base_model", "answerdotai/ModernBERT-base")
    print(f"Loading base model: {model_name}")
    from transformers import AutoTokenizer, AutoModel
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

    # Load training data
    print("Loading training data...")
    triplets = load_training_data(DATASETS, max_rows_per_dataset=30000)
    print(f"Total training pairs: {len(triplets)}")

    if not triplets:
        print("WARNING: No training data loaded. Using scaffold baseline.")
        triplets = []

    stages = config.get("stages", {})

    # ---- Stage 1: Warmup ----
    warmup_cfg = stages.get("warmup", {})
    # For warmup: NLI entailment only
    # Warmup: use all-nli (shorter, varied NLI pairs) or all data if not available
    nli_triplets = [t for t in triplets if t.get("source") == "sentence-transformers/all-nli"]
    warmup_data = nli_triplets if nli_triplets else triplets

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(warmup_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("optimizer", {}).get("weight_decay", 0.01)),
    )
    train_start = time.time()

    if warmup_data:
        run_training_stage(model, tokenizer, warmup_data, warmup_cfg, device, optimizer, "warmup")

    # ---- Stage 2: Full contrastive ----
    contrastive_cfg = stages.get("contrastive", {})
    # Lower LR for main contrastive stage
    for pg in optimizer.param_groups:
        pg["lr"] = float(contrastive_cfg.get("learning_rate", 5e-5))

    if triplets:
        run_training_stage(model, tokenizer, triplets, contrastive_cfg, device, optimizer, "contrastive")

    # Free MPS memory before mining
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    gc.collect()

    # ---- Stage 3: Hard negative mining ----
    mining_cfg = stages.get("hard_neg_mining", {})
    if triplets:
        # Cap at 20K for mining to keep similarity matrix manageable (CPU: 20K*256*4B = ~20MB)
        mining_triplets = triplets[:20000]
        triplets_with_negs = mine_hard_negatives(
            model, tokenizer, mining_triplets, device,
            top_k=int(mining_cfg.get("top_k", 7)),
            batch_size=64,  # small batch to avoid MPS OOM
        )
    else:
        triplets_with_negs = triplets

    # ---- Stage 4: Hard negative fine-tuning ----
    finetuning_cfg = stages.get("fine_tuning", {})
    for pg in optimizer.param_groups:
        pg["lr"] = float(finetuning_cfg.get("learning_rate", 1e-5))

    if triplets_with_negs:
        run_training_stage(model, tokenizer, triplets_with_negs, finetuning_cfg, device, optimizer, "fine_tuning")

    train_time = time.time() - train_start
    print(f"\nTotal training time: {train_time/60:.1f} min")

    # Save checkpoint
    ckpt_dir = Path("checkpoints") / f"exp_{time.strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "num_params": num_params,
        "train_time_s": train_time,
    }, ckpt_dir / "model.pt")
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    # Symlink as latest
    latest = Path("checkpoints/latest")
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(ckpt_dir.resolve())
    print(f"Checkpoint saved: {ckpt_dir}")

    # ---- Evaluation ----
    print("\nRunning MTEB evaluation...")
    model.eval()

    eval_tasks = QUICK_TASKS if args.quick_eval_only else FULL_TASKS
    eval_outdir = "mteb_results_quick" if args.quick_eval_only else "mteb_results"

    try:
        scores = run_mteb_eval(model, tokenizer, device, eval_tasks, output_dir=eval_outdir)
    except Exception as e:
        print(f"MTEB eval failed: {e}")
        import traceback; traceback.print_exc()
        scores = {}

    # Compute category averages
    sts_scores = [scores.get("STSBenchmark", 0), scores.get("SICK-R", 0)]
    sts_avg = np.mean([s for s in sts_scores if s > 0]) if any(s > 0 for s in sts_scores) else 0.0

    pair_scores = [scores.get("TwitterURLCorpus", 0), scores.get("SprintDuplicateQuestions", 0)]
    pair_avg = np.mean([s for s in pair_scores if s > 0]) if any(s > 0 for s in pair_scores) else 0.0

    cluster_scores = [scores.get("TwentyNewsgroupsClustering", 0), scores.get("RedditClustering", 0)]
    cluster_avg = np.mean([s for s in cluster_scores if s > 0]) if any(s > 0 for s in cluster_scores) else 0.0

    retrieval_scores = [scores.get("SciFact", 0), scores.get("NFCorpus", 0)]
    retrieval_avg = np.mean([s for s in retrieval_scores if s > 0]) if any(s > 0 for s in retrieval_scores) else 0.0

    primary = 0.3 * sts_avg + 0.2 * pair_avg + 0.2 * cluster_avg + 0.3 * retrieval_avg

    # Peak memory
    if torch.backends.mps.is_available():
        peak_mem = torch.mps.driver_allocated_memory() / 1024**3
    else:
        peak_mem = 0.0

    total_time = time.time() - total_start

    print("\n---")
    print(f"primary_score:     {primary:.4f}")
    print(f"sts_avg:           {sts_avg:.4f}")
    print(f"pair_class_avg:    {pair_avg:.4f}")
    print(f"cluster_avg:       {cluster_avg:.4f}")
    print(f"retrieval_avg:     {retrieval_avg:.4f}")
    print(f"training_minutes:  {train_time / 60:.1f}")
    print(f"peak_memory_gb:    {peak_mem:.1f}")
    print(f"num_params_M:      {num_params / 1e6:.1f}")
    print(f"base_model:        {model_name}")
    print(f"training_stage:    full_4stage")
    print(f"total_train_pairs: {len(triplets)}")
    print(f"\nPer-task scores:")
    for task, score in sorted(scores.items()):
        print(f"  {task}: {score:.2f}")


if __name__ == "__main__":
    main()
