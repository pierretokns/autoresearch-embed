"""
Multi-stage contrastive embedding training loop (MLX native).
This is the main file the agent modifies during experiments.

Usage: uv run src/train.py [--config configs/training_stages.yaml]

This file is AGENT-MUTABLE: architecture, optimizer, hyperparameters, stages,
batch size, model size — everything is fair game.
"""

import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np
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


# ---- MLX Loss Functions ----

def infonce_loss(query_emb: mx.array, positive_emb: mx.array, temperature: float = 0.05,
                 symmetric: bool = False, false_neg_threshold: float = 0.0) -> mx.array:
    """InfoNCE loss with in-batch negatives.
    symmetric=True adds positive→query direction.
    false_neg_threshold: if > 0, mask out in-batch negatives with cosine sim above this threshold
    to avoid penalizing semantically similar pairs as negatives."""
    sim_raw = mx.matmul(query_emb, positive_emb.T)  # cosine sim (inputs are L2-normalized)
    sim = sim_raw / temperature
    B = sim.shape[0]
    labels = mx.arange(B)

    # False negative filtering: mask high-similarity off-diagonal pairs
    if false_neg_threshold > 0:
        # Create mask: -inf for false negatives (high sim, not the positive pair)
        identity = mx.eye(B)
        is_false_neg = (sim_raw > false_neg_threshold) * (1 - identity)  # high sim AND not the diagonal
        fn_mask = mx.where(is_false_neg, mx.array(float("-inf")), mx.array(0.0))
        sim = sim + fn_mask

    lse = mx.logsumexp(sim, axis=1, keepdims=True)
    loss_fwd = -mx.mean((sim - lse)[mx.arange(B), labels])
    if symmetric:
        if false_neg_threshold > 0:
            sim_bwd = sim_raw.T / temperature + fn_mask.T
        else:
            sim_bwd = sim.T
        lse_bwd = mx.logsumexp(sim_bwd, axis=1, keepdims=True)
        loss_bwd = -mx.mean((sim_bwd - lse_bwd)[mx.arange(B), labels])
        return (loss_fwd + loss_bwd) * 0.5
    return loss_fwd


def matryoshka_infonce_loss(
    query_emb: mx.array, positive_emb: mx.array, temperature: float = 0.05,
    dims: list = None,
) -> mx.array:
    """Matryoshka InfoNCE: compute loss at multiple truncated dimensions and average.
    Forces the model to encode the most important info in the first few dims.
    """
    if dims is None:
        dims = [768, 512, 256, 128, 64]
    total_loss = mx.array(0.0)
    for d in dims:
        q_trunc = query_emb[:, :d]
        p_trunc = positive_emb[:, :d]
        # Re-normalize truncated embeddings
        q_trunc = q_trunc / mx.sqrt(mx.sum(q_trunc * q_trunc, axis=-1, keepdims=True) + 1e-8)
        p_trunc = p_trunc / mx.sqrt(mx.sum(p_trunc * p_trunc, axis=-1, keepdims=True) + 1e-8)
        sim = mx.matmul(q_trunc, p_trunc.T) / temperature
        labels = mx.arange(sim.shape[0])
        lse = mx.logsumexp(sim, axis=1, keepdims=True)
        total_loss = total_loss + (-mx.mean((sim - lse)[mx.arange(sim.shape[0]), labels]))
    return total_loss / len(dims)


def infonce_loss_with_hard_negs(
    query_emb: mx.array, positive_emb: mx.array, hard_neg_emb: mx.array,
    temperature: float = 0.05, hard_neg_weight: float = 1.0,
) -> mx.array:
    """InfoNCE with in-batch negatives plus explicit hard negatives.
    hard_neg_weight scales the hard neg LOSS contribution (not logits).
    Weighted combination: (1-α)·InfoNCE_inbatch + α·InfoNCE_with_hardnegs where α=hard_neg_weight/(1+hard_neg_weight)."""
    B = query_emb.shape[0]
    # Standard InfoNCE with in-batch negatives only
    sim_inbatch = mx.matmul(query_emb, positive_emb.T) / temperature
    labels = mx.arange(B)
    lse_inbatch = mx.logsumexp(sim_inbatch, axis=1, keepdims=True)
    loss_inbatch = -mx.mean((sim_inbatch - lse_inbatch)[mx.arange(B), labels])

    # InfoNCE with in-batch + hard negatives concatenated
    sim_hardneg = mx.sum(query_emb * hard_neg_emb, axis=-1, keepdims=True) / temperature
    logits_all = mx.concatenate([sim_inbatch, sim_hardneg], axis=1)
    lse_all = mx.logsumexp(logits_all, axis=1, keepdims=True)
    loss_with_hardnegs = -mx.mean((logits_all - lse_all)[mx.arange(B), labels])

    # Weighted combination: weight=0 → pure in-batch, weight=1 → equal mix
    alpha = hard_neg_weight / (1.0 + hard_neg_weight)
    return (1 - alpha) * loss_inbatch + alpha * loss_with_hardnegs


# ---- Data Loading ----
# Enable fast multi-connection HF downloads (Rust-based, ~5-10x faster)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

def _cache_path(ds_id: str, config: str | None, fmt: str, max_rows: int) -> Path:
    """Return path to cached triplets JSON for a dataset spec."""
    import hashlib
    key = f"{ds_id}|{config}|{fmt}|{max_rows}"
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    safe_name = ds_id.replace("/", "_")
    return Path("data_cache") / f"{safe_name}_{h}.json"


def load_training_data(datasets_to_load: list[str], max_rows_per_dataset: int = 50000) -> list[dict]:
    """Load and combine multiple training datasets. Uses local JSON cache after first download."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets library not available")
        return []

    Path("data_cache").mkdir(exist_ok=True)
    all_triplets = []

    for ds_spec in datasets_to_load:
        ds_id = ds_spec.get("id")
        config = ds_spec.get("config")
        fmt = ds_spec.get("format", "triplet")
        split = ds_spec.get("split", "train")

        # Check local cache first
        cache_file = _cache_path(ds_id, config, fmt, max_rows_per_dataset)
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            all_triplets.extend(cached)
            print(f"  Loading {ds_id} ({fmt})... [cached] -> {len(all_triplets)} total pairs so far")
            continue

        try:
            count_before = len(all_triplets)
            print(f"  Loading {ds_id} ({fmt})...")
            # Use streaming for large datasets to avoid downloading/processing GBs
            # we don't need. Streaming grabs rows on-the-fly.
            use_streaming = ds_spec.get("streaming", False)
            if use_streaming:
                if config:
                    ds_iter = load_dataset(ds_id, config, split=split, streaming=True)
                else:
                    ds_iter = load_dataset(ds_id, split=split, streaming=True)
                ds = list(itertools.islice(ds_iter, max_rows_per_dataset))
            elif config:
                ds = load_dataset(ds_id, config, split=f"{split}[:{max_rows_per_dataset}]")
            else:
                ds = load_dataset(ds_id, split=f"{split}[:{max_rows_per_dataset}]")

            if fmt == "nli":
                for row in ds:
                    label = row.get("label", -1)
                    premise = row.get("premise", row.get("sentence1", ""))
                    hypothesis = row.get("hypothesis", row.get("sentence2", ""))
                    if label == 0:
                        all_triplets.append({"query": premise, "positive": hypothesis, "source": ds_id})
            elif fmt == "glue_mrpc":
                for row in ds:
                    if row.get("label") == 1:
                        all_triplets.append({"query": row["sentence1"], "positive": row["sentence2"], "source": ds_id})
            elif fmt == "glue_qqp":
                for row in ds:
                    if row.get("label") == 1:
                        all_triplets.append({"query": row["question1"], "positive": row["question2"], "source": ds_id})
            elif fmt == "mnli_nonpicture":
                SAFE_GENRES = {"government", "fiction", "telephone", "travel", "slate", "nineeleven", "letters", "oup"}
                for row in ds:
                    label = row.get("label", -1)
                    genre = row.get("genre", "")
                    if label == 0 and genre in SAFE_GENRES:
                        all_triplets.append({"query": row.get("premise", ""), "positive": row.get("hypothesis", ""), "source": ds_id})
            elif fmt == "se_pairs":
                for row in ds:
                    t1 = row.get("title1", "")
                    t2 = row.get("title2", "")
                    if t1 and t2:
                        all_triplets.append({"query": t1, "positive": t2, "source": ds_id})
            elif fmt == "nq_pairs":
                for row in ds:
                    q = row.get("query", "")
                    a = row.get("answer", "")
                    if q and a and len(a) > 20:
                        all_triplets.append({"query": q, "positive": a[:300], "source": ds_id})
            elif fmt == "ms_marco":
                for row in ds:
                    query = row.get("query", "")
                    passages = row.get("passages", {})
                    if isinstance(passages, dict):
                        texts = passages.get("passage_text", [])
                        labels = passages.get("is_selected", [])
                        for text, label in zip(texts, labels):
                            if label == 1 and text:
                                all_triplets.append({"query": query, "positive": text, "source": ds_id})
                                break
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
            elif fmt == "reddit_title_body":
                # Reddit title (query) → post body (positive): diverse topic pairs
                for row in ds:
                    title = row.get("title", "")
                    body = row.get("body", row.get("selftext", ""))
                    if title and body and len(body) > 50:
                        all_triplets.append({"query": title, "positive": body[:400], "source": ds_id})
            elif fmt == "label_pairs":
                # Group texts by label, then make same-label pairs for topic clustering signal
                import random as _random
                label_to_texts: dict = {}
                for row in ds:
                    lbl = str(row.get("label", row.get("category", row.get("class_idx", -1))))
                    text = row.get("text", row.get("sentence", row.get("content", "")))
                    if text and len(text) > 20:
                        label_to_texts.setdefault(lbl, []).append(str(text)[:400])
                for lbl, texts in label_to_texts.items():
                    if len(texts) < 2:
                        continue
                    _random.shuffle(texts)
                    # Pair consecutive texts with same label
                    for i in range(0, min(len(texts) - 1, max_rows_per_dataset // len(label_to_texts)), 2):
                        all_triplets.append({"query": texts[i], "positive": texts[i + 1], "source": ds_id})
            elif fmt == "paws_pairs":
                # PAWS: sentence1/sentence2 pairs with label=1 meaning paraphrase
                for row in ds:
                    if row.get("label") == 1:
                        s1 = row.get("sentence1", "")
                        s2 = row.get("sentence2", "")
                        if s1 and s2:
                            all_triplets.append({"query": s1, "positive": s2, "source": ds_id})
            elif fmt == "hotpotqa_retrieval":
                # HotpotQA: question → supporting passage for factual/scientific retrieval training
                for row in ds:
                    question = row.get("question", "")
                    context = row.get("context", {})
                    supporting_facts = row.get("supporting_facts", {})
                    if not question or not context:
                        continue
                    titles = context.get("title", [])
                    sentences_list = context.get("sentences", [])
                    sup_titles = set(supporting_facts.get("title", []))
                    sup_sents = supporting_facts.get("sent_id", [])
                    for idx, title in enumerate(titles):
                        if title in sup_titles and idx < len(sentences_list):
                            sents = sentences_list[idx]
                            passage = " ".join(sents[:3])
                            if passage and len(passage) > 30:
                                all_triplets.append({"query": question, "positive": passage[:400], "source": ds_id})
                                break
            elif fmt == "yahoo_answers_label_pairs":
                # Yahoo Answers Topics: group by topic, pair same-topic questions
                import random as _random
                label_to_texts: dict = {}
                for row in ds:
                    topic = str(row.get("topic", row.get("label", -1)))
                    title = row.get("question_title", "")
                    content = row.get("question_content", "")
                    text = (title + " " + content).strip()
                    if text and len(text) > 20:
                        label_to_texts.setdefault(topic, []).append(str(text)[:400])
                for lbl, texts in label_to_texts.items():
                    if len(texts) < 2:
                        continue
                    _random.shuffle(texts)
                    for i in range(0, min(len(texts) - 1, max_rows_per_dataset // max(len(label_to_texts), 1)), 2):
                        all_triplets.append({"query": texts[i], "positive": texts[i + 1], "source": ds_id})
            elif fmt == "triviaqa_retrieval":
                # TriviaQA: question → answer entity pair for factual/retrieval training
                for row in ds:
                    question = row.get("question", "")
                    answer = row.get("answer", {})
                    answer_value = answer.get("value", "") if isinstance(answer, dict) else ""
                    # Also try search_results for longer context
                    search_results = row.get("search_results", {})
                    if question and answer_value and len(answer_value) > 2:
                        # Pair question with its canonical answer
                        all_triplets.append({"query": question, "positive": str(answer_value)[:400], "source": ds_id})
                        # Also pair with first relevant search result excerpt if available
                        if isinstance(search_results, dict):
                            snippets = search_results.get("search_context", [])
                            if snippets:
                                snippet = str(snippets[0])[:400]
                                if len(snippet) > 50:
                                    all_triplets.append({"query": question, "positive": snippet, "source": ds_id})
            elif fmt == "pair_score":
                cols = ds.column_names
                s1_col = next((c for c in cols if c in ("sentence1", "text1", "anchor")), cols[0])
                s2_col = next((c for c in cols if c in ("sentence2", "text2", "positive")), cols[1])
                score_col = next((c for c in cols if c in ("score", "label", "similarity")), None)
                for row in ds:
                    score = float(row[score_col]) if score_col else 1.0
                    if score >= 3.5:
                        all_triplets.append({"query": str(row[s1_col]), "positive": str(row[s2_col]), "source": ds_id})
            # Cache the new triplets for this dataset
            new_triplets = all_triplets[count_before:]
            if new_triplets:
                cache_file.write_text(json.dumps(new_triplets))
                print(f"    -> {len(all_triplets)} total pairs so far [cached to {cache_file.name}]")
            else:
                print(f"    -> {len(all_triplets)} total pairs so far")
        except Exception as e:
            print(f"    -> FAILED: {e}")

    return all_triplets


# ---- Training Stage (MLX) ----

def run_training_stage(
    model,
    tokenizer,
    triplets: list[dict],
    stage_cfg: dict,
    optimizer,
    stage_name: str,
    max_seq_length: int = 256,
    grad_accum_steps: int = 1,
    ema_state: dict | None = None,
):
    """Run one training stage for the configured duration using MLX value_and_grad."""
    duration_s = float(stage_cfg.get("duration_minutes", 10)) * 60
    batch_size = int(stage_cfg.get("batch_size", 128))
    temperature = float(stage_cfg.get("temperature", 0.05))
    max_seq_len = max_seq_length
    hard_neg_weight = float(stage_cfg.get("hard_neg_weight", 1.0))
    symmetric = bool(stage_cfg.get("symmetric", False))
    use_matryoshka = bool(stage_cfg.get("matryoshka", False))
    use_instructions = bool(stage_cfg.get("instruction_prefix", False))
    false_neg_threshold = float(stage_cfg.get("false_neg_threshold", 0.0))

    # Task-specific instruction prefixes mapped by data source
    QUERY_PREFIXES = {
        "glue": "Find semantically equivalent questions: ",
        "sentence-transformers/stackexchange-duplicates": "Find duplicate technical questions: ",
        "ms_marco": "Retrieve a passage that answers this question: ",
        "sentence-transformers/natural-questions": "Retrieve a Wikipedia passage answering: ",
        "sentence-transformers/reddit-title-body": "Find the body text for this Reddit title: ",
        "ag_news": "Find news articles on the same topic: ",
        "fancyzhx/dbpedia_14": "Find documents about the same entity: ",
        "yahoo_answers_topics": "Find questions on the same topic: ",
        "stanfordnlp/snli": "Find an entailing or paraphrasing sentence: ",
        "hotpot_qa": "Retrieve a supporting passage for this question: ",
    }
    lr_schedule = stage_cfg.get("lr_schedule", "constant")
    base_lr = float(stage_cfg.get("learning_rate", 5e-5))
    warmup_ratio = float(stage_cfg.get("warmup_ratio", 0.0))
    warmup_s = duration_s * warmup_ratio  # warmup in seconds

    def get_lr_by_time(elapsed_s: float) -> float:
        """Compute LR based on elapsed seconds (avoids step-count estimation errors)."""
        if lr_schedule == "cosine_decay":
            if elapsed_s < warmup_s and warmup_s > 0:
                return base_lr * elapsed_s / warmup_s
            progress = (elapsed_s - warmup_s) / max(1.0, duration_s - warmup_s)
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        elif lr_schedule == "linear_warmup_constant":
            if elapsed_s < warmup_s and warmup_s > 0:
                return base_lr * elapsed_s / warmup_s
            return base_lr
        return base_lr  # constant

    stage_start = time.time()
    step = 0
    total_loss = 0.0
    accum_grads = None
    accum_count = 0

    print(f"\n=== Stage: {stage_name} ({stage_cfg.get('duration_minutes', 10)} min) ===")

    data = list(triplets)
    random.shuffle(data)

    def loss_fn(model, q_ids, q_mask, p_ids, p_mask, n_ids=None, n_mask=None):
        """Compute loss given tokenized inputs."""
        q_emb = model(q_ids, q_mask)
        p_emb = model(p_ids, p_mask)
        if n_ids is not None:
            n_emb = model(n_ids, n_mask)
            return infonce_loss_with_hard_negs(q_emb, p_emb, n_emb,
                                               temperature=temperature,
                                               hard_neg_weight=hard_neg_weight)
        if use_matryoshka:
            return matryoshka_infonce_loss(q_emb, p_emb, temperature=temperature)
        return infonce_loss(q_emb, p_emb, temperature=temperature, symmetric=symmetric,
                           false_neg_threshold=false_neg_threshold)

    loss_grad_fn = nn.value_and_grad(model, loss_fn)

    while time.time() - stage_start < duration_s:
        for batch_start in range(0, len(data), batch_size):
            if time.time() - stage_start >= duration_s:
                break
            batch = data[batch_start:batch_start + batch_size]
            if len(batch) < 2:
                continue

            if use_instructions:
                queries = [
                    QUERY_PREFIXES.get(t.get("source", ""), "") + t["query"][:480]
                    for t in batch
                ]
            else:
                queries = [t["query"][:500] for t in batch]
            positives = [t["positive"][:500] for t in batch]

            q_enc = tokenizer(queries, padding=True, truncation=True,
                              max_length=max_seq_len, return_tensors="np")
            p_enc = tokenizer(positives, padding=True, truncation=True,
                              max_length=max_seq_len, return_tensors="np")

            q_ids = mx.array(q_enc["input_ids"])
            q_mask = mx.array(q_enc["attention_mask"])
            p_ids = mx.array(p_enc["input_ids"])
            p_mask = mx.array(p_enc["attention_mask"])

            # Check for hard negatives
            has_hard_negs = any(t.get("negatives") for t in batch)
            if has_hard_negs and hard_neg_weight > 0.0:
                negs_text = []
                for t in batch:
                    negs = t.get("negatives", [])
                    negs_text.append(negs[0][:500] if negs else t["positive"][:500])
                n_enc = tokenizer(negs_text, padding=True, truncation=True,
                                  max_length=max_seq_len, return_tensors="np")
                n_ids = mx.array(n_enc["input_ids"])
                n_mask = mx.array(n_enc["attention_mask"])
                loss, grads = loss_grad_fn(model, q_ids, q_mask, p_ids, p_mask, n_ids, n_mask)
            else:
                loss, grads = loss_grad_fn(model, q_ids, q_mask, p_ids, p_mask)

            # Track loss for all micro-batches
            micro_loss = float(loss.item())
            total_loss += micro_loss

            # Gradient accumulation
            if grad_accum_steps > 1:
                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = tree_map(lambda a, g: a + g, accum_grads, grads)
                accum_count += 1

                if accum_count < grad_accum_steps:
                    mx.eval(accum_grads)
                    continue

                # Average accumulated gradients and apply
                grads = tree_map(lambda g: g / grad_accum_steps, accum_grads)
                accum_grads = None
                accum_count = 0

            # Grad clipping
            grads = tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)

            # Update LR according to schedule (time-based for accuracy)
            if lr_schedule != "constant":
                current_lr = get_lr_by_time(time.time() - stage_start)
                if not hasattr(optimizer, '_llrd_ratios'):
                    optimizer.learning_rate = current_lr

            # Apply LLRD: scale gradients by per-param ratio (optimizer keeps correct base LR)
            if hasattr(optimizer, '_llrd_ratios'):
                ratios = optimizer._llrd_ratios
                flat_grads = tree_flatten(grads)
                scaled_flat = [(k, g * ratios.get(k, 1.0)) for k, g in flat_grads]
                from mlx.utils import tree_unflatten
                grads = tree_unflatten(scaled_flat)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            # EMA update (per optimizer step, not per micro-batch)
            if ema_state is not None:
                decay = ema_state["decay"]
                ema_w = ema_state["weights"]
                for k, v in tree_flatten(model.parameters()):
                    if k in ema_w:
                        ema_w[k] = decay * ema_w[k] + (1 - decay) * v
                if step % 100 == 0:
                    mx.eval(list(ema_w.values()))

            step += 1
            micro_steps = step * max(grad_accum_steps, 1)  # total micro-batches processed

            if step % 50 == 0:
                elapsed = time.time() - stage_start
                avg_loss = total_loss / micro_steps
                print(f"  [{stage_name}] Step {step} | loss={avg_loss:.4f} | {elapsed:.0f}s/{duration_s:.0f}s", flush=True)
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"loss": loss_val, "avg_loss": avg_loss, "step": step, "stage": stage_name})
                except Exception:
                    pass

        random.shuffle(data)

    stage_time = time.time() - stage_start
    total_micro = step * max(grad_accum_steps, 1)
    avg_loss = total_loss / max(total_micro, 1)
    print(f"  [{stage_name}] Done: {step} steps, avg_loss={avg_loss:.4f}, {stage_time:.0f}s", flush=True)
    return step


def mine_hard_negatives(model, tokenizer, triplets: list[dict], top_k: int = 7, batch_size: int = 128):
    """Mine hard negatives using numpy (model.encode_sentences returns numpy)."""
    print("\n=== Hard Negative Mining ===")
    max_len = 128

    all_texts = [t["positive"][:400] for t in triplets]
    query_texts = [t["query"][:400] for t in triplets]

    print(f"  Embedding {len(all_texts)} documents...")
    doc_embs = model.encode_sentences(all_texts, tokenizer, batch_size=batch_size, max_length=max_len)

    print(f"  Embedding {len(query_texts)} queries...")
    query_embs = model.encode_sentences(query_texts, tokenizer, batch_size=batch_size, max_length=max_len)

    print(f"  Mining top-{top_k} hard negatives per query...")
    mined = list(triplets)

    chunk_size = 2000
    for start in range(0, len(query_texts), chunk_size):
        end = min(start + chunk_size, len(query_texts))
        chunk_q = query_embs[start:end]
        sims = chunk_q @ doc_embs.T

        for i in range(end - start):
            idx = start + i
            row = sims[i].copy()
            row[idx] = -1.0
            top_indices = np.argsort(row)[-top_k - 5:][::-1]
            hard_negs = []
            for neg_idx in top_indices:
                sim_val = row[neg_idx]
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

def run_mteb_eval(model, tokenizer, tasks: list[str], output_dir: str = "mteb_results") -> dict:
    """Run MTEB evaluation via the official mteb library."""
    import mteb
    import warnings
    from mteb.models.abs_encoder import AbsEncoder

    class ModelWrapper(AbsEncoder):
        """MTEB AbsEncoder wrapper around our MLX model."""
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
                    emb = self.model.encode_sentences(sentences, self.tokenizer, batch_size=256)
                    all_embs.append(emb)
            if all_embs:
                return np.concatenate(all_embs, axis=0)
            return np.zeros((0, self.model.output_dim))

    wrapper = ModelWrapper(model, tokenizer)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    import signal
    TASK_TIMEOUT = 600  # 10 minutes per task (RedditClustering needs >5min)

    def _timeout_handler(signum, frame):
        raise TimeoutError("MTEB task timed out")

    scores = {}
    for task_name in tasks:
        print(f"  Evaluating: {task_name}...", flush=True)
        try:
            task_objects = mteb.get_tasks(tasks=[task_name], languages=["eng"])
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(TASK_TIMEOUT)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                ev = mteb.MTEB(tasks=task_objects)
                results = ev.run(wrapper, output_folder=output_dir, overwrite_results=True)

            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

            for task_result in results:
                tn = getattr(task_result, 'task_name', None)
                if tn is None:
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
                            scores[tn] = float(score) * 100
                            print(f"  {tn}: {scores[tn]:.2f}", flush=True)
                            break
        except TimeoutError:
            print(f"  WARNING: {task_name} timed out after {TASK_TIMEOUT}s, skipping", flush=True)
            signal.alarm(0)
            continue
        except Exception as e:
            print(f"  WARNING: {task_name} failed: {e}", flush=True)
            continue

    # Fallback: parse from output files if results parsing failed
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
    {"id": "glue", "config": "qqp", "format": "glue_qqp"},
    {"id": "sentence-transformers/stackexchange-duplicates", "config": "title-title-pair", "format": "se_pairs"},
    {"id": "ms_marco", "config": "v2.1", "format": "ms_marco"},
    {"id": "sentence-transformers/natural-questions", "config": None, "format": "nq_pairs"},
    # Reddit title-body pairs: high topic diversity → better clustering signal
    {"id": "sentence-transformers/reddit-title-body", "config": None, "format": "reddit_title_body", "streaming": True},
    # AG News: same-label pairs improve news topic clustering (4 classes)
    {"id": "ag_news", "config": None, "format": "label_pairs"},
    # DBpedia-14: 14 ontology classes (Company, School, Artist, ...) for finer topic separation
    {"id": "fancyzhx/dbpedia_14", "config": None, "format": "label_pairs"},
    # Yahoo Answers Topics: 10 classes for Q&A topic diversity
    {"id": "yahoo_answers_topics", "config": None, "format": "yahoo_answers_label_pairs"},
    # SNLI: entailment pairs for semantic similarity training (strongly correlates with STS)
    {"id": "stanfordnlp/snli", "config": None, "format": "nli"},
    # HotpotQA: question → supporting passage pairs for factual/scientific retrieval
    {"id": "hotpot_qa", "config": "distractor", "format": "hotpotqa_retrieval"},
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--quick-eval-only", action="store_true")
    parser.add_argument("--resume-stage", type=str, default=None,
                        choices=["contrastive", "mining", "finetune", "eval"],
                        help="Skip stages before this one, loading checkpoint from prior stage")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    mx.random.seed(42)

    config = load_config(args.config)
    total_start = time.time()

    print(f"MLX device: {mx.default_device()}")
    print(f"MLX version: {mx.__version__}")

    experiment_desc = os.environ.get("EXPERIMENT_DESC", "unnamed")
    _wandb_run = None
    try:
        import wandb
        _wandb_run = wandb.init(
            project="autoresearch-embed", entity="gourmand-labs",
            name=experiment_desc, config=config,
        )
    except Exception as e:
        print(f"wandb init failed (non-fatal): {e}")
        wandb = None

    from src.model import get_model_config, EmbeddingModel, load_from_safetensors

    model_name = config.get("base_model", "answerdotai/ModernBERT-base")
    print(f"Loading base model: {model_name}")

    bert_config = get_model_config(model_name)
    model = EmbeddingModel(
        bert_config,
        projection_dim=config.get("projection_dim"),
        pooling=config.get("pooling", "mean"),
        normalize=config.get("normalize_embeddings", True),
        simclr_head=config.get("simclr_head", False),
    )
    model = load_from_safetensors(model, model_name)
    mx.eval(model.parameters())

    num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Model parameters: {num_params / 1e6:.1f}M")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_seq_length = int(config.get("max_seq_length", 256))
    grad_accum_steps = int(config.get("memory", {}).get("gradient_accumulation_steps", 1))
    use_grad_ckpt = config.get("memory", {}).get("gradient_checkpointing", False)
    print(f"[config] max_seq_length={max_seq_length}, grad_accum_steps={grad_accum_steps}, grad_ckpt={use_grad_ckpt}")

    # Enable gradient checkpointing to reduce activation memory (~80% savings)
    if use_grad_ckpt:
        model.encoder.gradient_checkpointing = True

    # EMA setup is deferred until after opt_cfg is defined (see below)
    ema_state = None

    max_rows = int(config.get("data", {}).get("max_rows_per_dataset", 30000))
    ds_key = hashlib.sha256(json.dumps({"datasets": DATASETS, "max_rows": max_rows}, sort_keys=True).encode()).hexdigest()[:12]
    cache_path = Path("data_cache") / f"clean_triplets_{ds_key}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached clean triplets from {cache_path}...")
        triplets = json.loads(cache_path.read_text())
        removed = 0
        print(f"Loaded {len(triplets)} clean pairs from cache (decontaminated).")
    else:
        print("Loading training data...")
        triplets = load_training_data(DATASETS, max_rows_per_dataset=max_rows)
        print(f"Total training pairs (pre-decontam): {len(triplets)}")

        print("Building MTEB test LSH for decontamination...")
        from src.data.decontaminate import build_test_lsh, filter_triplets, DECONTAM_TASKS
        test_lsh = build_test_lsh(DECONTAM_TASKS)
        triplets, removed = filter_triplets(triplets, test_lsh)
        print(f"Decontamination removed {removed} samples. Clean pairs: {len(triplets)}")

        cache_path.write_text(json.dumps(triplets))
        print(f"Cached clean triplets to {cache_path}")

    if not triplets:
        print("WARNING: No training data loaded.")
        triplets = []

    stages = config.get("stages", {})
    resume_stage = args.resume_stage
    STAGE_ORDER = ["warmup", "contrastive", "mining", "finetune", "eval"]
    CHECKPOINT_DIR = Path("checkpoints/stages")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def save_stage_checkpoint(stage_name: str):
        """Save model weights after completing a stage."""
        path = CHECKPOINT_DIR / f"after_{stage_name}.npz"
        flat = dict(tree_flatten(model.parameters()))
        mx.savez(str(path), **flat)
        print(f"  [checkpoint] Saved after {stage_name} → {path}", flush=True)

    def load_stage_checkpoint(stage_name: str) -> bool:
        """Load model weights from a stage checkpoint. Returns True if loaded."""
        path = CHECKPOINT_DIR / f"after_{stage_name}.npz"
        if not path.exists():
            print(f"  [checkpoint] ERROR: {path} not found, cannot resume", flush=True)
            return False
        weights = dict(mx.load(str(path)))
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        print(f"  [checkpoint] Loaded from {path}", flush=True)
        return True

    def should_skip(stage_name: str) -> bool:
        """Return True if this stage should be skipped due to --resume-stage."""
        if resume_stage is None:
            return False
        return STAGE_ORDER.index(stage_name) < STAGE_ORDER.index(resume_stage)

    # ---- Helper: filter data by stage config ----
    def select_stage_data(stage_cfg: dict, all_triplets: list) -> list:
        """Select training data based on stage config 'data' field."""
        data_spec = stage_cfg.get("data", "all_curated")
        if data_spec == "all_curated" or data_spec == "all_curated_with_hard_negs":
            return all_triplets
        elif data_spec == "qqp_se_paraphrase":
            filtered = [t for t in all_triplets if "qqp" in t.get("source", "") or "stackexchange" in t.get("source", "") or "reddit" in t.get("source", "")]
            return filtered if filtered else all_triplets
        else:
            # Try matching source name directly
            filtered = [t for t in all_triplets if data_spec in t.get("source", "")]
            return filtered if filtered else all_triplets

    # ---- Stage 1: Warmup ----
    warmup_cfg = stages.get("warmup", {})
    warmup_data = select_stage_data(warmup_cfg, triplets)

    warmup_lr = float(warmup_cfg.get("learning_rate", 1e-4))
    opt_cfg = config.get("optimizer", {})
    weight_decay = float(opt_cfg.get("weight_decay", 0.01))
    opt_betas = opt_cfg.get("betas", [0.9, 0.999])
    opt_eps = float(opt_cfg.get("eps", 1e-8))
    print(f"[config] optimizer: weight_decay={weight_decay}, betas={opt_betas}, eps={opt_eps}")

    llrd_decay = float(opt_cfg.get("llrd_decay", 1.0))  # 1.0 = no decay, 0.95 = typical

    def make_optimizer(lr):
        if llrd_decay < 1.0:
            # Layer-wise learning rate decay: lower layers get smaller LR
            # We scale gradients by (layer_lr / base_lr) so the optimizer's single LR
            # still applies correctly, and weight_decay scales proportionally.
            num_layers = len(model.encoder.layers)

            all_params = dict(tree_flatten(model.parameters()))
            llrd_ratios = {}  # param_name → lr_ratio (multiply gradient by this)

            # Embeddings get lowest ratio
            embed_ratio = llrd_decay ** (num_layers + 1)
            for k in all_params:
                if k.startswith("encoder.tok_embeddings") or k.startswith("encoder.embedding_norm"):
                    llrd_ratios[k] = embed_ratio

            # Each layer gets progressively higher ratio
            for i in range(num_layers):
                ratio = llrd_decay ** (num_layers - i)
                for k in all_params:
                    if k.startswith(f"encoder.layers.{i}"):
                        llrd_ratios[k] = ratio

            # Head params (final_norm, pooling, projection) get ratio=1.0 (full LR)
            # No entry needed — default is 1.0

            print(f"[config] LLRD: decay={llrd_decay}, embed_ratio={embed_ratio:.4f}, layer0_ratio={llrd_decay**num_layers:.4f}, top_layer_ratio={llrd_decay:.4f}")
            opt = optim.AdamW(learning_rate=lr, weight_decay=weight_decay,
                              betas=opt_betas, eps=opt_eps)
            opt._llrd_ratios = llrd_ratios
            return opt
        else:
            return optim.AdamW(learning_rate=lr, weight_decay=weight_decay,
                               betas=opt_betas, eps=opt_eps)

    optimizer = make_optimizer(warmup_lr)

    # EMA: maintain exponential moving average of weights for evaluation
    ema_decay = float(opt_cfg.get("ema_decay", 0.0))
    if ema_decay > 0:
        ema_w = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
        ema_state = {"decay": ema_decay, "weights": ema_w}
        print(f"[config] EMA enabled: decay={ema_decay}")

    # ---- Freeze encoder layers if configured ----
    freeze_n = int(warmup_cfg.get("freeze_encoder_layers", 0))
    if freeze_n > 0:
        for i in range(min(freeze_n, len(model.encoder.layers))):
            model.encoder.layers[i].freeze()
        print(f"[config] Froze first {freeze_n} encoder layers")

    train_start = time.time()

    # Enable SimCLR projection head during training (skipped at eval)
    model.training_mode = True
    if config.get("simclr_head", False):
        print(f"[config] SimCLR projection head enabled (train-only)")

    if should_skip("warmup"):
        print("=== Stage: warmup — SKIPPED (--resume-stage) ===", flush=True)
    elif warmup_data:
        run_training_stage(model, tokenizer, warmup_data, warmup_cfg, optimizer, "warmup", max_seq_length, grad_accum_steps, ema_state)
        save_stage_checkpoint("warmup")

    # Unfreeze all layers for subsequent stages
    if freeze_n > 0:
        for i in range(min(freeze_n, len(model.encoder.layers))):
            model.encoder.layers[i].unfreeze()
        print(f"[config] Unfroze encoder layers for contrastive stage")

    # ---- Stage 2: Full contrastive ----
    contrastive_cfg = stages.get("contrastive", {})
    contrastive_lr = float(contrastive_cfg.get("learning_rate", 5e-5))
    optimizer = make_optimizer(contrastive_lr)

    if should_skip("contrastive"):
        print("=== Stage: contrastive — SKIPPED (--resume-stage) ===", flush=True)
    elif triplets:
        if resume_stage == "contrastive":
            load_stage_checkpoint("warmup")
        run_training_stage(model, tokenizer, triplets, contrastive_cfg, optimizer, "contrastive", max_seq_length, grad_accum_steps, ema_state)
        save_stage_checkpoint("contrastive")

    # Free MLX memory before hard negative mining (inference mode)
    try:
        mx.metal.clear_cache()
    except Exception:
        pass

    # ---- Stage 3: Hard negative mining ----
    # Mining is inference — disable SimCLR head to use raw encoder embeddings
    model.training_mode = False
    mining_cfg = stages.get("hard_neg_mining", {})
    if should_skip("mining"):
        print("=== Stage: hard_neg_mining — SKIPPED (--resume-stage) ===", flush=True)
        triplets_with_negs = triplets
    elif triplets:
        if resume_stage == "mining":
            load_stage_checkpoint("contrastive")
        # Use shuffled subset for mining (avoid bias toward early-loaded datasets)
        mining_max = int(mining_cfg.get("max_samples", 8000))
        mining_pool = list(triplets)
        random.shuffle(mining_pool)
        mining_triplets = mining_pool[:mining_max]
        mining_bs = int(mining_cfg.get("batch_size", 32))
        triplets_with_negs = mine_hard_negatives(
            model, tokenizer, mining_triplets,
            top_k=int(mining_cfg.get("top_k", 7)),
            batch_size=mining_bs,
        )
        save_stage_checkpoint("mining")
    else:
        triplets_with_negs = triplets

    # ---- Stage 4: Hard negative fine-tuning ----
    model.training_mode = True  # re-enable SimCLR head for fine-tuning
    finetuning_cfg = stages.get("fine_tuning", {})
    finetuning_lr = float(finetuning_cfg.get("learning_rate", 1e-5))
    optimizer = make_optimizer(finetuning_lr)

    if should_skip("finetune"):
        print("=== Stage: fine_tuning — SKIPPED (--resume-stage) ===", flush=True)
    elif triplets_with_negs:
        if resume_stage == "finetune":
            load_stage_checkpoint("mining")
        run_training_stage(model, tokenizer, triplets_with_negs, finetuning_cfg, optimizer, "fine_tuning", max_seq_length, grad_accum_steps, ema_state)
        save_stage_checkpoint("finetune")

    # Load latest checkpoint if resuming directly to eval
    if resume_stage == "eval":
        # Try finetune checkpoint first, fall back through the chain
        for ckpt in ["finetune", "mining", "contrastive", "warmup"]:
            if load_stage_checkpoint(ckpt):
                break

    train_time = time.time() - train_start
    print(f"\nTotal training time: {train_time/60:.1f} min")

    # Save checkpoint
    ckpt_dir = Path("checkpoints") / f"exp_{time.strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    weights = dict(tree_flatten(model.parameters()))
    mx.savez(str(ckpt_dir / "model.npz"), **weights)
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    latest = Path("checkpoints/latest")
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(ckpt_dir.resolve())
    print(f"Checkpoint saved: {ckpt_dir}")

    # ---- Evaluation ----
    print("\nRunning MTEB evaluation...")

    # Swap in EMA weights for evaluation (better generalization)
    train_weights = None
    if ema_state is not None:
        from mlx.utils import tree_unflatten
        train_weights = dict(tree_flatten(model.parameters()))
        ema_w = ema_state["weights"]
        mx.eval(list(ema_w.values()))
        model.load_weights(list(ema_w.items()))
        mx.eval(model.parameters())
        print("[EMA] Loaded EMA weights for evaluation")

    # Disable SimCLR head and gradient checkpointing for eval
    model.training_mode = False
    model.encoder.gradient_checkpointing = False
    # Free training memory before eval
    try:
        mx.metal.clear_cache()
    except Exception:
        pass

    eval_tasks = QUICK_TASKS if args.quick_eval_only else FULL_TASKS
    eval_outdir = "mteb_results_quick" if args.quick_eval_only else "mteb_results"

    try:
        scores = run_mteb_eval(model, tokenizer, eval_tasks, output_dir=eval_outdir)
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

    peak_mem = mx.get_peak_memory() / 1024**3
    total_time = time.time() - total_start

    # === DO NOT REMOVE: result.json output required by experiment.py ===
    result_data = {
        "primary_score": round(primary, 4),
        "sts_avg": round(float(sts_avg), 4),
        "pair_class_avg": round(float(pair_avg), 4),
        "cluster_avg": round(float(cluster_avg), 4),
        "retrieval_avg": round(float(retrieval_avg), 4),
        "training_minutes": round(train_time / 60, 1),
        "peak_memory_gb": round(peak_mem, 1),
        "num_params_M": round(num_params / 1e6, 1),
        "base_model": model_name,
        "training_stage": "full_4stage",
        "total_train_pairs": len(triplets),
        "task_scores": {k: round(v, 2) for k, v in scores.items()},
        "config": config,
        "datasets": list({t.get("source", "unknown") for t in triplets}),
        "decontam_removed": removed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "framework": "mlx",
    }
    if _wandb_run is not None:
        result_data["wandb_run_id"] = _wandb_run.id

    with open("result.json", "w") as f:
        json.dump(result_data, f, indent=2)
    with open(ckpt_dir / "result.json", "w") as f:
        json.dump(result_data, f, indent=2)

    manifest_path = Path("data_cache/manifest.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for ds_spec in DATASETS:
        manifest_entry = {
            "id": ds_spec.get("id"),
            "config": ds_spec.get("config"),
            "format": ds_spec.get("format"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(manifest_path, "a") as f:
            f.write(json.dumps(manifest_entry) + "\n")

    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"primary_score": primary, "sts_avg": float(sts_avg),
                        "pair_class_avg": float(pair_avg), "cluster_avg": float(cluster_avg),
                        "retrieval_avg": float(retrieval_avg)})
            for task, score in scores.items():
                wandb.log({f"mteb/{task}": score})
            wandb.finish()
    except Exception as e:
        print(f"wandb finish failed (non-fatal): {e}")
    # === END result.json output ===

    print("\n---")
    print(f"primary_score:     {primary:.4f}")
    print(f"sts_avg:           {float(sts_avg):.4f}")
    print(f"pair_class_avg:    {float(pair_avg):.4f}")
    print(f"cluster_avg:       {float(cluster_avg):.4f}")
    print(f"retrieval_avg:     {float(retrieval_avg):.4f}")
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
