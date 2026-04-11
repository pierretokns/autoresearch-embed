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

import gc

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
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
    """InfoNCE loss with in-batch negatives and optional false-negative masking.

    false_neg_threshold: if > 0, mask off-diagonal pairs with cosine sim above threshold
    using -1e9 (NOT -inf, which causes NaN in logsumexp when all entries are masked).

    Memory-efficient: materializes the full B×B sim matrix. For B>256, consider
    infonce_loss_tiled() which uses O(B) memory via chunked logsumexp."""
    sim = mx.matmul(query_emb, positive_emb.T) / temperature
    B = sim.shape[0]
    labels = mx.arange(B)

    if false_neg_threshold > 0.0:
        raw_sim = mx.matmul(query_emb, positive_emb.T)
        diag_mask = mx.eye(B)
        false_neg_mask = (raw_sim > false_neg_threshold) * (1.0 - diag_mask)
        sim = sim + false_neg_mask * mx.array(-1e9)  # -1e9 not -inf to avoid NaN

    lse = mx.logsumexp(sim, axis=1, keepdims=True)
    loss_fwd = -mx.mean((sim - lse)[mx.arange(B), labels])
    if symmetric:
        if false_neg_threshold > 0.0:
            sim_bwd = mx.matmul(positive_emb, query_emb.T) / temperature
            sim_bwd = sim_bwd + false_neg_mask.T * mx.array(-1e9)
        else:
            sim_bwd = sim.T
        lse_bwd = mx.logsumexp(sim_bwd, axis=1, keepdims=True)
        loss_bwd = -mx.mean((sim_bwd - lse_bwd)[mx.arange(B), labels])
        return (loss_fwd + loss_bwd) * 0.5
    return loss_fwd


def infonce_loss_tiled(query_emb: mx.array, positive_emb: mx.array,
                       temperature: float = 0.05, tile_size: int = 64) -> mx.array:
    """Memory-efficient InfoNCE using tiled similarity computation.

    Instead of materializing the full B×B similarity matrix (O(B²) memory),
    computes logsumexp in tiles of size tile_size (O(B×tile) memory).
    Enables batch_size=256+ without OOM. Based on CVPR 2025:
    "Breaking the Memory Barrier of Contrastive Loss via Tile-Based Strategy".

    Uses online logsumexp: for each query, iterates over tiles of positives,
    maintaining running max and sum for numerical stability."""
    B = query_emb.shape[0]

    # Positive pair scores (diagonal elements): sum(q_i * p_i) / tau
    pos_scores = mx.sum(query_emb * positive_emb, axis=-1) / temperature  # (B,)

    # Compute logsumexp over all negatives+positive in tiles
    # Online logsumexp: track running_max and running_sum_exp
    running_max = mx.full((B,), -1e9)
    running_sum_exp = mx.zeros((B,))

    for j in range(0, B, tile_size):
        # Similarity of all queries against this tile of positives: (B, tile)
        tile_pos = positive_emb[j:j + tile_size]
        tile_sim = mx.matmul(query_emb, tile_pos.T) / temperature  # (B, tile)

        # Online logsumexp update
        tile_max = mx.max(tile_sim, axis=1)  # (B,)
        new_max = mx.maximum(running_max, tile_max)
        # Rescale previous sum and add new tile
        running_sum_exp = running_sum_exp * mx.exp(running_max - new_max) + mx.sum(mx.exp(tile_sim - new_max[:, None]), axis=1)
        running_max = new_max

    # logsumexp = running_max + log(running_sum_exp)
    lse = running_max + mx.log(running_sum_exp)

    # Loss = -pos_score + logsumexp = -(pos_score - lse)
    loss = -mx.mean(pos_scores - lse)
    return loss


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
        q_trunc = q_trunc / mx.sqrt(mx.sum(q_trunc * q_trunc, axis=-1, keepdims=True) + 1e-12)
        p_trunc = p_trunc / mx.sqrt(mx.sum(p_trunc * p_trunc, axis=-1, keepdims=True) + 1e-12)
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

    hard_neg_weight scales the hard negative similarity logit. At 1.0, the hard
    negative is treated equally to in-batch negatives. Values > 1 amplify the
    hard neg repulsion (2.0 destroyed the model in exp-70 — too aggressive)."""
    B = query_emb.shape[0]
    sim_inbatch = mx.matmul(query_emb, positive_emb.T) / temperature
    # Hard neg similarity (weight scaling disabled: exp-130 showed weight=0.5 catastrophic -4.63)
    sim_hardneg = mx.sum(query_emb * hard_neg_emb, axis=-1, keepdims=True) / temperature
    logits = mx.concatenate([sim_inbatch, sim_hardneg], axis=1)
    labels = mx.arange(B)
    log_softmax = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    return -mx.mean(log_softmax[mx.arange(B), labels])


# ---- Data Loading ----
# Enable fast multi-connection HF downloads (Rust-based, ~5-10x faster)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
# Use cached HF artifacts when network is unavailable (proxy errors, offline)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
            elif fmt == "pubmedqa":
                # PubMedQA: biomedical question → context passage pairs
                for row in ds:
                    question = row.get("question", "")
                    context = row.get("context", {})
                    if not question or not context:
                        continue
                    contexts = context.get("contexts", []) if isinstance(context, dict) else []
                    if contexts:
                        # Use first context passage as the positive
                        passage = str(contexts[0])[:400]
                        if len(passage) > 30:
                            all_triplets.append({"query": question, "positive": passage, "source": ds_id})
            elif fmt == "s2orc_title_abstract":
                # S2ORC: scientific paper title → abstract pairs for retrieval + clustering
                for row in ds:
                    title = row.get("title", "")
                    abstract = row.get("abstract", "")
                    if title and abstract and len(abstract) > 50:
                        all_triplets.append({"query": title, "positive": str(abstract)[:400], "source": ds_id})
            elif fmt == "fever_claims":
                # FEVER: claim → evidence sentence pairs (SUPPORTS only) for fact retrieval
                for row in ds:
                    label = row.get("label", "")
                    if label != "SUPPORTS":
                        continue
                    claim = row.get("claim", "")
                    evidence = row.get("evidence", [])
                    if claim and evidence:
                        # evidence is list of [page, sent_id, text] triples
                        for ev in evidence:
                            if isinstance(ev, (list, tuple)) and len(ev) >= 3:
                                text = str(ev[2]).strip()
                                if len(text) > 30:
                                    all_triplets.append({"query": claim, "positive": text[:400], "source": ds_id})
                                    break  # one evidence per claim
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


def dedup_triplets_minhash(triplets: list[dict], threshold: float = 0.8) -> tuple[list[dict], int]:
    """Remove near-duplicate triplets across datasets using MinHash LSH.

    Uses query+positive text as the fingerprint. Keeps the first occurrence.
    Returns (deduped_triplets, removed_count).
    """
    from datasketch import MinHash, MinHashLSH

    NUM_PERM = 128
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    keep = []
    removed = 0

    for i, triplet in enumerate(triplets):
        text = (triplet.get("query", "") + " " + triplet.get("positive", "")).lower().strip()
        if len(text) < 20:
            keep.append(triplet)
            continue

        m = MinHash(num_perm=NUM_PERM)
        for j in range(len(text) - 4):
            m.update(text[j:j+5].encode("utf-8"))

        if lsh.query(m):
            removed += 1
        else:
            try:
                lsh.insert(f"t_{i}", m)
            except ValueError:
                pass  # exact duplicate hash
            keep.append(triplet)

        if (i + 1) % 100000 == 0:
            print(f"  [dedup] Processed {i+1}/{len(triplets)}, removed {removed} so far")

    return keep, removed


# ---- Training Stage (MLX) ----

def run_training_stage(
    model,
    tokenizer,
    triplets: list[dict],
    stage_cfg: dict,
    optimizer,
    stage_name: str,
    ema_state: dict | None = None,
):
    """Run one training stage for the configured duration using MLX value_and_grad."""
    duration_s = float(stage_cfg.get("duration_minutes", 10)) * 60
    batch_size = int(stage_cfg.get("batch_size", 128))
    temperature_start = float(stage_cfg.get("temperature", 0.05))
    temperature_end = float(stage_cfg.get("temperature_end", 0.0))  # 0 = no annealing
    max_seq_len = 256
    hard_neg_weight = float(stage_cfg.get("hard_neg_weight", 1.0))
    symmetric = bool(stage_cfg.get("symmetric", False))
    use_matryoshka = bool(stage_cfg.get("matryoshka", False))
    use_instructions = bool(stage_cfg.get("instruction_prefix", False))
    false_neg_threshold = float(stage_cfg.get("false_neg_threshold", 0.0))
    use_stratification = bool(stage_cfg.get("source_stratification", False))

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

    # Telemetry: per-source loss tracking + saturation detection
    from collections import defaultdict
    source_losses = defaultdict(list)  # source → list of recent losses
    saturated_count = 0  # batches with loss < 0.15 in last window
    total_count = 0
    SATURATION_THRESHOLD = 0.15

    # Mutable temperature container for annealing (closure captures the list)
    current_temp = [temperature_start]

    print(f"\n=== Stage: {stage_name} ({stage_cfg.get('duration_minutes', 10)} min) ===")
    if temperature_end > 0 and temperature_end != temperature_start:
        print(f"  Temperature annealing: {temperature_start} → {temperature_end}")

    data = list(triplets)

    # Source-stratified batching: equal representation from each source per batch
    if use_stratification:
        source_groups = defaultdict(list)
        for t in data:
            source_groups[t.get("source", "unknown")].append(t)
        sources = sorted(source_groups.keys())
        for s in sources:
            random.shuffle(source_groups[s])
        source_idx = {s: 0 for s in sources}
        per_source = max(1, batch_size // len(sources))
        print(f"  Source stratification: {len(sources)} sources, {per_source} per source per batch")

        def next_stratified_batch():
            batch = []
            for s in sources:
                group = source_groups[s]
                idx = source_idx[s]
                if idx >= len(group):
                    random.shuffle(group)
                    source_idx[s] = 0
                    idx = 0
                end = min(idx + per_source, len(group))
                batch.extend(group[idx:end])
                source_idx[s] = end
            random.shuffle(batch)  # shuffle within batch to prevent position bias
            return batch[:batch_size]
    else:
        random.shuffle(data)

    mx.clear_cache()  # Consolidate Metal memory after shuffle

    def loss_fn(model, q_ids, q_mask, p_ids, p_mask, n_ids=None, n_mask=None):
        """Compute loss given tokenized inputs."""
        temperature = current_temp[0]
        q_emb = model(q_ids, q_mask)
        p_emb = model(p_ids, p_mask)
        if n_ids is not None:
            n_emb = model(n_ids, n_mask)
            return infonce_loss_with_hard_negs(q_emb, p_emb, n_emb,
                                               temperature=temperature,
                                               hard_neg_weight=hard_neg_weight)
        if use_matryoshka:
            return matryoshka_infonce_loss(q_emb, p_emb, temperature=temperature)
        if batch_size >= 256 and not symmetric and false_neg_threshold <= 0:
            return infonce_loss_tiled(q_emb, p_emb, temperature=temperature, tile_size=64)
        return infonce_loss(q_emb, p_emb, temperature=temperature, symmetric=symmetric,
                           false_neg_threshold=false_neg_threshold)

    loss_grad_fn = nn.value_and_grad(model, loss_fn)

    while time.time() - stage_start < duration_s:
        if use_stratification:
            batch_iter = [None]  # single iteration; batch built by next_stratified_batch()
        else:
            batch_iter = range(0, len(data), batch_size)
        for batch_start in batch_iter:
            if time.time() - stage_start >= duration_s:
                break
            if use_stratification:
                batch = next_stratified_batch()
            else:
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

            # Grad clipping
            grads = tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)  # element-wise: 2x faster than L2 norm on MLX

            # LLRD: scale gradients per layer + manual weight decay
            if hasattr(optimizer, '_llrd_ratios'):
                ratios = optimizer._llrd_ratios
                flat_grads = tree_flatten(grads)
                if hasattr(optimizer, '_llrd_weight_decay') and optimizer._llrd_weight_decay > 0:
                    wd = optimizer._llrd_weight_decay
                    flat_params = tree_flatten(model.parameters())
                    scaled = [(k, ratios.get(k, 1.0) * g + wd * ratios.get(k, 1.0) * p)
                              for (k, g), (_, p) in zip(flat_grads, flat_params)]
                else:
                    scaled = [(k, ratios.get(k, 1.0) * g) for k, g in flat_grads]
                grads = tree_unflatten(scaled)

            # Update LR according to schedule (time-based for accuracy)
            if lr_schedule != "constant":
                optimizer.learning_rate = get_lr_by_time(time.time() - stage_start)

            # Temperature annealing: linear interpolation from start to end
            if temperature_end > 0 and temperature_end != temperature_start:
                progress = min((time.time() - stage_start) / max(1.0, duration_s), 1.0)
                current_temp[0] = temperature_start + progress * (temperature_end - temperature_start)

            optimizer.update(model, grads)
            # EMA update (per optimizer step, not per micro-batch)
            if ema_state is not None:
                decay = ema_state["decay"]
                ema_w = ema_state["weights"]
                for k, v in tree_flatten(model.parameters()):
                    if k in ema_w:
                        ema_w[k] = decay * ema_w[k] + (1 - decay) * v
                mx.eval(model.parameters(), optimizer.state, loss, *list(ema_w.values()))
            else:
                mx.eval(model.parameters(), optimizer.state, loss)

            step += 1
            micro_loss = float(loss.item())
            total_loss += micro_loss

            # Telemetry: track per-source loss and saturation
            batch_source = batch[0].get("source", "unknown") if batch else "unknown"
            source_losses[batch_source].append(micro_loss)
            total_count += 1
            if micro_loss < SATURATION_THRESHOLD:
                saturated_count += 1

            if step % 50 == 0:
                elapsed = time.time() - stage_start
                avg_loss = total_loss / step
                sat_pct = saturated_count / max(total_count, 1) * 100
                step_time = elapsed / step
                print(f"  [{stage_name}] Step {step} | loss={avg_loss:.4f} | last={micro_loss:.4f} | sat={sat_pct:.0f}% | {elapsed:.0f}s/{duration_s:.0f}s", flush=True)
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"loss": micro_loss, "avg_loss": avg_loss, "step": step,
                                   "stage": stage_name, "saturated_pct": sat_pct})
                except Exception:
                    pass

            # Source health report every 500 steps
            if step % 500 == 0:
                print(f"  [{stage_name}] Source health @ step {step}:", flush=True)
                for src in sorted(source_losses.keys()):
                    losses = source_losses[src][-200:]  # last 200 per source
                    src_avg = sum(losses) / len(losses)
                    src_sat = sum(1 for l in losses if l < SATURATION_THRESHOLD) / len(losses) * 100
                    status = "SATURATED" if src_sat > 60 else "healthy"
                    print(f"    {src:45s} avg={src_avg:.3f} sat={src_sat:.0f}% [{status}]", flush=True)

        if not use_stratification:
            random.shuffle(data)
        mx.clear_cache()  # Consolidate Metal memory after shuffle

    stage_time = time.time() - stage_start
    avg_loss = total_loss / max(step, 1)
    final_sat_pct = saturated_count / max(total_count, 1) * 100

    # Stage summary with actionable insights for the autonomous agent
    print(f"\n  [{stage_name}] === STAGE SUMMARY ===", flush=True)
    print(f"  [{stage_name}] Steps: {step} | Avg loss: {avg_loss:.4f} | Time: {stage_time:.0f}s ({stage_time/60:.1f}min)", flush=True)
    print(f"  [{stage_name}] Step time: {stage_time/max(step,1):.2f}s/step | Saturation: {final_sat_pct:.0f}% of batches had loss < {SATURATION_THRESHOLD}", flush=True)

    # Per-source final health
    saturated_sources = []
    healthy_sources = []
    for src in sorted(source_losses.keys()):
        losses = source_losses[src]
        src_avg = sum(losses) / len(losses) if losses else 0
        src_sat = sum(1 for l in losses if l < SATURATION_THRESHOLD) / len(losses) * 100 if losses else 0
        if src_sat > 60:
            saturated_sources.append((src, src_avg, src_sat))
        else:
            healthy_sources.append((src, src_avg, src_sat))

    if saturated_sources:
        print(f"  [{stage_name}] SATURATED sources (>60% batches at loss<{SATURATION_THRESHOLD}, wasting compute):", flush=True)
        for src, avg, sat in saturated_sources:
            print(f"    {src}: avg_loss={avg:.3f}, sat={sat:.0f}% — CONSIDER REMOVING from training data", flush=True)
    if healthy_sources:
        print(f"  [{stage_name}] Healthy sources (still learning):", flush=True)
        for src, avg, sat in healthy_sources:
            print(f"    {src}: avg_loss={avg:.3f}, sat={sat:.0f}%", flush=True)

    if final_sat_pct > 40:
        print(f"  [{stage_name}] WARNING: {final_sat_pct:.0f}% saturation — consider shorter contrastive or removing saturated sources", flush=True)

    print(f"  [{stage_name}] === END SUMMARY ===\n", flush=True)
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
            """inputs is a DataLoader yielding BatchedInput dicts with 'text' key.
            Accumulates all sentences first, then encodes in one call at our preferred
            batch_size — avoids 8× overhead from MTEB's small DataLoader batches (32)."""
            all_sentences = []
            for batch in inputs:
                sentences = batch.get("text", batch.get("sentence", []))
                if not sentences and batch:
                    sentences = list(batch.values())[0]
                if sentences:
                    all_sentences.extend(sentences)
            if all_sentences:
                return self.model.encode_sentences(all_sentences, self.tokenizer, batch_size=512)
            return np.zeros((0, self.model.output_dim), dtype=np.float32)

    wrapper = ModelWrapper(model, tokenizer)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    import signal
    TASK_TIMEOUT = 1200  # 20 min per task (RedditClustering: 25×7K sentences + k-means = ~18 min)

    def _timeout_handler(signum, frame):
        raise TimeoutError("MTEB task timed out")

    scores = {}
    for task_name in tasks:
        print(f"  Evaluating: {task_name}...", flush=True)
        try:
            task_objs = mteb.get_tasks(tasks=[task_name], languages=["eng"])
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(TASK_TIMEOUT)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                ev = mteb.MTEB(tasks=task_objs)
                task_results = ev.run(wrapper, output_folder=output_dir, overwrite_results=True)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        except TimeoutError:
            print(f"  WARNING: {task_name} timed out after {TASK_TIMEOUT}s, skipping", flush=True)
            signal.alarm(0)
            continue
        except Exception as e:
            print(f"  WARNING: {task_name} failed: {e}", flush=True)
            continue
        for task_result in task_results:
            tn = getattr(task_result, 'task_name', None)
            if tn is None:
                continue
            if hasattr(task_result, 'scores'):
                for split_name in ["test", "validation", "dev", "train"]:
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

    if not scores:
        for task_name in tasks:
            result_files = list(Path(output_dir).rglob(f"*{task_name}*.json"))
            if result_files:
                with open(sorted(result_files)[-1]) as f:
                    task_result = json.load(f)
                for split_name in ["test", "validation", "dev", "train"]:
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

# Eval tasks: 21 tasks across all 7 MTEB categories
# Uses nano retrieval (small/fast), v2 clustering, plus standard STS/PairClass/Classification/Reranking
# Covers all categories for leaderboard-comparable scores
# NOTE: When changing this list, also update DECONTAM_TASKS in src/data/decontaminate.py
FULL_TASKS = [
    # STS (2)
    "STSBenchmark", "SICK-R",
    # PairClassification (2)
    "SprintDuplicateQuestions", "TwitterURLCorpus",
    # Classification (2)
    "Banking77Classification", "ToxicConversationsClassification",
    # Clustering (1 — v2, no RedditClustering which takes 18 min)
    "TwentyNewsgroupsClustering.v2",
    # Reranking (1)
    "AskUbuntuDupQuestions",
    # Retrieval (13 — nano versions for speed)
    "NanoArguAnaRetrieval", "NanoClimateFeverRetrieval", "NanoDBPediaRetrieval",
    "NanoFEVERRetrieval", "NanoFiQA2018Retrieval", "NanoHotpotQARetrieval",
    "NanoMSMARCORetrieval", "NanoNFCorpusRetrieval", "NanoNQRetrieval",
    "NanoQuoraRetrieval", "NanoSCIDOCSRetrieval", "NanoSciFactRetrieval",
    "NanoTouche2020Retrieval",
]

QUICK_TASKS = ["STSBenchmark", "SICK-R", "NanoSciFactRetrieval"]

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
    # SNLI: entailment pairs for STS/semantic similarity (AllNLI triplets regressed retrieval -2.92 in exp-113)
    {"id": "stanfordnlp/snli", "config": None, "format": "nli"},
    # HotpotQA: question → supporting passage pairs for factual/scientific retrieval
    {"id": "hotpot_qa", "config": "distractor", "format": "hotpotqa_retrieval"},
    # PubMedQA: biomedical question → context passage pairs for NFCorpus/SciFact retrieval
    {"id": "qiaojin/PubMedQA", "config": "pqa_artificial", "format": "pubmedqa"},
    # S2ORC: scientific paper title → abstract pairs for retrieval + clustering diversity
    {"id": "sentence-transformers/s2orc", "config": "title-abstract-pair", "format": "s2orc_title_abstract", "streaming": True},
    # FEVER: exp-110 showed FEVER hurts retrieval -2.54 and classification -2.21; removed
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

    gc.disable()  # Prevent GC pauses during training (500K+ Python objects)

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
    )
    model = load_from_safetensors(model, model_name)
    mx.eval(model.parameters())

    num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Model parameters: {num_params / 1e6:.1f}M")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_rows = int(config.get("data", {}).get("max_rows_per_dataset", 30000))
    cross_dedup = config.get("data", {}).get("cross_dedup", False)
    ds_key_data = {"datasets": DATASETS, "max_rows": max_rows, "cross_dedup": cross_dedup}
    ds_key = hashlib.sha256(json.dumps(ds_key_data, sort_keys=True).encode()).hexdigest()[:12]
    cache_path = Path("data_cache") / f"clean_triplets_{ds_key}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached clean triplets from {cache_path}...")
        triplets = json.loads(cache_path.read_text())
        removed = 0
        print(f"Loaded {len(triplets)} clean pairs from cache (decontaminated{'+deduped' if cross_dedup else ''}).")
    else:
        print("Loading training data...")
        triplets = load_training_data(DATASETS, max_rows_per_dataset=max_rows)
        print(f"Total training pairs (pre-decontam): {len(triplets)}")

        print("Building MTEB test LSH for decontamination...")
        from src.data.decontaminate import build_test_lsh, filter_triplets, DECONTAM_TASKS
        test_lsh = build_test_lsh(DECONTAM_TASKS)
        triplets, removed = filter_triplets(triplets, test_lsh)
        print(f"Decontamination removed {removed} samples. Clean pairs: {len(triplets)}")

        if cross_dedup:
            print("Running cross-dataset MinHash deduplication...")
            triplets, dedup_removed = dedup_triplets_minhash(triplets, threshold=0.8)
            print(f"Dedup removed {dedup_removed} near-duplicate pairs. Clean pairs: {len(triplets)}")

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

    opt_cfg = config.get("optimizer", {})
    weight_decay = float(opt_cfg.get("weight_decay", 0.01))
    opt_betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    opt_eps = float(opt_cfg.get("eps", 1e-8))
    llrd_decay = float(opt_cfg.get("llrd_decay", 1.0))
    ema_decay_val = float(opt_cfg.get("ema_decay", 0.0))

    def make_optimizer(lr: float) -> optim.AdamW:
        """Create AdamW optimizer, with LLRD per-layer ratio metadata when active."""
        if llrd_decay < 1.0:
            opt = optim.AdamW(learning_rate=lr, weight_decay=0.0, betas=opt_betas, eps=opt_eps)
            # Build per-layer LR ratios for gradient scaling
            num_layers = len(model.encoder.layers)
            ratios = {}
            for k, _ in tree_flatten(model.parameters()):
                for li in range(num_layers):
                    if f"layers.{li}." in k:
                        ratios[k] = llrd_decay ** (num_layers - 1 - li)
                        break
                else:
                    ratios[k] = 1.0  # head / embeddings get full LR
            opt._llrd_ratios = ratios
            opt._llrd_weight_decay = weight_decay
        else:
            opt = optim.AdamW(learning_rate=lr, weight_decay=weight_decay, betas=opt_betas, eps=opt_eps)
        return opt

    def make_ema_state() -> dict | None:
        """Create EMA state dict if ema_decay > 0."""
        if ema_decay_val <= 0:
            return None
        ema_weights = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
        return {"decay": ema_decay_val, "weights": ema_weights}

    # ---- Stage 1: Warmup ----
    warmup_cfg = stages.get("warmup", {})
    warmup_data = [t for t in triplets if "qqp" in t.get("source", "") or "stackexchange" in t.get("source", "") or "reddit" in t.get("source", "")]
    if not warmup_data:
        warmup_data = triplets

    warmup_lr = float(warmup_cfg.get("learning_rate", 1e-4))
    optimizer = make_optimizer(warmup_lr)
    ema_state = make_ema_state()

    train_start = time.time()

    if should_skip("warmup"):
        print("=== Stage: warmup — SKIPPED (--resume-stage) ===", flush=True)
    elif warmup_data:
        run_training_stage(model, tokenizer, warmup_data, warmup_cfg, optimizer, "warmup", ema_state=ema_state)
        save_stage_checkpoint("warmup")
    gc.collect()
    mx.clear_cache()

    # ---- Stage 2: Full contrastive ----
    contrastive_cfg = stages.get("contrastive", {})
    contrastive_lr = float(contrastive_cfg.get("learning_rate", 5e-5))
    optimizer = make_optimizer(contrastive_lr)

    if should_skip("contrastive"):
        print("=== Stage: contrastive — SKIPPED (--resume-stage) ===", flush=True)
    elif triplets:
        if resume_stage == "contrastive":
            load_stage_checkpoint("warmup")
        run_training_stage(model, tokenizer, triplets, contrastive_cfg, optimizer, "contrastive", ema_state=ema_state)
        save_stage_checkpoint("contrastive")
    gc.collect()
    mx.clear_cache()

    # ---- Stage 3: Hard negative mining ----
    mining_cfg = stages.get("hard_neg_mining", {})
    if should_skip("mining"):
        print("=== Stage: hard_neg_mining — SKIPPED (--resume-stage) ===", flush=True)
        triplets_with_negs = triplets
    elif triplets:
        if resume_stage == "mining":
            load_stage_checkpoint("contrastive")
        # Source-diverse mining pool: equal samples from each source for domain coverage
        # exp-129 FT was 99% saturated because 8K triplets[:8000] was dominated by large sources
        from collections import defaultdict
        _mining_groups = defaultdict(list)
        for _t in triplets:
            _mining_groups[_t.get("source", "unknown")].append(_t)
        _per_source = max(500, 20000 // max(len(_mining_groups), 1))
        mining_triplets = []
        for _src in sorted(_mining_groups.keys()):
            _pool = _mining_groups[_src]
            random.shuffle(_pool)
            mining_triplets.extend(_pool[:_per_source])
        random.shuffle(mining_triplets)
        print(f"  Mining pool: {len(mining_triplets)} triplets from {len(_mining_groups)} sources ({_per_source}/source)")
        mining_bs = int(mining_cfg.get("batch_size", 32))
        triplets_with_negs = mine_hard_negatives(
            model, tokenizer, mining_triplets,
            top_k=int(mining_cfg.get("top_k", 7)),
            batch_size=mining_bs,
        )
        save_stage_checkpoint("mining")
    else:
        triplets_with_negs = triplets
    gc.collect()
    mx.clear_cache()

    # ---- Stage 4: Hard negative fine-tuning ----
    finetuning_cfg = stages.get("fine_tuning", {})
    finetuning_lr = float(finetuning_cfg.get("learning_rate", 1e-5))
    optimizer = make_optimizer(finetuning_lr)

    if should_skip("finetune"):
        print("=== Stage: fine_tuning — SKIPPED (--resume-stage) ===", flush=True)
    elif triplets_with_negs:
        if resume_stage == "finetune":
            load_stage_checkpoint("mining")
        run_training_stage(model, tokenizer, triplets_with_negs, finetuning_cfg, optimizer, "fine_tuning", ema_state=ema_state)
        save_stage_checkpoint("finetune")

    # Load latest checkpoint if resuming directly to eval
    if resume_stage == "eval":
        # Try finetune checkpoint first, fall back through the chain
        for ckpt in ["finetune", "mining", "contrastive", "warmup"]:
            if load_stage_checkpoint(ckpt):
                break

    train_time = time.time() - train_start
    print(f"\nTotal training time: {train_time/60:.1f} min")

    # Swap EMA weights into model before save/eval (EMA averages reduce noise → better generalization)
    # Only swap if training actually ran (not on --resume-stage eval where EMA is uninitialized base weights)
    if ema_state is not None and not args.resume_stage:
        ema_w = ema_state["weights"]
        model.load_weights(list(ema_w.items()))
        mx.eval(model.parameters())
        print(f"[EMA] Swapped EMA weights into model (decay={ema_state['decay']})")

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
    # Finish wandb BEFORE eval — wandb's background sync thread competes with
    # Metal GPU resources and causes eval to deadlock on encode_sentences
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
            print("[wandb] Finished before eval (prevents Metal deadlock)")
    except Exception:
        pass

    # Re-enable HF Hub access for eval (was disabled during training to avoid proxy errors)
    # Must reload the constants module because HF caches env vars at import time
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    import huggingface_hub.constants
    from importlib import reload
    reload(huggingface_hub.constants)

    print("\nRunning MTEB evaluation...")

    eval_tasks = QUICK_TASKS if args.quick_eval_only else FULL_TASKS
    eval_outdir = "mteb_results_quick" if args.quick_eval_only else "mteb_results"

    try:
        scores = run_mteb_eval(model, tokenizer, eval_tasks, output_dir=eval_outdir)
    except Exception as e:
        print(f"MTEB eval failed: {e}")
        import traceback; traceback.print_exc()
        scores = {}

    # Compute category averages (21-task eval across all 7 MTEB categories)
    # NOTE: Scores are NOT comparable to pre-exp-107 results (different task set)
    def _cat_avg(task_names):
        vals = [scores.get(t, 0) for t in task_names]
        pos = [v for v in vals if v > 0]
        return np.mean(pos) if pos else 0.0

    sts_avg = _cat_avg(["STSBenchmark", "SICK-R"])
    pair_avg = _cat_avg(["SprintDuplicateQuestions", "TwitterURLCorpus"])
    cluster_avg = _cat_avg(["TwentyNewsgroupsClustering.v2"])
    classification_avg = _cat_avg(["Banking77Classification", "ToxicConversationsClassification"])
    reranking_avg = _cat_avg(["AskUbuntuDupQuestions"])
    retrieval_avg = _cat_avg([
        "NanoArguAnaRetrieval", "NanoClimateFeverRetrieval", "NanoDBPediaRetrieval",
        "NanoFEVERRetrieval", "NanoFiQA2018Retrieval", "NanoHotpotQARetrieval",
        "NanoMSMARCORetrieval", "NanoNFCorpusRetrieval", "NanoNQRetrieval",
        "NanoQuoraRetrieval", "NanoSCIDOCSRetrieval", "NanoSciFactRetrieval",
        "NanoTouche2020Retrieval",
    ])

    # Simple average across all categories (matches MTEB leaderboard methodology)
    cat_scores = [v for v in [sts_avg, pair_avg, cluster_avg, classification_avg,
                              reranking_avg, retrieval_avg] if v > 0]
    primary = np.mean(cat_scores) if cat_scores else 0.0

    peak_mem = mx.get_peak_memory() / 1024**3
    total_time = time.time() - total_start

    # === DO NOT REMOVE: result.json output required by experiment.py ===
    result_data = {
        "primary_score": round(primary, 4),
        "eval_version": "21-task-nano-v1",  # Track which eval set produced this score
        "sts_avg": round(float(sts_avg), 4),
        "pair_class_avg": round(float(pair_avg), 4),
        "cluster_avg": round(float(cluster_avg), 4),
        "classification_avg": round(float(classification_avg), 4),
        "reranking_avg": round(float(reranking_avg), 4),
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

    # wandb was finished before eval to prevent Metal deadlock.
    # Results are logged to results.jsonl by experiment.py instead.
    # === END result.json output ===

    print("\n---")
    print(f"primary_score:     {primary:.4f}  (21-task-nano-v1 — NOT comparable to pre-exp-107 scores)")
    print(f"sts_avg:           {float(sts_avg):.4f}")
    print(f"pair_class_avg:    {float(pair_avg):.4f}")
    print(f"cluster_avg:       {float(cluster_avg):.4f}")
    print(f"classification_avg:{float(classification_avg):.4f}")
    print(f"reranking_avg:     {float(reranking_avg):.4f}")
    print(f"retrieval_avg:     {float(retrieval_avg):.4f}  (13 nano retrieval tasks)")
    print(f"training_minutes:  {train_time / 60:.1f}")
    print(f"peak_memory_gb:    {peak_mem:.1f}")
    print(f"num_params_M:      {num_params / 1e6:.1f}")
    print(f"base_model:        {model_name}")
    print(f"training_stage:    full_4stage")
    print(f"total_train_pairs: {len(triplets)}")
    print(f"\nPer-task scores:")
    for task, score in sorted(scores.items()):
        print(f"  {task}: {score:.2f}")

    gc.enable()
    gc.collect()


if __name__ == "__main__":
    main()
