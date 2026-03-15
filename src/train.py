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

def infonce_loss(query_emb: mx.array, positive_emb: mx.array, temperature: float = 0.05) -> mx.array:
    """InfoNCE loss with in-batch negatives (query→positive direction)."""
    sim = mx.matmul(query_emb, positive_emb.T) / temperature
    labels = mx.arange(sim.shape[0])
    lse = mx.logsumexp(sim, axis=1, keepdims=True)
    return -mx.mean((sim - lse)[mx.arange(sim.shape[0]), labels])


def infonce_loss_with_hard_negs(
    query_emb: mx.array, positive_emb: mx.array, hard_neg_emb: mx.array,
    temperature: float = 0.05, hard_neg_weight: float = 2.0,
) -> mx.array:
    """InfoNCE with in-batch negatives plus explicit hard negatives."""
    B = query_emb.shape[0]
    sim_inbatch = mx.matmul(query_emb, positive_emb.T) / temperature
    sim_hardneg = mx.sum(query_emb * hard_neg_emb, axis=-1, keepdims=True) / temperature * hard_neg_weight
    logits = mx.concatenate([sim_inbatch, sim_hardneg], axis=1)
    labels = mx.arange(B)
    log_softmax = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    return -mx.mean(log_softmax[mx.arange(B), labels])


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
):
    """Run one training stage for the configured duration using MLX value_and_grad."""
    duration_s = float(stage_cfg.get("duration_minutes", 10)) * 60
    batch_size = int(stage_cfg.get("batch_size", 128))
    temperature = float(stage_cfg.get("temperature", 0.05))
    max_seq_len = 256
    hard_neg_weight = float(stage_cfg.get("hard_neg_weight", 1.0))

    stage_start = time.time()
    step = 0
    total_loss = 0.0

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
        return infonce_loss(q_emb, p_emb, temperature=temperature)

    loss_grad_fn = nn.value_and_grad(model, loss_fn)

    while time.time() - stage_start < duration_s:
        for batch_start in range(0, len(data), batch_size):
            if time.time() - stage_start >= duration_s:
                break
            batch = data[batch_start:batch_start + batch_size]
            if len(batch) < 2:
                continue

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
            if has_hard_negs and hard_neg_weight > 1.0:
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
            grads = tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            step += 1
            loss_val = float(loss.item())
            total_loss += loss_val

            if step % 50 == 0:
                elapsed = time.time() - stage_start
                avg_loss = total_loss / step
                print(f"  [{stage_name}] Step {step} | loss={avg_loss:.4f} | {elapsed:.0f}s/{duration_s:.0f}s", flush=True)
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"loss": loss_val, "avg_loss": avg_loss, "step": step, "stage": stage_name})
                except Exception:
                    pass

        random.shuffle(data)

    stage_time = time.time() - stage_start
    avg_loss = total_loss / max(step, 1)
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
                    emb = self.model.encode_sentences(sentences, self.tokenizer, batch_size=64)
                    all_embs.append(emb)
            if all_embs:
                return np.concatenate(all_embs, axis=0)
            return np.zeros((0, self.model.output_dim))

    wrapper = ModelWrapper(model, tokenizer)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    task_objects = mteb.get_tasks(tasks=tasks, languages=["eng"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ev = mteb.MTEB(tasks=task_objects)
        results = ev.run(wrapper, output_folder=output_dir, overwrite_results=True)

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
    )
    model = load_from_safetensors(model, model_name)
    mx.eval(model.parameters())

    num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Model parameters: {num_params / 1e6:.1f}M")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds_key = hashlib.sha256(json.dumps(DATASETS, sort_keys=True).encode()).hexdigest()[:12]
    cache_path = Path("data_cache") / f"clean_triplets_{ds_key}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached clean triplets from {cache_path}...")
        triplets = json.loads(cache_path.read_text())
        removed = 0
        print(f"Loaded {len(triplets)} clean pairs from cache (decontaminated).")
    else:
        print("Loading training data...")
        triplets = load_training_data(DATASETS, max_rows_per_dataset=30000)
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

    # ---- Stage 1: Warmup ----
    warmup_cfg = stages.get("warmup", {})
    warmup_data = [t for t in triplets if "qqp" in t.get("source", "") or "stackexchange" in t.get("source", "") or "reddit" in t.get("source", "")]
    if not warmup_data:
        warmup_data = triplets

    warmup_lr = float(warmup_cfg.get("learning_rate", 1e-4))
    weight_decay = float(config.get("optimizer", {}).get("weight_decay", 0.01))
    optimizer = optim.AdamW(learning_rate=warmup_lr, weight_decay=weight_decay)

    train_start = time.time()

    if should_skip("warmup"):
        print("=== Stage: warmup — SKIPPED (--resume-stage) ===", flush=True)
    elif warmup_data:
        run_training_stage(model, tokenizer, warmup_data, warmup_cfg, optimizer, "warmup")
        save_stage_checkpoint("warmup")

    # ---- Stage 2: Full contrastive ----
    contrastive_cfg = stages.get("contrastive", {})
    contrastive_lr = float(contrastive_cfg.get("learning_rate", 5e-5))
    optimizer = optim.AdamW(learning_rate=contrastive_lr, weight_decay=weight_decay)

    if should_skip("contrastive"):
        print("=== Stage: contrastive — SKIPPED (--resume-stage) ===", flush=True)
    elif triplets:
        if resume_stage == "contrastive":
            load_stage_checkpoint("warmup")
        run_training_stage(model, tokenizer, triplets, contrastive_cfg, optimizer, "contrastive")
        save_stage_checkpoint("contrastive")

    # Free MLX memory before hard negative mining (inference mode)
    try:
        mx.metal.clear_cache()
    except Exception:
        pass

    # ---- Stage 3: Hard negative mining ----
    mining_cfg = stages.get("hard_neg_mining", {})
    if should_skip("mining"):
        print("=== Stage: hard_neg_mining — SKIPPED (--resume-stage) ===", flush=True)
        triplets_with_negs = triplets
    elif triplets:
        if resume_stage == "mining":
            load_stage_checkpoint("contrastive")
        # Use subset for mining to avoid OOM on 64GB system
        mining_triplets = triplets[:8000]
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
    finetuning_cfg = stages.get("fine_tuning", {})
    finetuning_lr = float(finetuning_cfg.get("learning_rate", 1e-5))
    optimizer = optim.AdamW(learning_rate=finetuning_lr, weight_decay=weight_decay)

    if should_skip("finetune"):
        print("=== Stage: fine_tuning — SKIPPED (--resume-stage) ===", flush=True)
    elif triplets_with_negs:
        if resume_stage == "finetune":
            load_stage_checkpoint("mining")
        run_training_stage(model, tokenizer, triplets_with_negs, finetuning_cfg, optimizer, "fine_tuning")
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
