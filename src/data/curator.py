"""
Automated dataset discovery from HuggingFace Hub and Kaggle.
Searches, filters, normalizes to triplet format, and registers in manifest.

This file is AGENT-MUTABLE.
"""

from __future__ import annotations

from huggingface_hub import HfApi
from datasets import load_dataset

from src.data.registry import register_dataset, compute_content_hash


# Known-good embedding training datasets (seeds)
SEED_DATASETS = [
    {"id": "sentence-transformers/all-nli", "config": "triplet", "format": "triplet"},
    {"id": "sentence-transformers/stsb", "config": None, "format": "pair_score"},
    {"id": "ms_marco", "config": "v2.1", "format": "retrieval"},
    {"id": "quora", "config": None, "format": "duplicate"},
    {"id": "snli", "config": None, "format": "nli"},
    {"id": "multi_nli", "config": None, "format": "nli"},
]

# Search keywords for HF Hub discovery
SEARCH_KEYWORDS = [
    "sentence similarity",
    "semantic textual similarity",
    "paraphrase detection",
    "question answering pairs",
    "passage retrieval",
    "text matching",
    "duplicate detection",
    "natural language inference",
]

MAX_DOWNLOAD_SIZE_GB = 10


def search_hf_datasets(keyword: str, max_results: int = 20) -> list[dict]:
    """Search HuggingFace Hub for embedding-relevant datasets."""
    api = HfApi()
    results = api.list_datasets(search=keyword, sort="downloads", direction=-1, limit=max_results)
    found = []
    for ds in results:
        info = {
            "id": ds.id,
            "downloads": ds.downloads,
            "tags": ds.tags or [],
            "size_bytes": getattr(ds, "size", None),
        }
        # Skip very large datasets
        if info["size_bytes"] and info["size_bytes"] > MAX_DOWNLOAD_SIZE_GB * 1e9:
            continue
        found.append(info)
    return found


def normalize_nli(dataset_id: str, config: str | None = None, split: str = "train", max_rows: int = 500_000):
    """
    Normalize NLI dataset to triplet format.
    Entailment = positive, Contradiction = hard negative.
    """
    ds = load_dataset(dataset_id, config, split=split)
    if len(ds) > max_rows:
        ds = ds.select(range(max_rows))

    triplets = []
    for row in ds:
        label = row.get("label", -1)
        premise = row.get("premise", row.get("sentence1", ""))
        hypothesis = row.get("hypothesis", row.get("sentence2", ""))
        if label == 0:  # entailment
            triplets.append({"query": premise, "positive": hypothesis, "negatives": []})
        # contradiction pairs can be added as hard negatives later

    content_hash = compute_content_hash([t["query"] for t in triplets])
    register_dataset(
        source=f"huggingface:{dataset_id}",
        config=config,
        split=split,
        row_count=len(triplets),
        columns=["query", "positive", "negatives"],
        content_hash=content_hash,
        normalization="nli_entailment_as_positive",
    )
    return triplets


def normalize_triplet(dataset_id: str, config: str | None = None, split: str = "train", max_rows: int = 500_000):
    """Normalize a dataset already in triplet format."""
    ds = load_dataset(dataset_id, config, split=split)
    if len(ds) > max_rows:
        ds = ds.select(range(max_rows))

    triplets = []
    cols = ds.column_names
    # Try common column name patterns
    anchor_col = next((c for c in cols if c in ("anchor", "query", "sentence1", "text1")), cols[0])
    pos_col = next((c for c in cols if c in ("positive", "pos", "sentence2", "text2")), cols[1] if len(cols) > 1 else cols[0])
    neg_col = next((c for c in cols if c in ("negative", "neg", "sentence3", "text3")), None)

    for row in ds:
        t = {"query": str(row[anchor_col]), "positive": str(row[pos_col]), "negatives": []}
        if neg_col and row.get(neg_col):
            t["negatives"] = [str(row[neg_col])]
        triplets.append(t)

    content_hash = compute_content_hash([t["query"] for t in triplets])
    register_dataset(
        source=f"huggingface:{dataset_id}",
        config=config,
        split=split,
        row_count=len(triplets),
        columns=["query", "positive", "negatives"],
        content_hash=content_hash,
        normalization="triplet_direct",
    )
    return triplets


def discover_and_register_all(seeds_only: bool = True):
    """Run discovery pipeline. If seeds_only, skip HF Hub search."""
    all_datasets = {}

    for seed in SEED_DATASETS:
        print(f"Processing seed: {seed['id']}")
        try:
            if seed["format"] == "nli":
                data = normalize_nli(seed["id"], seed.get("config"))
            else:
                data = normalize_triplet(seed["id"], seed.get("config"))
            all_datasets[seed["id"]] = data
            print(f"  -> {len(data)} triplets")
        except Exception as e:
            print(f"  -> FAILED: {e}")

    if not seeds_only:
        for keyword in SEARCH_KEYWORDS:
            print(f"Searching HF Hub: '{keyword}'")
            candidates = search_hf_datasets(keyword)
            for c in candidates[:5]:  # limit per keyword
                if c["id"] not in all_datasets and c["id"] not in {s["id"] for s in SEED_DATASETS}:
                    print(f"  Candidate: {c['id']} ({c['downloads']} downloads)")
                    # Could auto-download and normalize here

    return all_datasets
