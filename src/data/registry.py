"""
Dataset manifest manager. Tracks every dataset used for training with full provenance.
Append-only: never delete entries, only add or update contamination status.

This file is AGENT-MUTABLE.
"""

import hashlib
import json
import time
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent.parent.parent / "data_cache" / "manifest.json"


def _load_manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return []


def _save_manifest(entries: list[dict]):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def register_dataset(
    source: str,  # e.g. "huggingface:sentence-transformers/all-nli"
    config: str | None = None,
    split: str = "train",
    row_count: int = 0,
    columns: list[str] | None = None,
    content_hash: str = "",
    normalization: str = "",  # how it was converted to triplet format
) -> dict:
    """Register a new dataset in the manifest."""
    entries = _load_manifest()
    entry = {
        "source": source,
        "config": config,
        "split": split,
        "row_count": row_count,
        "columns": columns or [],
        "content_hash": content_hash,
        "normalization": normalization,
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "contamination_checked": False,
        "contamination_rate": None,
        "contamination_details": None,
    }
    entries.append(entry)
    _save_manifest(entries)
    return entry


def update_contamination(source: str, rate: float, details: str):
    """Update contamination check results for a dataset."""
    entries = _load_manifest()
    for entry in entries:
        if entry["source"] == source:
            entry["contamination_checked"] = True
            entry["contamination_rate"] = rate
            entry["contamination_details"] = details
    _save_manifest(entries)


def get_clean_datasets() -> list[dict]:
    """Return all datasets that passed contamination checks."""
    entries = _load_manifest()
    return [e for e in entries if e["contamination_checked"] and (e["contamination_rate"] or 0) < 0.01]


def get_all_datasets() -> list[dict]:
    """Return all registered datasets."""
    return _load_manifest()


def compute_content_hash(texts: list[str]) -> str:
    """Compute a hash of text content for deduplication."""
    h = hashlib.sha256()
    for t in sorted(texts[:10000]):  # hash first 10K sorted texts
        h.update(t.encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]
