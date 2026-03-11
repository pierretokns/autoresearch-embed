"""
MinHash-based decontamination filter for training data.
Removes training samples that overlap with MTEB test sets.

This module is in src/data/ and MUST NOT import from src/eval/.
It independently computes MinHash signatures to detect overlap.

Usage:
    from src.data.decontaminate import build_test_lsh, filter_triplets
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datasketch import MinHash, MinHashLSH

NUM_PERM = 128
JACCARD_THRESHOLD = 0.8  # matches contamination.py threshold


def _text_to_minhash(text: str) -> MinHash:
    """Convert text to MinHash signature using character 5-grams."""
    m = MinHash(num_perm=NUM_PERM)
    text = text.lower().strip()
    for i in range(len(text) - 4):
        m.update(text[i:i+5].encode("utf-8"))
    return m


def build_test_lsh(task_names: list[str]) -> MinHashLSH:
    """
    Build an LSH index of MTEB test set texts.
    Loads test data directly via mteb library (separate from training code).
    """
    import mteb

    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM)
    idx = 0

    for task_name in task_names:
        try:
            tasks = mteb.get_tasks(tasks=[task_name], languages=["eng"])
            if not tasks:
                continue
            task = tasks[0]
            task.load_data()

            if hasattr(task, "dataset") and isinstance(task.dataset, dict):
                for split_name in ["test", "validation"]:
                    if split_name not in task.dataset:
                        continue
                    split = task.dataset[split_name]
                    for col in split.column_names:
                        if any(kw in col.lower() for kw in ["text", "sentence", "query", "document"]):
                            for text in split[col]:
                                if text and len(str(text)) > 20:
                                    mh = _text_to_minhash(str(text))
                                    try:
                                        lsh.insert(f"test_{idx}", mh)
                                        idx += 1
                                    except ValueError:
                                        pass  # duplicate
        except Exception as e:
            print(f"  [decontaminate] Warning: could not load {task_name}: {e}")

    print(f"  [decontaminate] Built LSH index with {idx} test texts from {len(task_names)} tasks")
    return lsh


def filter_triplets(
    triplets: list[dict],
    lsh: MinHashLSH,
    text_fields: tuple[str, ...] = ("query", "positive"),
) -> tuple[list[dict], int]:
    """
    Filter out triplets where any text field matches the test set LSH.

    Returns:
        (clean_triplets, removed_count)
    """
    clean = []
    removed = 0

    for triplet in triplets:
        contaminated = False
        for field in text_fields:
            text = triplet.get(field, "")
            if text and len(text) > 20:
                mh = _text_to_minhash(str(text))
                if lsh.query(mh):
                    contaminated = True
                    break
        if contaminated:
            removed += 1
        else:
            clean.append(triplet)

    return clean, removed


# Target tasks for decontamination (must match contamination.py TARGET_TASKS)
DECONTAM_TASKS = [
    "STSBenchmark",
    "SICK-R",
    "TwitterURLCorpus",
    "SprintDuplicateQuestions",
    "TwentyNewsgroupsClustering",
    "RedditClustering",
    "SciFact",
    "NFCorpus",
]
