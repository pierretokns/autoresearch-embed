"""
Train/test overlap checker using MinHash for near-duplicate detection.
Gates every full MTEB evaluation — FAIL = automatic experiment rejection.

*** READ-ONLY — DO NOT MODIFY THIS FILE ***

Threshold: per-task contamination rate must be below 1%.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from datasets import load_dataset

from src.eval.mteb_runner import TARGET_TASKS, TASK_CATEGORIES

# MinHash configuration
NUM_PERM = 128
JACCARD_THRESHOLD = 0.8
CONTAMINATION_THRESHOLD = 0.01  # 1% max overlap per task

# Cache test set hashes to avoid recomputing
_test_set_cache: dict[str, list[tuple[str, MinHash]]] = {}


def _text_to_minhash(text: str) -> MinHash:
    """Convert text to MinHash signature using character n-grams."""
    m = MinHash(num_perm=NUM_PERM)
    # Use character 5-grams
    text = text.lower().strip()
    for i in range(len(text) - 4):
        m.update(text[i:i+5].encode("utf-8"))
    return m


def _load_test_texts(task_name: str) -> list[str]:
    """Load test set texts for an MTEB task."""
    # This uses the mteb library's data loading to get official test sets
    import mteb

    tasks = mteb.get_tasks(tasks=[task_name])
    if not tasks:
        return []

    task = tasks[0]
    task.load_data()

    texts = []
    # Extract text from whatever format the task uses
    if hasattr(task, "dataset"):
        ds = task.dataset
        if isinstance(ds, dict):
            for split_name in ["test", "validation"]:
                if split_name in ds:
                    split = ds[split_name]
                    for col in split.column_names:
                        if any(keyword in col.lower() for keyword in ["text", "sentence", "query", "document"]):
                            texts.extend(str(x) for x in split[col] if x)

    return texts


def get_test_hashes(task_name: str) -> list[tuple[str, MinHash]]:
    """Get MinHash signatures for a task's test set. Cached."""
    if task_name not in _test_set_cache:
        texts = _load_test_texts(task_name)
        _test_set_cache[task_name] = [(t, _text_to_minhash(t)) for t in texts if len(t) > 20]
    return _test_set_cache[task_name]


def check_contamination(training_texts: list[str], tasks: list[str] | None = None) -> dict:
    """
    Check for overlap between training data and MTEB test sets.

    Args:
        training_texts: list of all training texts (queries + positives)
        tasks: MTEB tasks to check against (defaults to TARGET_TASKS)

    Returns:
        {
            "passed": bool,
            "per_task": {task_name: {"rate": float, "count": int, "total": int}},
            "overall_rate": float,
        }
    """
    tasks = tasks or TARGET_TASKS

    # Build LSH index for training data
    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM)
    train_hashes = []
    for i, text in enumerate(training_texts):
        if len(text) > 20:
            mh = _text_to_minhash(text)
            train_hashes.append((text, mh))
            try:
                lsh.insert(f"train_{i}", mh)
            except ValueError:
                pass  # duplicate key

    per_task = {}
    total_contaminated = 0
    total_test = 0

    for task_name in tasks:
        test_items = get_test_hashes(task_name)
        if not test_items:
            per_task[task_name] = {"rate": 0.0, "count": 0, "total": 0}
            continue

        contaminated = 0
        for text, mh in test_items:
            candidates = lsh.query(mh)
            if candidates:
                contaminated += 1

        rate = contaminated / len(test_items) if test_items else 0.0
        per_task[task_name] = {
            "rate": rate,
            "count": contaminated,
            "total": len(test_items),
        }
        total_contaminated += contaminated
        total_test += len(test_items)

    overall_rate = total_contaminated / total_test if total_test > 0 else 0.0
    passed = all(info["rate"] < CONTAMINATION_THRESHOLD for info in per_task.values())

    return {
        "passed": passed,
        "per_task": per_task,
        "overall_rate": overall_rate,
    }


def print_contamination_report(result: dict):
    """Print a human-readable contamination report."""
    status = "PASS" if result["passed"] else "FAIL"
    print(f"Contamination check: {status}")
    print(f"Overall rate: {result['overall_rate']:.4f}")
    for task, info in sorted(result["per_task"].items()):
        flag = " !!!" if info["rate"] >= CONTAMINATION_THRESHOLD else ""
        print(f"  {task}: {info['rate']:.4f} ({info['count']}/{info['total']}){flag}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check training data for MTEB contamination")
    parser.add_argument("--training-manifest", required=True, help="Path to dataset manifest")
    args = parser.parse_args()

    from src.data.registry import get_all_datasets
    print(f"Would check contamination for manifest: {args.training_manifest}")
    print(f"Checking against {len(TARGET_TASKS)} MTEB tasks")
