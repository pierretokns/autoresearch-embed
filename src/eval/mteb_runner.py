"""
Official MTEB evaluation wrapper.

*** READ-ONLY — DO NOT MODIFY THIS FILE ***

Uses the official `mteb` library. The MTEBModelWrapper is the only bridge
between our MLX model and the MTEB evaluation framework.

Anti-cheat: this file must use the official mteb evaluation code path.
No custom scoring implementations. No reimplemented metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mteb
import numpy as np
from transformers import AutoTokenizer

# These are the target tasks — do not change without discussion
TARGET_TASKS = [
    # STS
    "STSBenchmark",
    "SICK-R",
    # Pair Classification
    "TwitterURLCorpus",
    "SprintDuplicateQuestions",
    # Clustering
    "TwentyNewsgroupsClustering",
    "RedditClustering",
    # Retrieval
    "SciFact",
    "NFCorpus",
]

# Weights for primary score calculation
CATEGORY_WEIGHTS = {
    "STS": 0.3,
    "PairClassification": 0.2,
    "Clustering": 0.2,
    "Retrieval": 0.3,
}

TASK_CATEGORIES = {
    "STSBenchmark": "STS",
    "SICK-R": "STS",
    "TwitterURLCorpus": "PairClassification",
    "SprintDuplicateQuestions": "PairClassification",
    "TwentyNewsgroupsClustering": "Clustering",
    "RedditClustering": "Clustering",
    "SciFact": "Retrieval",
    "NFCorpus": "Retrieval",
}


class MTEBModelWrapper:
    """
    Wraps an MLX embedding model for use with the official MTEB library.
    Implements the encode() interface that MTEB expects.
    """

    def __init__(self, model, tokenizer, batch_size: int = 64, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        """Encode sentences to embeddings. This is what MTEB calls."""
        batch_size = kwargs.get("batch_size", self.batch_size)
        return self.model.encode_sentences(
            sentences,
            self.tokenizer,
            batch_size=batch_size,
            max_length=self.max_length,
        )


def evaluate(model, tokenizer, tasks: list[str] | None = None, output_dir: str = "mteb_results") -> dict:
    """
    Run official MTEB evaluation.

    Args:
        model: MLX EmbeddingModel instance
        tokenizer: HuggingFace tokenizer
        tasks: list of task names (defaults to TARGET_TASKS)
        output_dir: where to save MTEB result JSON files

    Returns:
        dict with per-task scores and primary composite score
    """
    tasks = tasks or TARGET_TASKS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = MTEBModelWrapper(model, tokenizer)

    # Run official MTEB evaluation
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(wrapper, output_folder=str(output_dir))

    # Parse results from output files
    scores = {}
    for task_name in tasks:
        result_files = list(output_dir.glob(f"**/{task_name}*.json"))
        if result_files:
            with open(result_files[0]) as f:
                task_result = json.load(f)
            # Extract main score (MTEB stores it in the test split)
            for split_name in ["test", "validation", "dev"]:
                if split_name in task_result:
                    split_data = task_result[split_name]
                    if isinstance(split_data, dict):
                        # Get the primary metric for this task type
                        score = split_data.get("main_score", split_data.get("cos_sim", {}).get("spearman", 0))
                        if isinstance(score, dict):
                            score = score.get("spearman", score.get("main_score", 0))
                        scores[task_name] = float(score) * 100  # Convert to percentage
                    break

    # Compute category averages and primary score
    category_scores = {}
    for cat in CATEGORY_WEIGHTS:
        cat_tasks = [t for t, c in TASK_CATEGORIES.items() if c == cat and t in scores]
        if cat_tasks:
            category_scores[cat] = sum(scores[t] for t in cat_tasks) / len(cat_tasks)

    primary = sum(
        CATEGORY_WEIGHTS.get(cat, 0) * category_scores.get(cat, 0)
        for cat in CATEGORY_WEIGHTS
    )

    return {
        "primary_score": primary,
        "category_scores": category_scores,
        "task_scores": scores,
    }


def print_results(results: dict):
    """Print results in the structured format the experiment loop parses."""
    print("---")
    print(f"primary_score:     {results['primary_score']:.4f}")
    for cat, score in sorted(results.get("category_scores", {}).items()):
        print(f"{cat.lower()}_avg:       {score:.4f}")
    for task, score in sorted(results.get("task_scores", {}).items()):
        print(f"  {task}: {score:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MTEB evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint directory")
    parser.add_argument("--tasks", nargs="+", default=None, help="MTEB tasks to evaluate")
    parser.add_argument("--output-dir", default="mteb_results", help="Output directory for results")
    args = parser.parse_args()

    # Import here to avoid circular deps
    from src.checkpoint import load_checkpoint
    from src.model import EmbeddingModel

    config, weights, metadata = load_checkpoint(args.checkpoint)
    # Model loading would go here — depends on how the model is constructed
    # This will be filled in when the base model preparation is done
    print(f"Would evaluate checkpoint: {args.checkpoint}")
    print(f"Tasks: {args.tasks or TARGET_TASKS}")
