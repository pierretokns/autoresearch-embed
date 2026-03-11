"""
Fast proxy MTEB evaluation for rapid iteration.
Runs only STSBenchmark, SICK-R, TwitterURLCorpus (~2 minutes).

*** READ-ONLY — DO NOT MODIFY THIS FILE ***

If quick eval shows regression, skip full eval and discard immediately.
"""

from __future__ import annotations

from src.eval.mteb_runner import evaluate, print_results, CATEGORY_WEIGHTS

QUICK_TASKS = [
    "STSBenchmark",
    "SICK-R",
    "TwitterURLCorpus",
]


def quick_evaluate(model, tokenizer, output_dir: str = "mteb_results_quick") -> dict:
    """
    Run quick MTEB evaluation on a small subset of tasks.
    Returns results in the same format as full evaluation.
    """
    return evaluate(model, tokenizer, tasks=QUICK_TASKS, output_dir=output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick MTEB evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="mteb_results_quick")
    args = parser.parse_args()

    from src.checkpoint import load_checkpoint

    config, weights, metadata = load_checkpoint(args.checkpoint)
    print(f"Would quick-evaluate checkpoint: {args.checkpoint}")
    print(f"Tasks: {QUICK_TASKS}")
