#!/usr/bin/env python3
"""
Experiment wrapper. Handles ALL ops so the agent only does research.
*** READ-ONLY — DO NOT MODIFY THIS FILE ***

Usage: uv run scripts/experiment.py "description of experiment"
"""

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RESULT_JSON = ROOT / "result.json"
RUN_LOG = ROOT / "run.log"
RESULTS_JSONL = ROOT / "results.jsonl"
LEADERBOARD_JSONL = ROOT / "leaderboard.jsonl"
RESULTS_TSV = ROOT / "results.tsv"
LEADERBOARD_TSV = ROOT / "leaderboard.tsv"
LOGS_DIR = ROOT / "logs"
CONFIG_PATH = ROOT / "configs" / "training_stages.yaml"

TASK_KEY_MAP = {
    "STSBenchmark": "sts_bench",
    "SICK-R": "sick_r",
    "TwitterURLCorpus": "twitter_url",
    "SprintDuplicateQuestions": "sprint_dup",
    "TwentyNewsgroupsClustering": "20news",
    "RedditClustering": "reddit_clust",
    "SciFact": "scifact",
    "NFCorpus": "nfcorpus",
}

LEADERBOARD_TSV_COLS = [
    "commit", "primary", "sts_bench", "sick_r", "twitter_url",
    "sprint_dup", "20news", "reddit_clust", "scifact", "nfcorpus",
    "description",
]

RESULTS_TSV_COLS = [
    "commit", "primary_score", "memory_gb", "training_min",
    "datasets", "status", "description",
]

EXIT_CODES = {"keep": 0, "discard": 1, "contaminated": 2, "crash": 3}

# Keys parsed from the fallback --- block in run.log
FALLBACK_KEYS = {
    "primary_score", "sts_avg", "pair_class_avg", "cluster_avg",
    "retrieval_avg", "training_minutes", "peak_memory_gb", "num_params_M",
    "base_model", "training_stage", "total_train_pairs",
}

# Numeric keys that should be cast to float when parsed from fallback
FALLBACK_NUMERIC = {
    "primary_score", "sts_avg", "pair_class_avg", "cluster_avg",
    "retrieval_avg", "training_minutes", "peak_memory_gb", "num_params_M",
    "total_train_pairs",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(args: list[str], *, check: bool = True, ignore_errors: bool = False,
            cwd: Path = ROOT, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess command with sensible defaults."""
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                              check=check, **kwargs)
    except subprocess.CalledProcessError:
        if ignore_errors:
            return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr="")
        raise


def load_yaml(path: Path) -> dict:
    """Load a YAML file. Uses PyYAML."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Returns [] if missing/empty."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON record as one line."""
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def current_best_score(path: Path) -> float:
    """Return the highest primary_score from kept experiments, or 0."""
    records = read_jsonl(path)
    if not records:
        return 0.0
    kept = [r.get("primary_score", 0.0) for r in records if r.get("status") == "keep"]
    return max(kept, default=0.0)


def existing_tags() -> set[str]:
    """Return the set of existing git tags."""
    result = run_cmd(["git", "tag", "--list"], ignore_errors=True)
    return set(result.stdout.strip().splitlines())


def parse_fallback_log(log_path: Path) -> dict:
    """Parse the --- block from run.log as a fallback for result.json."""
    if not log_path.exists():
        return {}
    lines = log_path.read_text().splitlines()
    in_block = False
    parsed = {}
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            in_block = True
            continue
        if in_block:
            if not stripped:
                break
            m = re.match(r"^(\w+)\s*:\s*(.+)$", stripped)
            if m:
                key, val = m.group(1), m.group(2).strip()
                if key in FALLBACK_KEYS:
                    if key in FALLBACK_NUMERIC:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    parsed[key] = val
    return parsed


def regenerate_results_tsv() -> None:
    """Regenerate results.tsv from results.jsonl."""
    records = read_jsonl(RESULTS_JSONL)
    lines = ["\t".join(RESULTS_TSV_COLS)]
    for r in records:
        datasets_str = ";".join(r.get("datasets", []))
        row = [
            str(r.get("commit", "")),
            f"{r.get('primary_score', 0):.2f}" if isinstance(r.get("primary_score"), (int, float)) else str(r.get("primary_score", "")),
            f"{r.get('memory_gb', 0):.1f}" if isinstance(r.get("memory_gb"), (int, float)) else str(r.get("memory_gb", "")),
            f"{r.get('training_min', 0):.1f}" if isinstance(r.get("training_min"), (int, float)) else str(r.get("training_min", "")),
            datasets_str,
            str(r.get("status", "")),
            str(r.get("description", "")),
        ]
        lines.append("\t".join(row))
    RESULTS_TSV.write_text("\n".join(lines) + "\n")


def regenerate_leaderboard_tsv() -> None:
    """Regenerate leaderboard.tsv from leaderboard.jsonl."""
    records = read_jsonl(LEADERBOARD_JSONL)
    lines = ["\t".join(LEADERBOARD_TSV_COLS)]
    for r in records:
        task_scores = r.get("task_scores", {})
        row = [
            str(r.get("commit", "")),
            f"{r.get('primary_score', 0):.2f}" if isinstance(r.get("primary_score"), (int, float)) else str(r.get("primary_score", "")),
        ]
        # Task score columns in order
        for col in LEADERBOARD_TSV_COLS[2:-1]:  # skip commit, primary, description
            score = task_scores.get(col, "")
            if isinstance(score, (int, float)):
                row.append(f"{score:.2f}")
            else:
                row.append(str(score))
        row.append(str(r.get("description", "")))
        lines.append("\t".join(row))
    LEADERBOARD_TSV.write_text("\n".join(lines) + "\n")


def log_wandb(record: dict, description: str) -> None:
    """Log experiment results to Weights & Biases."""
    if record.get("wandb_run_id"):
        print("[wandb] Run already logged, skipping.")
        return
    try:
        import wandb  # type: ignore
        config = record.get("config", {})
        config["description"] = description
        config["commit"] = record.get("commit", "")
        run = wandb.init(
            project="autoresearch-embed",
            entity="gourmand-labs",
            name=description,
            config=config,
        )
        # Log numeric fields
        metrics: dict = {}
        for key in ("primary_score", "memory_gb", "training_min", "num_params_M",
                     "decontam_removed"):
            val = record.get(key)
            if isinstance(val, (int, float)):
                metrics[key] = val
        # Log per-task scores
        task_scores = record.get("task_scores", {})
        for task_name, score in task_scores.items():
            if isinstance(score, (int, float)):
                metrics[f"mteb/{task_name}"] = score
        if metrics:
            wandb.log(metrics)
        wandb.finish()
        print(f"[wandb] Logged run: {run.url}")
    except Exception as e:
        print(f"[wandb] Skipped: {e}")


def upload_decontaminated(record: dict) -> None:
    """Upload decontaminated parquet files to HuggingFace Hub."""
    decontam_removed = record.get("decontam_removed", 0)
    if not decontam_removed or decontam_removed <= 0:
        return
    try:
        from huggingface_hub import HfApi  # type: ignore
        api = HfApi()
        cache_dir = ROOT / "data_cache"
        if not cache_dir.exists():
            print("[hf] data_cache/ not found, skipping upload.")
            return
        uploaded = 0
        for pq_file in sorted(cache_dir.glob("clean_*.parquet")):
            dataset_name = pq_file.stem.removeprefix("clean_")
            repo_id = f"pierretokns/{dataset_name}-decontaminated-mteb"
            print(f"[hf] Uploading {pq_file.name} → {repo_id}")
            api.upload_file(
                path_or_fileobj=str(pq_file),
                path_in_repo=pq_file.name,
                repo_id=repo_id,
                repo_type="dataset",
            )
            uploaded += 1
        if uploaded:
            print(f"[hf] Uploaded {uploaded} decontaminated file(s).")
    except Exception as e:
        print(f"[hf] Upload skipped: {e}")


def print_summary(record: dict) -> None:
    """Print a structured summary box for the agent to read."""
    status = record.get("status", "unknown")
    primary = record.get("primary_score", 0)
    commit = record.get("commit", "???????")
    desc = record.get("description", "")
    training_min = record.get("training_min", 0)
    memory_gb = record.get("memory_gb", 0)
    num_params = record.get("num_params_M", "?")
    decontam = record.get("decontam_removed", 0)

    width = 60
    border = "=" * width
    print(f"\n{border}")
    print(f"  EXPERIMENT RESULT: {status.upper()}")
    print(f"{border}")
    print(f"  commit:         {commit}")
    print(f"  primary_score:  {primary:.2f}" if isinstance(primary, (int, float)) else f"  primary_score:  {primary}")
    print(f"  training_min:   {training_min:.1f}" if isinstance(training_min, (int, float)) else f"  training_min:   {training_min}")
    print(f"  memory_gb:      {memory_gb:.1f}" if isinstance(memory_gb, (int, float)) else f"  memory_gb:      {memory_gb}")
    print(f"  num_params_M:   {num_params}")
    print(f"  decontam_removed: {decontam}")
    print(f"  description:    {desc}")

    task_scores = record.get("task_scores", {})
    if task_scores:
        print(f"  {'---':^{width-4}}")
        print(f"  Task Scores:")
        for task, score in task_scores.items():
            if isinstance(score, (int, float)):
                print(f"    {task:30s} {score:.2f}")
            else:
                print(f"    {task:30s} {score}")

    datasets = record.get("datasets", [])
    if datasets:
        print(f"  datasets:       {'; '.join(datasets)}")

    print(f"{border}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: uv run scripts/experiment.py \"description of experiment\"", file=sys.stderr)
        sys.exit(1)

    description = sys.argv[1]
    extra_args: list[str] = []
    if "--quick-eval-only" in sys.argv:
        extra_args.append("--quick-eval-only")
    for i, arg in enumerate(sys.argv):
        if arg == "--resume-stage" and i + 1 < len(sys.argv):
            extra_args.extend(["--resume-stage", sys.argv[i + 1]])

    print(f"[experiment] Starting: {description}")
    os.chdir(ROOT)

    # ------------------------------------------------------------------
    # 1. Delete stale result.json
    # ------------------------------------------------------------------
    if RESULT_JSON.exists():
        RESULT_JSON.unlink()
        print("[experiment] Removed stale result.json")

    # ------------------------------------------------------------------
    # 2. Git commit source changes
    # ------------------------------------------------------------------
    run_cmd(["git", "add", "src/", "configs/"], ignore_errors=True)
    commit_result = run_cmd(
        ["git", "commit", "-m", f"experiment: {description}"],
        ignore_errors=True,
    )
    if commit_result.returncode == 0:
        print("[experiment] Committed source changes.")
    else:
        print("[experiment] Nothing new to commit (continuing).")

    # ------------------------------------------------------------------
    # 3. Get commit hash
    # ------------------------------------------------------------------
    hash_result = run_cmd(["git", "rev-parse", "--short", "HEAD"])
    commit_short = hash_result.stdout.strip()
    print(f"[experiment] Commit: {commit_short}")

    # ------------------------------------------------------------------
    # 4. Run training
    # ------------------------------------------------------------------
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    train_cmd = ["uv", "run", "src/train.py", "--config", str(CONFIG_PATH)] + extra_args
    train_env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXPERIMENT_DESC": description}

    crashed = False
    print(f"[experiment] Running: {' '.join(train_cmd)}")
    try:
        with open(RUN_LOG, "w") as log_file:
            proc = subprocess.Popen(
                train_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=train_env,
                cwd=ROOT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            # Use readline() instead of iterator to avoid Python's
            # internal read-ahead buffer (fixes log buffering delay)
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_file.write(line)
                    log_file.flush()
        if proc.returncode != 0:
            print(f"[experiment] Training exited with code {proc.returncode}")
            crashed = True
    except Exception as e:
        print(f"[experiment] Training crashed: {e}")
        crashed = True

    # ------------------------------------------------------------------
    # 5. Copy run.log to logs/
    # ------------------------------------------------------------------
    if RUN_LOG.exists():
        dest_log = LOGS_DIR / f"exp_{commit_short}.log"
        shutil.copy(RUN_LOG, dest_log)
        print(f"[experiment] Log saved: {dest_log}")

    # ------------------------------------------------------------------
    # 6. Parse results
    # ------------------------------------------------------------------
    result_data: dict = {}
    if RESULT_JSON.exists():
        try:
            result_data = json.loads(RESULT_JSON.read_text())
            print("[experiment] Parsed result.json")
        except json.JSONDecodeError:
            print("[experiment] result.json malformed, falling back to log parser.")
            result_data = parse_fallback_log(RUN_LOG)
    else:
        print("[experiment] No result.json, falling back to log parser.")
        result_data = parse_fallback_log(RUN_LOG)

    # ------------------------------------------------------------------
    # 7. Current best score
    # ------------------------------------------------------------------
    best_score = current_best_score(RESULTS_JSONL)
    primary_score = float(result_data.get("primary_score", 0.0))
    print(f"[experiment] primary_score={primary_score:.2f}, current_best={best_score:.2f}")

    # ------------------------------------------------------------------
    # 8. Determine status
    # ------------------------------------------------------------------
    contamination_failed = False
    # Check result_data for contamination flag
    if result_data.get("contamination_failed") or result_data.get("status") == "contaminated":
        contamination_failed = True
    # Also check run.log for contamination failure signals
    if RUN_LOG.exists():
        log_text = RUN_LOG.read_text()
        if "CONTAMINATION CHECK FAILED" in log_text or "contamination_failed: true" in log_text.lower():
            contamination_failed = True

    if crashed:
        status = "crash"
    elif contamination_failed:
        status = "contaminated"
    elif primary_score > best_score:
        status = "keep"
    else:
        status = "discard"

    print(f"[experiment] Status: {status}")

    # ------------------------------------------------------------------
    # 9. Build result record
    # ------------------------------------------------------------------
    # Resolve task_scores: map original task names to short keys
    raw_task_scores = result_data.get("task_scores", {})
    task_scores: dict[str, float] = {}
    for original_name, short_key in TASK_KEY_MAP.items():
        if original_name in raw_task_scores:
            task_scores[short_key] = float(raw_task_scores[original_name])
        elif short_key in raw_task_scores:
            task_scores[short_key] = float(raw_task_scores[short_key])

    # Load config
    config: dict = result_data.get("config", {})
    if not config and CONFIG_PATH.exists():
        try:
            config = load_yaml(CONFIG_PATH)
        except Exception:
            config = {}

    # Datasets
    datasets = result_data.get("datasets", [])
    if isinstance(datasets, str):
        datasets = [d.strip() for d in datasets.split(";") if d.strip()]

    record = {
        "commit": commit_short,
        "primary_score": primary_score,
        "memory_gb": float(result_data.get("memory_gb", result_data.get("peak_memory_gb", 0))),
        "training_min": float(result_data.get("training_min", result_data.get("training_minutes", 0))),
        "datasets": datasets,
        "status": status,
        "description": description,
        "task_scores": task_scores,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "num_params_M": result_data.get("num_params_M", None),
        "decontam_removed": int(result_data.get("decontam_removed", 0)),
    }

    # Preserve wandb_run_id if present in result.json
    if result_data.get("wandb_run_id"):
        record["wandb_run_id"] = result_data["wandb_run_id"]

    # ------------------------------------------------------------------
    # 10. Append to results.jsonl
    # ------------------------------------------------------------------
    append_jsonl(RESULTS_JSONL, record)
    print(f"[experiment] Appended to {RESULTS_JSONL.name}")

    # ------------------------------------------------------------------
    # 11. Leaderboard + milestone tags
    # ------------------------------------------------------------------
    if status == "keep":
        append_jsonl(LEADERBOARD_JSONL, record)
        print(f"[experiment] Appended to {LEADERBOARD_JSONL.name}")

        # Check for milestone tag (round-10 thresholds: 30, 40, 50, ...)
        tags = existing_tags()
        for milestone in range(10, 101, 10):
            tag_name = f"score-{milestone}"
            if primary_score >= milestone and tag_name not in tags:
                run_cmd(
                    ["git", "tag", tag_name, "-m", f"Primary score crossed {milestone}"],
                    ignore_errors=True,
                )
                print(f"[experiment] Created tag: {tag_name}")

    # ------------------------------------------------------------------
    # 12. Regenerate TSV files
    # ------------------------------------------------------------------
    regenerate_results_tsv()
    regenerate_leaderboard_tsv()
    print("[experiment] Regenerated TSV files.")

    # ------------------------------------------------------------------
    # 13. wandb logging
    # ------------------------------------------------------------------
    log_wandb(record, description)

    # ------------------------------------------------------------------
    # 14. HF upload of decontaminated data
    # ------------------------------------------------------------------
    upload_decontaminated(record)

    # ------------------------------------------------------------------
    # 15. Git commit results
    # ------------------------------------------------------------------
    run_cmd(
        ["git", "add", "results.jsonl", "leaderboard.jsonl",
         "results.tsv", "leaderboard.tsv", "logs/"],
        ignore_errors=True,
    )
    run_cmd(
        ["git", "commit", "-m", f"result: {status} {description}"],
        ignore_errors=True,
    )
    print("[experiment] Committed results.")

    # ------------------------------------------------------------------
    # 16. Post to agenthub
    # ------------------------------------------------------------------
    run_cmd(
        ["ah", "post", "embed-results",
         f"{status.upper()} | primary={primary_score:.2f} | {description}"],
        ignore_errors=True,
    )

    # ------------------------------------------------------------------
    # 17. Git push
    # ------------------------------------------------------------------
    run_cmd(["git", "push", "origin", "HEAD"], ignore_errors=True)

    # ------------------------------------------------------------------
    # 18. Print summary
    # ------------------------------------------------------------------
    print_summary(record)

    # ------------------------------------------------------------------
    # 19. Exit code
    # ------------------------------------------------------------------
    return EXIT_CODES.get(status, 3)


if __name__ == "__main__":
    sys.exit(main())
