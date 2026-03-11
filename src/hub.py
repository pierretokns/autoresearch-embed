"""
Agenthub integration wrapper around the `ah` CLI.
Posts findings, pushes commits, reads channels.

Failure-tolerant: if agenthub is unreachable, log locally and continue.
"""

import json
import subprocess
from pathlib import Path

LOG_FILE = Path(__file__).parent.parent / "logs" / "hub_offline.log"


def _run_ah(args: list[str], timeout: int = 30) -> tuple[bool, str]:
    """Run an ah CLI command. Returns (success, output)."""
    try:
        result = subprocess.run(
            ["ah"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, str(e)


def _log_offline(action: str, data: str):
    """Log a failed agenthub action locally."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    import time
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {action} | {data}\n")


def post_finding(channel: str, message: str):
    """Post a message to an agenthub channel."""
    ok, out = _run_ah(["post", channel, message])
    if not ok:
        _log_offline("post", f"{channel}: {message}")


def push_commit():
    """Push the current HEAD to the agenthub git DAG."""
    ok, out = _run_ah(["push"], timeout=60)
    if not ok:
        _log_offline("push", "failed to push HEAD")


def read_channel(channel: str, limit: int = 10) -> list[str]:
    """Read recent posts from a channel."""
    ok, out = _run_ah(["board", "read", channel, "--limit", str(limit)])
    if ok and out:
        return out.split("\n")
    return []


def post_result(exp_num: int, status: str, description: str, scores: dict | None = None):
    """Post a structured experiment result."""
    score_str = ""
    if scores:
        score_str = " | ".join(f"{k}={v:.2f}" for k, v in scores.items())
        score_str = f" | {score_str}"
    msg = f"{status.upper()} exp-{exp_num:03d}{score_str} | {description}"
    post_finding("embed-results", msg)


def post_leaderboard(scores: dict, exp_num: int):
    """Post current leaderboard to agenthub."""
    lines = [f"Best after exp-{exp_num:03d}:"]
    for task, score in scores.items():
        lines.append(f"  {task}: {score:.2f}")
    post_finding("embed-leaderboard", "\n".join(lines))
