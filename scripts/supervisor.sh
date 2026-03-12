#!/usr/bin/env bash
# Supervisor: restarts Claude Code sessions on crash or context exhaustion.
# Carries zero state — all state lives in git and JSONL files.
#
# Usage: bash scripts/supervisor.sh
# Can be run via launchctl or in tmux.
#
# *** READ-ONLY — DO NOT MODIFY THIS FILE ***

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MAX_RESTARTS=500
RESTART_DELAY=15
restart_count=0

# Prevent sleep while running
caffeinate -dims &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null" EXIT

# Ensure PATH includes required tools
export PATH="/Users/pierre/.local/bin:/opt/homebrew/bin:$PATH"

# Force unbuffered Python output so run.log streams in real-time
export PYTHONUNBUFFERED=1

echo "[supervisor] Project: $PROJECT_DIR"
echo "[supervisor] Logs: $LOG_DIR"
echo "[supervisor] Max restarts: $MAX_RESTARTS"
echo "[supervisor] caffeinate PID: $CAFFEINATE_PID"
echo "[supervisor] Claude: $(which claude)"

while [ $restart_count -lt $MAX_RESTARTS ]; do
    timestamp=$(date +%Y%m%d_%H%M%S)
    log_file="$LOG_DIR/session_${timestamp}.log"

    echo ""
    echo "[supervisor] =========================================="
    echo "[supervisor] Starting session $restart_count at $(date)"
    echo "[supervisor] Log: $log_file"
    echo "[supervisor] =========================================="

    # Run Claude Code in non-interactive mode
    # --print: output only (no TUI)
    # --max-turns: prevent context exhaustion crash (clean exit instead)
    # --model: use sonnet for experiment planning (cheaper, fast)
    cd "$PROJECT_DIR" && claude \
        --print \
        --model claude-sonnet-4-6 \
        --dangerously-skip-permissions \
        --max-turns 200 \
        "Resume the experiment loop. Read results.jsonl and git log --oneline -20. If uncommitted changes exist, git checkout . to clean up. Decide what to try, edit code, run uv run scripts/experiment.py. CRITICAL: experiment.py is synchronous — just call it and wait for it to return. NEVER create while/sleep/pgrep polling loops or background monitoring scripts. They deadlock. Never stop." \
        2>&1 | tee "$log_file" || true

    restart_count=$((restart_count + 1))
    echo "[supervisor] Session ended at $(date). Restart $restart_count/$MAX_RESTARTS"

    # Brief pause before restart (lets transient issues resolve)
    sleep $RESTART_DELAY
done

echo "[supervisor] Max restarts reached ($MAX_RESTARTS). Exiting."
