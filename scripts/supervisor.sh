#!/usr/bin/env bash
# Supervisor: restarts Claude Code sessions on crash or context exhaustion.
# Carries zero state — all state lives in git and TSV files.
#
# Usage: bash scripts/supervisor.sh
# Run in tmux and walk away.
#
# *** READ-ONLY — DO NOT MODIFY THIS FILE ***

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MAX_RESTARTS=100
RESTART_DELAY=10
restart_count=0

echo "[supervisor] Project: $PROJECT_DIR"
echo "[supervisor] Logs: $LOG_DIR"
echo "[supervisor] Max restarts: $MAX_RESTARTS"

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
    # --allowedTools: permit autonomous operation
    cd "$PROJECT_DIR" && claude --print \
        --prompt "Read program.md and resume the experiment loop. Check results.tsv and git log --oneline -20 for current state. If uncommitted changes exist, git checkout . to clean up. Then continue experimenting." \
        --allowedTools "Bash,Read,Write,Edit,Glob,Grep" \
        --max-turns 200 \
        2>&1 | tee "$log_file" || true

    restart_count=$((restart_count + 1))
    echo "[supervisor] Session ended at $(date). Restart $restart_count/$MAX_RESTARTS"

    # Brief pause before restart (lets transient issues resolve)
    sleep $RESTART_DELAY
done

echo "[supervisor] Max restarts reached ($MAX_RESTARTS). Exiting."
