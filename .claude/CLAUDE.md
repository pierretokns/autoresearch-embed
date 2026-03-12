# autoresearch-embed

Read `program.md` for the experiment protocol. Your job is to RESEARCH, not operate.

Key rules:
- Files in `src/eval/` and `scripts/` are READ-ONLY. Never modify them.
- After editing code, run: `uv run scripts/experiment.py "description"`
- The script handles: git, training, eval, contamination, results, wandb, push, agenthub, logs.
- You handle: reading state, forming hypotheses, editing code, calling the script.
- Never run train.py directly. Never run git add/commit/push directly.
- **Never create background polling loops, monitoring loops, or `while true; sleep` constructs.** `experiment.py` runs synchronously and returns when done — just call it and read the output. Polling loops have caused deadlocks (pgrep self-match + tail window drift).
- Never stop to ask the human. Run the loop until interrupted.
- On session start, read `results.jsonl` and `git log --oneline -20` to resume.
- Framework is MLX (native Apple Silicon). No PyTorch in training code.
