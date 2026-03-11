# autoresearch-embed

Read `program.md` in the project root for the complete experiment protocol. Follow it exactly.

Key rules:
- Files in src/eval/ are READ-ONLY. Never modify them.
- All MTEB evaluation must use the official mteb library via src/eval/mteb_runner.py.
- Never use git add -A. Always stage specific files.
- Never stop to ask the human if you should continue. Run the loop until interrupted.
- On session start, read results.tsv and git log to resume from where you left off.
