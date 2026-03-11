# autoresearch-embed

Autonomous embedding model research on Apple Silicon (MLX). Train, evaluate, and iterate
on text embedding models using contrastive learning, with MTEB as the ground-truth benchmark.

**Monorepo note:** This project may be a submodule. Always stage only `autoresearch-embed/` paths. Never use blind `git add -A`.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch
   `embed/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b embed/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context.
   - `program.md` — this file, the protocol.
   - `configs/` — model and training stage configurations.
   - `src/train.py` — training loop (you modify this).
   - `src/model.py` — encoder architecture (you modify this).
   - `src/losses.py` — loss functions (you modify this).
   - `src/data/curator.py` — dataset discovery (you modify this).
   - `src/data/loader.py` — data loading (you modify this).
   - `src/eval/mteb_runner.py` — MTEB evaluation (READ-ONLY, never modify).
   - `src/eval/quick_eval.py` — fast proxy eval (READ-ONLY, never modify).
   - `src/eval/contamination.py` — contamination checker (READ-ONLY, never modify).
4. **Verify base model exists**: Check that `checkpoints/base/` contains the base encoder
   weights. If not, tell the human to run `uv run scripts/prepare_base.py`.
5. **Verify agenthub connectivity**: Run `ah channels` to confirm connectivity. If not
   configured, tell the human to run `uv run scripts/setup_hub.py`.
6. **Initialize results.tsv**: Ensure header row exists. Run the baseline experiment.
7. **Confirm and go**: Show the human the setup summary, get confirmation, then begin
   the autonomous experiment loop.

## Rules

**What you CAN modify (experiment freely):**
- `src/train.py` — training loop, hyperparameters, stages, schedule.
- `src/model.py` — encoder architecture, pooling strategy, projection head.
- `src/losses.py` — loss functions, temperature, negative weighting.
- `src/data/curator.py` — dataset discovery and filtering logic.
- `src/data/synthetic.py` — synthetic data generation prompts and logic.
- `src/data/loader.py` — batching, sampling, augmentation.
- `src/data/hard_negatives.py` — mining strategy, top-k, filtering.
- `src/data/registry.py` — dataset manifest.
- `src/checkpoint.py` — checkpoint format, metadata.
- `src/hub.py` — agenthub posting format.
- `configs/*.yaml` — all configuration files.

**What you CANNOT modify (integrity boundary):**
- `src/eval/mteb_runner.py` — READ-ONLY. Official MTEB evaluation.
- `src/eval/quick_eval.py` — READ-ONLY. Fast proxy evaluation.
- `src/eval/contamination.py` — READ-ONLY. Contamination detection.
- `scripts/supervisor.sh` — READ-ONLY. Restart mechanism.

**Anti-cheat rules (violations invalidate all results):**
- All MTEB evaluation MUST go through `src/eval/mteb_runner.py` using the official `mteb`
  library. No custom eval loops. No reimplemented metrics.
- Before every full MTEB run, `src/eval/contamination.py` MUST pass with all per-task
  contamination rates below 1%.
- Training code (`src/data/`, `src/train.py`) must NEVER import from `src/eval/`.
- Eval code (`src/eval/`) must NEVER import from `src/data/`.
- MTEB test data must NEVER appear in training. Do not load MTEB datasets for training.
- Document every data source in `src/data/registry.py` with URL, date, count, and hash.
- Never peek at test labels or test data during training or data curation.

## Metrics

**Primary score** (the single number for keep/discard decisions):
```
primary = 0.3 * mean(STS) + 0.2 * mean(PairClassification) +
          0.2 * mean(Clustering) + 0.3 * mean(Retrieval)
```

**Target MTEB tasks:**
- STS: STSBenchmark, SICK-R
- Pair Classification: TwitterURLCorpus, SprintDuplicateQuestions
- Clustering: TwentyNewsgroupsClustering, RedditClustering
- Retrieval: SciFact, NFCorpus

**Quick eval** (for rapid iteration): STSBenchmark, SICK-R, TwitterURLCorpus. ~2 minutes.
**Full eval** (for kept experiments): All 8 tasks. ~10 minutes.

## Output format

After training, the script must print these exact lines (parsed by the experiment loop):

```
---
primary_score:     <float>
sts_avg:           <float>
pair_class_avg:    <float>
cluster_avg:       <float>
retrieval_avg:     <float>
training_minutes:  <float>
peak_memory_gb:    <float>
num_params_M:      <float>
base_model:        <string>
training_stage:    <string>
total_train_pairs: <string>
```

## Logging

**results.tsv** (tab-separated, one row per experiment):
```
commit	primary_score	memory_gb	status	description
```
Status is one of: `keep`, `discard`, `crash`, `contaminated`.

**leaderboard.tsv** (tab-separated, one row per kept experiment with full MTEB):
```
commit	primary	sts_bench	sick_r	twitter_url	sprint_dup	20news	reddit_clust	scifact	nfcorpus	description
```

## The experiment loop

LOOP FOREVER:

1. **Read state.** `results.tsv`, `leaderboard.tsv`, `git log --oneline -20`.
   Know what you tried, what worked, what the current best is.

2. **Decide what to try.** Pick ONE change per experiment. Priority order:
   a. No baseline yet → run default config, establish baseline.
   b. Data: add a new dataset, change sampling ratios, add synthetic data.
   c. Loss: change InfoNCE temperature, hard negative weight, try margin loss.
   d. Schedule: change stage durations, learning rates, warmup/cooldown ratios.
   e. Architecture: change pooling (CLS vs mean vs weighted), projection dim,
      which encoder layers to use, whether to freeze early layers.
   f. Hard negatives: change mining top-k, mining frequency, filtering threshold.
   g. Synthetic data: generate for domains where scores are weakest.
   h. Base model: try Qwen3-0.6B instead of ModernBERT (or vice versa).

3. **Edit code.** Modify the relevant files in `src/` or `configs/`.

4. **Commit.** `git add src/ configs/ && git commit -m "experiment: <description>"`
   NEVER use `git add -A` or `git add .`. Always specify paths.

5. **Train.** `PYTHONUNBUFFERED=1 uv run src/train.py --config configs/training_stages.yaml > run.log 2>&1`
   Redirect everything — do NOT let output flood your context.
   IMPORTANT: Always set PYTHONUNBUFFERED=1 so run.log streams in real-time for monitoring.

6. **Handle crashes.** If training crashes:
   - Read `tail -50 run.log` to diagnose.
   - If OOM: reduce batch size, retry.
   - If code bug: fix the bug, amend the commit, retry.
   - If data issue: log, remove bad data source, retry.
   - Max 3 retries. After 3 failures, log as crash in results.tsv and move on.

7. **Quick eval.** `uv run src/eval/quick_eval.py --checkpoint checkpoints/latest/ > eval.log 2>&1`

8. **Decision gate.**

   IF primary_score > current_best:
     a. Run contamination check:
        `uv run src/eval/contamination.py --training-manifest data_cache/manifest.json`
     b. IF contamination PASSES:
        - Run full MTEB: `uv run src/eval/mteb_runner.py --checkpoint checkpoints/latest/`
        - Append to results.tsv (status=keep) and leaderboard.tsv
        - `git add results.tsv leaderboard.tsv && git commit --amend --no-edit`
        - `ah post embed-results "KEEP exp-N | <details>"`
        - `ah push`
     c. IF contamination FAILS:
        - Append to results.tsv (status=contaminated)
        - `git reset --hard HEAD~1`
        - `ah post embed-results "CONTAMINATED exp-N | <details>"`

   IF primary_score <= current_best:
     - Record in results.tsv (status=discard), noting the commit hash
     - `git reset --hard <last kept commit>` to discard the experiment

9. **Periodic maintenance.**
   - Every ~5 experiments: re-mine hard negatives with current best checkpoint.
   - Every ~10 experiments: re-search HF Hub for newly published datasets.
   - When a specific MTEB category is lagging: target synthetic data for that domain.

10. **Post reasoning.** Every few experiments, post your current thinking to
    `#embed-ideas`. What hypotheses are you testing? What patterns have you noticed?

## Training stages explained

Each experiment runs a multi-stage pipeline. Default durations are in
`configs/training_stages.yaml` but you can change them.

**Stage 1: Warmup** (default 10 min)
Easy positive pairs only (NLI entailment). High LR. No hard negatives.
Purpose: adapt projection head without destroying pretrained knowledge.

**Stage 2: Full contrastive** (default 20 min)
All datasets mixed. In-batch negatives. Lower LR with cosine decay.
Purpose: main training, learn similarity function across diverse data.

**Stage 3: Hard negative mining** (default 5 min, inference only)
Embed training corpus with current model. Find top-k nearest non-positives.
Save hard negative assignments for stage 4.

**Stage 4: Hard negative fine-tuning** (default 20 min)
Training data augmented with mined hard negatives. Very low LR.
Purpose: refine embedding space, push apart confusing near-misses.

Total: ~55 minutes training + ~2 min quick eval + ~10 min full eval = ~67 min per experiment.
Expect 20-25 complete experiments per 24 hours.

## Memory budget

M2 Ultra: 64GB unified memory.
Training + model + data must stay under 55GB peak.
If peak exceeds 55GB, reduce batch size by half and retry.
Always log peak memory in results.tsv.

## Crash handling

| Failure | Action |
|---|---|
| OOM | Halve batch size, retry |
| Python exception | Read traceback, fix code, retry (up to 3x) |
| Data loading error | Remove bad dataset from mix, retry |
| MTEB eval timeout | Reduce task list, retry |
| agenthub unreachable | Log locally, continue without posting |
| Persistent failure | Log as crash in results.tsv, skip to next experiment |

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human if you should continue.
Do NOT ask "should I keep going?" or "would you like me to try something else?".
The human expects you to run indefinitely until manually stopped.

If you run out of obvious ideas:
- Re-read the pplx-embed insights: multi-stage contrastive, LLM-labeled data,
  iterative hard negative mining, diffusion pretraining for bidirectional attention.
- Try combining two changes that each nearly won individually.
- Try more radical architecture changes (different base model, novel pooling).
- Try domain-specific optimization (train separate heads for different task types).
- Read the MTEB leaderboard on HuggingFace for ideas about what top models do.
- Generate more synthetic training data for your weakest category.
- Search HF Hub and Kaggle for new datasets (`hf datasets`, `kaggle datasets list`).

There is always something to try. The loop runs until the human interrupts.

## Resumption protocol

If you are starting a new session (the supervisor restarted you after context exhaustion
or a crash):

1. `git log --oneline -20` — see recent history.
2. Read `results.tsv` — see experiment count, last status, current best score.
3. Read `leaderboard.tsv` — see per-task scores for all kept checkpoints.
4. `git status` — check for uncommitted changes.
5. If uncommitted changes exist, they are from a crashed experiment. Clean up:
   `git checkout .`
6. Verify the branch name starts with `embed/`. If on master, something is wrong;
   read git log and check out the correct branch.
7. Resume the experiment loop from step 1 (read state, decide what to try).

Do NOT re-read program.md on every resumption (you already know it). Do NOT re-run
setup steps. Jump straight into the loop.

## Paper-ready outputs

Maintain results that can go directly into a publication:

**results.tsv** is the ablation table. Every experiment tried, with primary score,
memory usage, keep/discard status, and one-line description. This shows the full
search trajectory.

**leaderboard.tsv** is the main results table. Per-task MTEB scores for every kept
checkpoint. This shows the progression of quality across tasks.

When posting to agenthub, format scores as Markdown tables. Use consistent decimal
precision (2 places for MTEB scores, 1 place for memory and time).

## Key research references

- **pplx-embed** (Perplexity): Diffusion-continued pretraining on Qwen3 → multi-stage contrastive. Paper: arXiv 2602.11151
- **jxmo blog**: LLM-labeled data to avoid false negative poisoning. Exact softmax via coordinate ascent. https://blog.jxmo.io/p/how-to-train-the-best-embedding-model
- **Arctic-Embed** (Snowflake): Source stratification, 256-token docs, hard negative mining with tunable threshold
- **EmbeddingGemma**: Best sub-500M on MTEB multilingual, Matryoshka representations

## Commit hygiene

- NEVER use `git add -A` or `git add .`. Always specify: `git add src/ configs/` or
  `git add results.tsv leaderboard.tsv`.
- Commit messages follow the format: `experiment: <desc>` or `result: keep/discard <desc>`.
- Push to GitHub remote after every kept experiment: `git push origin HEAD`
- Keep commits atomic: one experiment per commit, not multiple changes bundled.
- After a discard, reset cleanly. No orphan commits.
