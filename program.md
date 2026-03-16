# autoresearch-embed

Autonomous embedding research. ModernBERT-base (149M) → competitive MTEB scores via contrastive learning on MLX.

**Current best: 57.51** (exp-67). Target: 65+ (gte-modernbert-base level). Direct competitor uses same backbone.

## Workflow

1. Read state: `git log --oneline -10` and `tail -5 results.jsonl | python3 -c "import sys,json;[print(json.loads(l)['primary_score'],json.loads(l)['status'],json.loads(l)['description'][:70]) for l in sys.stdin]"`
2. Decide what to try (see priorities below)
3. Edit code in `src/` or `configs/`
4. Pre-flight check (see below)
5. Run: `uv run scripts/experiment.py "description"`
6. Analyze output. GOTO 1.

**Quick eval** (`--quick-eval-only`): 3 tasks, ~2 min. Use for exploration.
**Full eval**: 8 tasks, ~10 min. Use for validation of promising results only.

## What matters (in order)

Data > training methodology > architecture > hyperparameters. Stop tuning hyperparameters. Focus on the first two.

### 1. Data (highest ROI)

**Scaling.** We have ~200K pairs capped at 30K/dataset. Run a scaling experiment: 200K vs 400K vs 800K pairs. Find where the curve flattens before spending time on anything else. Increase `data.max_rows_per_dataset` in config.

**Deduplication.** No cross-dataset dedup exists. Near-duplicate pairs waste compute and bias the loss. Add MinHash dedup over the training data (separate from eval contamination).

**False negative filtering.** With batch=16 (effective 64 via grad_accum), each query has 63 in-batch "negatives" — some are semantically similar (false negatives). This poisons the contrastive signal. Fix: compute pairwise similarity within each batch, mask pairs above a threshold (e.g., 0.7). This is why hard_neg_weight=2.0 destroyed the model (exp-70: 37.77) — the hard neg formulation amplifies false negatives.

**Weak category data.** Our weakest tasks are retrieval (SciFact ~44, NFCorpus ~19) and clustering (~35-40). Generate targeted synthetic data: question-passage pairs for scientific/biomedical domains, topic-labeled documents for clustering. Use Claude to generate high-quality pairs.

### 2. Training methodology

**Layer-wise LR decay (LLRD).** Lower layers need less adaptation (good features from pretrain), upper layers need more. Apply LR × decay^(num_layers - layer_idx). Standard for BERT fine-tuning. Used by NV-Embed, GTE, Arctic-Embed. Expect +1-2 points.

**Multi-task loss.** Replace sequential stages with a combined loss: `α·InfoNCE + β·STS_regression + γ·clustering_loss`. Sequential stages cause catastrophic forgetting — contrastive training forgets warmup, fine-tuning forgets contrastive. Every top MTEB model uses multi-task.

**Online hard negative mining.** Current pipeline mines hard negatives once (stage 3) then fine-tunes on stale negatives (stage 4). Mine negatives every N steps during contrastive training instead. This is what GTE and Arctic-Embed do.

**EMA.** Maintain exponential moving average of weights. Evaluate on EMA weights. Free +0.5-1.0 points, standard practice.

**Data curriculum.** Easy pairs first (high similarity, clear matches), hard pairs later. The contrastive equivalent of curriculum learning.

### 3. Architecture (only after 1 & 2 are solid)

**Latent attention pooling.** Now fixed (was broken for exp-44-46). Re-test with current best recipe. NV-Embed-v2 (#1 MTEB) uses this. Start with K=4 latent queries.

**Projection head (SimCLR pattern).** Train with a 2-layer MLP projection head on top of embeddings. Loss operates on projected space. Evaluate on raw encoder output (remove head). The projection absorbs task noise, keeping encoder representations general.

**Matryoshka (MRL).** Train embeddings at multiple dims simultaneously. Low priority until base quality is solid.

### 4. Bug fix re-runs (do these first)

All config fields now work. See issue #2 for details.

1. **max_seq_length=512** — was hardcoded 256 for all 70 experiments. Retrieval tasks most affected. Just run current best config — config already says 512.
2. **Latent attention pooling** — was broken (zero-init killed gradients). Fixed. Re-test.
3. **Hard neg weight** — was silently disabled for exp-33→69. The `>1.0` condition is fixed to `>0.0`. But exp-70 showed weight=2.0 destroys the model (37.77), so investigate *why* before sweeping weights. The loss formulation itself may be wrong.

## Pre-Flight Checklist

**MANDATORY before every experiment.** Skipping this wastes hours on broken code.

- **Config field changed?** `grep -r "field_name" src/` — if no matches, config is ignored.
- **New module/layer?** Print `model.trainable_parameters()` after init. If your params aren't listed, they're frozen.
- **Loss changed?** Print loss for first 3 steps. Non-zero and decreasing? If stuck at 0, the term has no gradient.
- **Data changed?** Print sample count and source distribution at training start.
- **5+ consecutive failures?** Stop ablating. Combine near-winners, or investigate if a bug is masking changes.

## Config & Infrastructure

**Framework:** MLX native (not PyTorch). Key patterns:
- `nn.value_and_grad()` + `optim.AdamW` for training
- `mx.checkpoint(layer)` for gradient checkpointing (enabled by default, ~80% memory savings)
- `mx.fast.scaled_dot_product_attention()` for attention
- Tokenizer returns `return_tensors="np"`, convert to `mx.array()`

**Memory:** M2 Ultra 64GB. With gradient checkpointing: ~4GB peak at B=16, T=512. Config: batch_size=16, grad_accum=4 (effective batch 64).

**Modifiable:** `src/train.py`, `src/model.py`, `src/losses.py`, `src/data/*.py`, `configs/*.yaml`
**Read-only:** `src/eval/`, `scripts/`
**Never:** run git directly (experiment.py handles it), create polling loops (deadlocks), import eval from training code

## Metrics

```
primary = 0.3·mean(STS) + 0.2·mean(PairClassification) + 0.2·mean(Clustering) + 0.3·mean(Retrieval)
```

8 tasks: STSBenchmark, SICK-R, TwitterURLCorpus, SprintDuplicateQuestions, TwentyNewsgroupsClustering, RedditClustering, SciFact, NFCorpus.

**Current weaknesses:** Retrieval (SciFact ~44, NFCorpus ~19) and Clustering (~35-40). These have 50% weight in the primary score. Fixing them is the fastest path to 65+.

## SOTA Context

| Model | Params | MTEB Avg | Key technique |
|-------|--------|----------|---------------|
| NV-Embed-v2 | 7B | 72.31 | Latent attention pooling |
| gte-modernbert-base | 149M | ~65 | Same backbone, our target |
| **Our best** | **149M** | **57.51** | CLS pooling, 60-min contrastive |

Gap to close: 7.5 points. That's data + training methodology, not architecture.

## Anti-Cheat

- All eval via official `mteb` library. No custom scoring.
- Contamination check must pass (<1% per task).
- Training code never imports from eval.

## Resumption

On new session: `git log --oneline -10` and `tail -3 results.jsonl`. If uncommitted changes: `git checkout .`. Go straight to experiments.

## NEVER STOP

Run experiments until interrupted. If stuck: focus on data scaling and training methodology, not hyperparameter tweaks. There is always something to try.
