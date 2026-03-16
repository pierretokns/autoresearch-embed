# autoresearch-embed

Autonomous embedding model research on Apple Silicon. Train, evaluate, and iterate on text embedding models using contrastive learning, with MTEB as the ground-truth benchmark.

**Goal**: Build an embedding model that achieves SOTA or near-SOTA results for its parameter class, or discovers a novel technique worth publishing. Our base (ModernBERT-base, 149M params) is already SOTA for encoder-only NLU (GLUE 88.4) and code retrieval (CoIR 79.31) — we need to unlock that potential for general embeddings.

## Your Workflow

1. **Read state**: `cat results.jsonl | python3 -c "import sys,json; [print(json.loads(l)['status'], json.loads(l)['primary_score'], json.loads(l)['description']) for l in sys.stdin]"`
   Or: `git log --oneline -20`
2. **Decide what to try** (see Research Directions below)
3. **Edit code** in `src/` or `configs/`
4. **Pre-flight check** (MANDATORY before every experiment — see Pre-Flight Checklist below)
5. **Run**: `uv run scripts/experiment.py "description of what you changed"`
6. **Read the output summary**. Think about what worked and what to try next.
7. GOTO 1

The script handles git, training, eval, contamination checks, results logging, wandb, agenthub posting, and git push. You just research.

## Pre-Flight Checklist

**Run these checks after editing code and BEFORE running experiment.py.** Skipping this wastes 30-90 minutes on broken experiments. Past bugs found: frozen pooling params, ignored config fields, hardcoded values shadowing config.

**If you added or changed a config field:**
- Grep for the field name in `src/`. Is it actually read? If `grep -r "field_name" src/` returns nothing, your config change will be silently ignored.
- Print the actual value at the start of the stage: `print(f"[config] field_name={value}")`. Confirm it matches what you set.

**If you added a new module or layer (pooling, projection, etc.):**
- After model init, verify it appears in trainable parameters: `print([(k, v.shape) for k, v in model.trainable_parameters()])`. If your new module's params aren't listed, they won't receive gradients — the module is frozen.
- In MLX, `mx.zeros(...)` creates a constant, NOT a learnable parameter. Use proper parameter registration.

**If you changed a loss function or added a loss term:**
- Print the loss value for the first 2-3 steps. Confirm it's non-zero and changing. A loss stuck at 0.0 means the term isn't contributing gradients.
- If you added a new loss term, confirm it's actually added to the total loss (not just computed and discarded).

**If you changed data loading or dataset selection:**
- Print the number of loaded samples and their source distribution at the start of training. Confirm the new data is actually present.

**After 5+ consecutive failures to beat the champion:**
- Before trying another single-variable ablation, combine the top 2-3 near-winners.
- Re-read this checklist and verify no silent bugs are masking your changes.

## Experiment Time Budget

**Target: 5 minutes per experiment** (Karpathy autoresearch principle). Fast iteration beats long training. Use short stage durations (2+3+2 min) for exploration, longer runs (15+30+30 min) only for final validation of promising configs. The agent that runs 50 short experiments learns more than the one that runs 3 long ones.

Tune these to hit the 5-min target:
- Max seq length: 256 for exploration, 512 for validation runs
- Batch size: whatever fits in memory (64-128 for MLX on M2 Ultra)
- Stage durations: warmup 2min, contrastive 3min, finetune 2min (exploration mode)

## What You Can Modify

- `src/train.py`, `src/model.py`, `src/losses.py`, `src/data/*.py`, `configs/*.yaml`
- Everything about training: architecture, optimizer, hyperparameters, stages, data loading, loss functions

## What You Cannot Modify

- `src/eval/` — READ-ONLY (integrity boundary)
- `scripts/` — READ-ONLY (automation)
- Never run `git add`/`commit`/`push` directly — `experiment.py` handles it
- Never run `train.py` directly — always use `experiment.py`
- **Never create polling loops** (`while true; sleep`, background monitors, `pgrep` watchers). `experiment.py` is synchronous — call it, wait for it to finish, read stdout. Polling loops cause deadlocks.
- Never import from `src/eval/` in training code (or vice versa)

## Framework: MLX (Native Apple Silicon)

Training uses MLX natively — no PyTorch, no MPS. Key patterns:
- Model is `mlx.nn.Module` in `src/model.py` (ModernBERT encoder + projection)
- Training uses `nn.value_and_grad()` + `mlx.optimizers.AdamW`
- Tokenizer returns `return_tensors="np"`, convert to `mx.array()` for forward pass
- `encode_sentences()` returns numpy for MTEB compatibility
- No `.to(device)`, no `autocast`, no `empty_cache` — MLX handles memory automatically
- Decontaminated data cached in `data_cache/` keyed by dataset config hash
- Use `mx.fast.scaled_dot_product_attention()` for efficient attention

## Metrics

**Primary score** (the single number for keep/discard decisions):
```
primary = 0.3 * mean(STS) + 0.2 * mean(PairClassification) + 0.2 * mean(Clustering) + 0.3 * mean(Retrieval)
```

**Target MTEB tasks (8):**
- STS: STSBenchmark, SICK-R
- Pair Classification: TwitterURLCorpus, SprintDuplicateQuestions
- Clustering: TwentyNewsgroupsClustering, RedditClustering
- Retrieval: SciFact, NFCorpus

**Quick eval** (use for 5-min experiments): STSBenchmark, SICK-R, TwitterURLCorpus (~2 min).
**Full eval** (use for validation): all 8 (~10 min).

## Research Directions (Priority Order)

Pick ONE change per experiment. These are ordered by expected impact for our setup:

### Tier 1: High-Impact Architecture Changes

a. **Advanced Pooling** (biggest potential win):
   - **Latent Attention Pooling**: Replace mean pooling with a cross-attention layer over a trainable dictionary. A small number of learned query vectors attend over all token representations. This captures richer structure than mean/CLS pooling. See NV-Embed-v2 (MTEB #1, score 72.31).
   - **Multi-Layer Trainable Pooling**: Use weighted combination of hidden states from ALL encoder layers, not just the final one. Each layer captures different granularity — early layers have syntax, late layers have semantics. Learn the weights. Research shows middle layers of encoders often capture MORE semantic information than the last layer (0.08 gap on STS for Mistral-7B). Use a trainable cross-attention network over layer hidden states.
   - **Weighted Mean Pooling**: Learn per-token or per-layer importance weights instead of uniform averaging.

b. **Matryoshka Representation Learning (MRL)**:
   - Train embeddings that work at multiple dimensionalities simultaneously (768, 512, 256, 128, 64).
   - Apply the contrastive loss at each truncated dimension during training.
   - Enables efficient retrieval at lower dims without quality collapse. See EmbeddingGemma.
   - Advanced: implement SMEC for progressive dimension reduction with minimal information loss.
   - Scaling law insight: it's more effective to use a larger model with smaller embedding dim than a smaller model with larger dim. Under fixed compute, increase both model size and dim proportionally.
   - Combined with int8 quantization: 768-dim float32 (3KB) → 256-dim int8 (256 bytes) = 12x storage reduction.

c. **Attention Strategy & Efficiency**:
   - ModernBERT already has bidirectional attention (crucial for STS and retrieval — outperforms causal attention on these tasks). This is our advantage over decoder-based models that need special adaptation for bidirectional.
   - However, causal attention can be superior for clustering and classification. Consider a hybrid: bidirectional for retrieval/STS heads, causal for classification/clustering.
   - Experiment with the local/global attention ratio in ModernBERT.
   - Implement unpadding (remove padding tokens from attention computation) for throughput.
   - Use `mx.fast.scaled_dot_product_attention()` for efficient attention.

### Tier 2: Training Methodology

d. **Teacher-Guided Hard Negative Mining**:
   - Use a larger model (or the current best checkpoint) as a teacher/reranker to score negatives.
   - Apply a margin to filter out "false negatives" — pairs labeled as negative but actually semantically similar. This is the #1 cause of training signal degradation.
   - See Arctic-Embed methodology: tunable similarity threshold for negative filtering.

e. **Distillation from Rerankers (zELO approach)**:
   - Instead of binary relevant/not-relevant labels, distill continuous Elo scores (0 to 1) from pairwise document "battles" scored by a reranker.
   - Train with soft labels using KL divergence or MSE loss alongside contrastive loss.
   - zembed-1 (4B, SOTA retrieval) uses this — distills from zerank-2 reranker, outperforms OpenAI Large by +7% Recall@100.
   - We can approximate this without a reranker: use our best checkpoint as a teacher to score pairs, generating soft relevance labels for the next training round (self-distillation).

f. **Instruction-Tuning for Embeddings**:
   - Prepend task-specific instructions to queries (e.g., "Retrieve similar questions:", "Classify this text:").
   - Train with diverse task prefixes to improve zero-shot generalization.
   - See bge-en-icl (MTEB #3, score 71.67), E5-Mistral approach.

g. **Loss Functions**:
   - InfoNCE temperature sweep (0.01 to 0.2) — this has outsized impact.
   - Hard negative weighting: upweight loss contribution of hard negatives.
   - Cosine similarity loss for STS tasks alongside contrastive loss (multi-task).
   - Triplet loss with adaptive margin as alternative to InfoNCE.

### Tier 3: Data Strategy

h. **Data Quality over Quantity**:
   - Source stratification: balance data across STS, classification, clustering, retrieval domains.
   - LLM-labeled data: use Claude/GPT to generate high-quality pairs for weak categories (clustering, retrieval).
   - Positive-aware sampling: ensure in-batch negatives don't accidentally contain true positives.
   - Upload all cleaned datasets to HF under `pierretokns`.

i. **Synthetic Data for Weak Categories**:
   - Generate clustering-oriented data: topic-labeled documents from diverse domains.
   - Generate retrieval pairs: question-passage pairs for scientific and biomedical domains (SciFact, NFCorpus weaknesses).
   - Use source diversity: Reddit, StackExchange, Wikipedia, arXiv, PubMed.

j. **Base Model Exploration**:
   - ModernBERT-base (current, 149M) — strong baseline, SOTA for code and NLU.
   - ModernBERT-large (395M) — if memory allows.
   - Consider decoder-based backbones if time permits (top MTEB models all use Mistral/Llama).

### Tier 4: Novel / Experimental

k. **Diffusion-Based Pretraining**: Convert the encoder backbone using diffusion noise objectives before contrastive fine-tuning. See pplx-embed approach (MTEB multilingual #1).

l. **Contextual Embeddings**: Train context-dependent representations where the embedding of a passage changes based on surrounding context. See pplx-embed-context-v1-4B (ConTEB SOTA, 81.96 nDCG@10).

m. **Progressive Dimension Training**: Start training at low dimension (64), progressively increase to full dimension (768). Curriculum learning for representation quality.

**Ablation discipline**: when you have a new best, run systematic ablations before exploring new directions. Change ONE variable at a time. Mark these in results as `ablation: <variable>=<value>`.

## SOTA Context (What We're Competing Against)

| Model | Params | MTEB Eng Avg | Notes |
|-------|--------|-------------|-------|
| NV-Embed-v2 | 7B | 72.31 | #1 overall, latent attention pooling |
| bge-en-icl | 7B | 71.67 | Instruction-tuned, in-context learning |
| stella_en_1.5B_v5 | 1.5B | ~71 | Mistral-based |
| gte-modernbert-base | 149M | ~65* | Same backbone as ours, code SOTA |
| Our current best | 149M | 27.20 | Clean baseline, huge room to improve |

At 149M params, gte-modernbert-base (~65 MTEB avg) is our direct competitor. Closing the gap from 27 → 65 requires better pooling, better training, and better data — not more parameters.

## Anti-Cheat Rules

- All eval via official `mteb` library through `src/eval/mteb_runner.py` — no custom eval loops.
- Contamination check must pass (<1% per task) before any result counts.
- Training code must never import from eval; eval must never import from data.
- Document every data source in `src/data/registry.py` with URL, date, count, and hash.
- Evaluate on out-of-domain tasks periodically to ensure generalization.

## Key Research References

- **NV-Embed-v2**: #1 MTEB English. Latent attention pooling, instruction-tuning. Key insight: pooling strategy matters more than model size.
- **pplx-embed** (Perplexity): Diffusion pretraining + multi-stage contrastive. #1 MTEB Multilingual. arXiv 2602.11151
- **gte-modernbert-base**: Same ModernBERT backbone, ~65 MTEB avg. Code retrieval SOTA. Our direct competitor to study and beat.
- **jxmo blog**: LLM-labeled data to avoid false negative poisoning. https://blog.jxmo.io/p/how-to-train-the-best-embedding-model
- **Arctic-Embed** (Snowflake): Source stratification, hard negative mining with tunable threshold.
- **EmbeddingGemma**: Matryoshka representations, best sub-500M on MTEB.
- **zembed-1**: zELO distillation — continuous relevance scores from rerankers.
- **jina-embeddings-v3**: RoPE for long context, multilingual SOTA.
- **dewey_en_beta**: 128k context, open-source long-context leader.

## Stage Checkpoints & Resume

Training saves checkpoints after each stage (`checkpoints/stages/after_*.npz`). If a later stage crashes (e.g. OOM in mining), resume from where it left off instead of retraining from scratch:

```
uv run scripts/experiment.py "description" --resume-stage mining
```

Valid stages: `contrastive`, `mining`, `finetune`, `eval`. This loads the checkpoint from the prior stage and continues.

## Bug Fixes Applied (issue #2) — experiments to re-run

All config fields now work. All pooling modes have proper gradient flow. See issue #2 on GitHub for details.

**Invalidated experiments — re-run these ideas with fixed code:**
1. **Latent attention pooling** (exp-44/45/46 invalid — zero-init killed gradients). Use current best recipe.
2. **max_seq_length=512** (was hardcoded 256 for all 70 experiments). Test retrieval impact.
3. **Hard neg weight sweep** (exp-33→69 had hard negs silently disabled — `>1.0` condition fixed to `>0.0`). Try 0.5, 1.0, 2.0.
4. **gradient_accumulation=4** (was ignored, now works). Effective batch 256.
5. Combine winners from above.

## Resumption

On new session: read `results.jsonl` and `git log --oneline -20`. If uncommitted changes exist, `git checkout .`. Check the "Known Bug Corrections" section above for any pending fixes. Jump into the loop. Do not re-read this file or re-run setup — go straight to research.

## NEVER STOP

Run the experiment loop until interrupted. If out of ideas: re-read the references above, combine two near-winners, try radical architecture changes, generate synthetic data for your weakest category, or search HF Hub / Kaggle for new datasets. There is always something to try.

## Memory Budget

M2 Ultra: 64GB unified memory. MLX peak should stay well under 55GB (typical: ~6-10GB). If OOM, halve batch size and retry. If only a later stage OOMs, use `--resume-stage` to skip the stages that already completed.
