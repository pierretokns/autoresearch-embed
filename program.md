# autoresearch-embed

Autonomous embedding model research on Apple Silicon. Train, evaluate, and iterate on text embedding models using contrastive learning, with MTEB as the ground-truth benchmark.

## Your Workflow

1. **Read state**: `cat results.jsonl | python3 -c "import sys,json; [print(json.loads(l)['status'], json.loads(l)['primary_score'], json.loads(l)['description']) for l in sys.stdin]"`
   Or: `git log --oneline -20`
2. **Decide what to try** (see Research Strategy below)
3. **Edit code** in `src/` or `configs/`
4. **Run**: `uv run scripts/experiment.py "description of what you changed"`
5. **Read the output summary**. Think about what worked and what to try next.
6. GOTO 1

That's it. The script handles git, training, eval, contamination checks, results logging, wandb, agenthub posting, and git push. You just research.

## What You Can Modify

- `src/train.py`, `src/model.py`, `src/losses.py`, `src/data/*.py`, `configs/*.yaml`
- Everything about training: architecture, optimizer, hyperparameters, stages, data loading, loss functions

## What You Cannot Modify

- `src/eval/` — READ-ONLY (integrity boundary)
- `scripts/` — READ-ONLY (automation)
- Never run `git add`/`commit`/`push` directly — `experiment.py` handles it
- Never run `train.py` directly — always use `experiment.py`
- Never import from `src/eval/` in training code (or vice versa)

## Framework: MLX (Native Apple Silicon)

Training uses MLX natively — no PyTorch, no MPS. Key patterns:
- Model is `mlx.nn.Module` in `src/model.py` (ModernBERT encoder + projection)
- Training uses `nn.value_and_grad()` + `mlx.optimizers.AdamW`
- Tokenizer returns `return_tensors="np"`, convert to `mx.array()` for forward pass
- `encode_sentences()` returns numpy for MTEB compatibility
- No `.to(device)`, no `autocast`, no `empty_cache` — MLX handles memory automatically
- Decontaminated data cached in `data_cache/` keyed by dataset config hash

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

**Quick eval**: STSBenchmark, SICK-R, TwitterURLCorpus (~2 min). **Full eval**: all 8 (~10 min).

## Research Strategy

Priority order — pick ONE change per experiment:

a. **Data**: new datasets, sampling ratios, synthetic data, decontamination. Upload cleaned datasets to HF under `pierretokns`.
b. **Loss**: InfoNCE temperature, hard negative weight, margin loss.
c. **Schedule**: stage durations, learning rates, warmup/cooldown ratios.
d. **Architecture**: pooling (CLS vs mean vs weighted), projection dim, layer selection, freezing early layers.
e. **Hard negatives**: mining top-k, mining frequency, filtering threshold.
f. **Synthetic data**: generate for domains where scores are weakest.
g. **Base model**: try different encoders (Qwen3-0.6B, ModernBERT, etc.).

**Ablation discipline**: when you have a new best, run systematic ablations before exploring new directions. Change ONE variable at a time (pooling, projection dim, temperature, LR, batch size). Mark these in results as `ablation: <variable>=<value>`.

## Anti-Cheat Rules

- All eval via official `mteb` library through `src/eval/mteb_runner.py` — no custom eval loops.
- Contamination check must pass (<1% per task) before any result counts.
- Training code must never import from eval; eval must never import from data.
- Document every data source in `src/data/registry.py` with URL, date, count, and hash.

## Key Research References

- **pplx-embed** (Perplexity): Diffusion-continued pretraining + multi-stage contrastive. arXiv 2602.11151
- **jxmo blog**: LLM-labeled data to avoid false negative poisoning. https://blog.jxmo.io/p/how-to-train-the-best-embedding-model
- **Arctic-Embed** (Snowflake): Source stratification, hard negative mining with tunable threshold.
- **EmbeddingGemma**: Best sub-500M on MTEB, Matryoshka representations.

## Resumption

On new session: read `results.jsonl` and `git log --oneline -20`. If uncommitted changes exist, `git checkout .`. Jump into the loop. Do not re-read this file or re-run setup — go straight to research.

## NEVER STOP

Run the experiment loop until interrupted. If out of ideas: re-read the references above, combine two near-winners, try radical architecture changes, generate synthetic data for your weakest category, or search HF Hub / Kaggle for new datasets. There is always something to try.

## Memory Budget

M2 Ultra: 64GB unified memory. MLX peak should stay well under 55GB (typical: ~6-10GB). If OOM, halve batch size and retry via `experiment.py`.
