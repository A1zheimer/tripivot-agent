# One-command Qwen LoRA training on HPC

This is the Qwen-only path. Gemma is intentionally not trained by this script.

## What it runs

The launcher:

1. verifies and extracts `artifacts/qwen_training_data_v3.tar.gz` if needed
2. creates `.venv-hpc`
3. installs the project with the `training` extra
4. checks CUDA/device capability
5. prepares four mirrored directions from the cleaned bilingual data:
   - `zh-en` and `en-zh`
   - `zh-my` and `my-zh`
6. injects only the 18-entry curated glossary by default
7. trains Qwen2.5-7B-Instruct with LoRA
8. saves the best adapter and tokenizer
9. evaluates a stratified 1,000-example validation subset during training, then
   generates 200 validation examples and reports BLEU, chrF, and required-term accuracy
10. writes `training_summary.json`

The test split is loaded by the data checker but never used as SFT or validation data.

## Direct run on an interactive GPU node

```bash
bash scripts/run_qwen_hpc.sh
```

Useful dry run:

```bash
DRY_RUN=1 bash scripts/run_qwen_hpc.sh
```

The dry run still loads the Qwen tokenizer and tokenizes the dataset, but it does not
load/train the model. It writes `runs/qwen-lora/data_summary.json`.

## SLURM

Edit the partition/module lines at the top:

```bash
vi scripts/qwen_lora_sbatch.sh
sbatch scripts/qwen_lora_sbatch.sh
```

If the cluster has no torch module, uncomment:

```bash
export INSTALL_TORCH=1
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
```

## Important controls

```bash
# Two epochs is the recommended first run.
EPOCHS=2

# Lower this before lowering gradient accumulation if CUDA OOM occurs.
TRAIN_BATCH=2

# Reviewed terminology ablation. glossary.gold.jsonl has 232 entries:
# 214 Codex-reviewed Wikipedia-title terms plus 18 curated seeds.
GLOSSARY=data/processed/glossary.gold.jsonl bash scripts/run_qwen_hpc.sh

# More conservative ablation using only the 18 curated seeds.
GLOSSARY=data/processed/glossary.jsonl bash scripts/run_qwen_hpc.sh

# For a shorter smoke run:
DRY_RUN=1 bash scripts/run_qwen_hpc.sh
```

Additional arguments after the launcher are forwarded to the Python trainer:

```bash
bash scripts/run_qwen_hpc.sh --max-train-samples 2000 --max-eval-samples 200
```

## Resume

The default policy is `--resume auto`. If `runs/qwen-lora/checkpoint-*` exists, the newest
checkpoint is resumed automatically. Start from scratch with:

```bash
bash scripts/run_qwen_hpc.sh --resume never
```

## Outputs

```text
runs/qwen-lora/
├── prepared_examples.jsonl
├── data_summary.json
├── checkpoint-*/
├── final_adapter/
├── validation_predictions.jsonl
└── training_summary.json
```

`final_adapter` is the deliverable. `training_summary.json` records the git commit, data
counts, tokenization counts, runtime, train metrics, and generation metrics.

## Dataset artifact

`data/processed/` is tracked in Git, so a fresh clone already carries the training data
and the launcher's extraction step is skipped. See
[../data/README.md](../data/README.md) for the file inventory and refresh commands.

The `artifacts/*.tar.gz` snapshots are not tracked; they are an optional way to move the
dataset as one checksummed file. If an archive is present and
`data/processed/zh-en.tech.jsonl` is not, the launcher verifies its SHA-256 against the
tracked JSON manifest and extracts it. If neither the data nor an archive is present, it
exits with an error instead of training on an empty dataset.

If the bilingual corpus changes materially, create a new versioned artifact rather than
overwriting the released one.
