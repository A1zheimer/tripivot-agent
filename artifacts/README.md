# Qwen training artifact

`qwen_training_data_v3.tar.gz` is a self-contained 31 MB training snapshot for the
Qwen-only LoRA run. It exists so the training data can move to an HPC node as a single
checksummed file instead of 211 MB of loose JSONL.

The archives themselves are not in Git (`artifacts/*.tar.gz` is ignored); only these
JSON manifests are, and each one records the SHA-256 of its archive. Copy an archive in
out of band, or rebuild `data/processed/` using [../data/README.md](../data/README.md).

Contents:

- six cleaned bilingual domain corpora (`zh-en` and `zh-my`)
- the 18-entry curated glossary (`glossary.jsonl`)
- `glossary.safe.jsonl`: 204 entries (18 curated + 186 strict silver)
- `glossary.silver.jsonl`, its evidence file, and its quality report
- `glossary.silver.review.tsv`, the 214-row review worksheet
- `glossary.gold.jsonl`, `glossary.gold.review.tsv`, and `glossary.gold.report.json`
  (232 model-assisted reviewed entries: 177 ACCEPT, 37 FIX, 0 REJECT, plus 18 curated)
- `domain.summary.json`

The HPC launcher extracts and checksum-verifies this archive automatically when
`data/processed/zh-en.tech.jsonl` is absent. Do not edit files inside the archive;
create a new versioned artifact instead.
