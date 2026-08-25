# 数据资产：入库范围与重建方式

`data/processed/` 的成品语料（约 211MB）随仓库分发：中文单语三域、zh-en 三域、zh-my
三域、`zh-my.parallel`、glossary*、summary/manifest。克隆后即可直接训练与评测。
`data/raw/`（约 129MB）仍不入库，需要时按下方命令重建。

数据本身不属于本仓库代码许可证：ALT 语料由 NICT 以 CC BY 4.0 发布，领域语料来自中文
维基百科（CC BY-SA 4.0）。使用时保留 manifest 中的来源与署名信息。

## 入库的文件（processed/）

| 文件 | 数量 / 内容 | 体积 |
|---|---|---|
| `ALT.{en-zh,my-zh}.{train,validation,test}.jsonl` | 各约 10,000 目标构建 | 约 12MB |
| `ALT.my-zh-en.{train,validation,test}.jsonl` | 中文枢纽精确对齐的三语数据 | 约 9MB |
| `zh.{tech,intl,finance}.jsonl` | 每域 10,000 条，每条 ≥100 汉字 | 约 29MB |
| `zh-en.{tech,intl,finance}.jsonl` | 合计 28,669 | 约 45MB |
| `zh-my.{tech,intl,finance}.jsonl` | 合计 22,233 | 约 59MB |
| `zh-my.parallel.jsonl` | 上面三个中缅文件的合并版 | 约 57MB |
| `glossary.domain.jsonl` | 20,000 条 raw 自动术语，只归档不默认使用 | 约 4.7MB |
| `glossary.zh.candidates.jsonl` | n-gram 挖掘中间产物 | 约 1MB |
| `glossary.jsonl` | 18 条 curated，翻译与训练的**默认**术语库 | 4KB |
| `glossary.silver.jsonl` + `.evidence.jsonl` + `.report.json` | 214 条机器审核术语 | 约 130KB |
| `glossary.safe.jsonl` | 204 条（18 curated + 186 条 strict silver） | 51KB |
| `glossary.gold.jsonl` + `.report.json` | 232 条 model-assisted reviewed 术语 | 约 69KB |
| `glossary.{silver,gold}.review.tsv` | 逐条复核工作表，人工判断留档 | 约 104KB |
| `ALT.*.manifest.json`、`bootstrap.summary.json`、`domain.summary.json` | 来源、SHA-256、切分计数、维护记录 | 约 12KB |
| `seeds/glossary.tsv` | 18 条人工 curated 术语种子 | — |

`zh-my.parallel.jsonl` 是仓库里最大的单文件（约 57MB），低于 GitHub 100MB 硬限制；
超过 50MB 时 GitHub 会提示但允许推送。它与三个分域中缅文件内容重复。

## 被忽略的文件

| 路径 | 体积 | 说明 |
|---|---|---|
| `raw/ALT/` | — | OPUS ALT 压缩包，manifest 里有 SHA-256 |
| `raw/wikipedia/` | — | 维基段落缓存 |
| `raw/distill/` | — | 蒸馏缓存，支持断点续跑 |
| `../artifacts/*.tar.gz` | 约 93MB | 可由 `data/processed/` 重新打包；JSON manifest 仍入库 |
| `../deliverables/` | 约 267MB | 汇报快照，与仓库现有数据重复 |
| `processed/*.sqlite3` | — | Agent 运行时状态，非数据资产 |

## 重建 raw / 刷新成品

成品已入库；下面的命令用于重建被忽略的 `data/raw/`，或从零刷新 `processed/`。
行数与内容由确定性切分和固定门禁决定，但蒸馏依赖模型输出，逐字节复现不作保证；
请以 manifest 与 summary 的计数口径为准。

### 1. ALT 双语与三语枢纽数据

```bash
translation-agent corpus bootstrap \
  --max-records 10000 \
  --raw-dir data/raw \
  --output-dir data/processed
```

### 2. 领域语料（科技 / 国际 / 财经）

先采集中文单语，再蒸馏中英与中缅（可中断续跑，本机 MPS 约 3 小时）：

```bash
translation-agent corpus fill-domains --collect-only --per-domain 10000

python -m pip install -e '.[nllb]'
translation-agent corpus fill-domains \
  --distill-backend nllb \
  --my-parallel 30000 \
  --glossary-terms 20000
```

退化循环清洗已内联在管线里；对存量文件可单独跑
`translation-agent corpus clean-distilled` 与 `translation-agent corpus clean-glossary`。

### 3. 术语分层

依赖第 2 步产出的 `glossary.domain.jsonl`：

```bash
python scripts/fetch_wikipedia_glossary.py   # 维基跨语言标题 → 候选
python scripts/build_safe_glossary.py        # → silver / safe + 报告与 evidence
python scripts/apply_glossary_review.py      # 复核工作表 → gold（离线，无需网络）
```

三个脚本的默认输入输出路径都指向 `data/processed/`，无参数直接运行即可。
`glossary.gold.jsonl` 可以只由入库的 `glossary.silver.review.tsv` 离线重放得到。

### 4. 重新打包训练快照

```bash
tar -czf artifacts/qwen_training_data_v4.tar.gz data/processed/*.jsonl
shasum -a 256 artifacts/qwen_training_data_v4.tar.gz
```

把校验和写进新的 `artifacts/qwen_training_data_v4.json`，不要覆盖已发布的版本。
在 HPC 上也可直接使用仓库内的 `data/processed/`，无需 tar 快照。
