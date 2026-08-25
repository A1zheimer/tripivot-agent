# 数据资产：入库范围与重建方式

`data/` 的成品语料约 211MB、原始与中间缓存约 129MB，全部不进 Git。仓库只保留可
审计的小体积文件：manifest、构建摘要、术语库和人工复核工作表。下面列出「入库什么」
「忽略什么」，以及从零重建被忽略文件的命令。

数据本身不属于本仓库代码许可证：ALT 语料由 NICT 以 CC BY 4.0 发布，领域语料来自中文
维基百科（CC BY-SA 4.0）。使用时保留 manifest 中的来源与署名信息。

## 入库的文件

| 文件 | 内容 |
|---|---|
| `seeds/glossary.tsv` | 18 条人工 curated 术语，是 `glossary.jsonl` 的来源 |
| `processed/ALT.{en-zh,my-zh}.manifest.json` | 下载 URL、OPUS 元数据、压缩包 SHA-256、各类拒绝计数、切分计数 |
| `processed/bootstrap.summary.json` | ALT 构建摘要 |
| `processed/domain.summary.json` | 领域语料与术语分层的构建摘要、维护记录、质量口径 |
| `processed/glossary.jsonl` | 18 条 curated，翻译与训练的**默认**术语库 |
| `processed/glossary.silver.jsonl` + `.evidence.jsonl` + `.report.json` | 214 条机器审核术语，evidence 保留来源 URL、回译文本与相似度 |
| `processed/glossary.safe.jsonl` | 204 条（18 curated + 186 条双路相似度 ≥0.90 的 strict silver） |
| `processed/glossary.gold.jsonl` + `.report.json` | 232 条 model-assisted reviewed 术语 |
| `processed/glossary.{silver,gold}.review.tsv` | 逐条复核工作表；是人工判断的留档，无法由程序重新生成 |

复核工作表入库是有意的：`glossary.gold.jsonl` 可以由 `glossary.silver.review.tsv`
离线重放得到，所以 Gold 层不依赖任何被忽略的文件。

## 被忽略的文件

| 路径 | 体积 | 说明 |
|---|---|---|
| `raw/ALT/` | — | OPUS ALT 压缩包，manifest 里有 SHA-256 |
| `raw/wikipedia/` | — | 维基段落缓存 |
| `raw/distill/` | — | 蒸馏缓存，支持断点续跑 |
| `processed/ALT.*.{train,validation,test}.jsonl` | 约 21MB | ALT 双语与三语切分 |
| `processed/zh.{tech,intl,finance}.jsonl` | 约 29MB | 中文单语，每域 10,000 条 |
| `processed/zh-en.{tech,intl,finance}.jsonl` | 约 45MB | 中英蒸馏，合计 28,669 |
| `processed/zh-my.{tech,intl,finance}.jsonl`、`zh-my.parallel.jsonl` | 约 118MB | 中缅蒸馏，合计 22,233（`parallel` 是合并版） |
| `processed/glossary.domain.jsonl` | 约 4.7MB | 20,000 条 raw 自动术语，只归档不默认使用 |
| `processed/glossary.zh.candidates.jsonl` | 约 1MB | n-gram 挖掘中间产物 |

## 重建

`data/processed/*.jsonl` 的行数与内容由确定性切分和固定门禁决定，但蒸馏依赖模型输出，
逐字节复现不作保证；请以 manifest 与 summary 的计数口径为准。

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

## 训练快照 `artifacts/*.tar.gz`

`artifacts/qwen_training_data_v3.tar.gz`（约 32MB）把上述成品语料打成一个自包含
训练快照，同样不入库；入库的是 `artifacts/qwen_training_data_v3.json`，其中记录了
SHA-256，`scripts/run_qwen_hpc.sh` 会在解包前校验。

在 HPC 上有两条路：把快照带外拷贝到 `artifacts/` 下，或按上面的命令重建
`data/processed/`。两者都不满足时 launcher 会直接报错退出，不会静默跑一个空数据集。
