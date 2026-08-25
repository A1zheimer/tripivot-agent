# Long-Horizon 多语文档翻译 Agent

面向 `my→zh`、`en→zh` 的可运行 Python 框架，核心接口对 `my / zh / en` 任意语向通用。项目同时提供可复现的 OPUS 语料下载、Unicode/语言质量过滤、去重、切分、三语精确枢纽对齐和术语库构建流程。

## 当前纵向能力

- 前置 ingestion：Markdown/TXT/HTML/DOCX/PDF 统一转入 canonical Markdown，并输出结构校验报告。
- 层级规划：Markdown 标题、段落、列表、表格、fenced code → 句块。
- 四层记忆：
  - L1 滑动窗口上下文；
  - L2 SQLite 句对、术语命中和修订记录；
  - L3 三语术语语义记忆（精确命中检索）；
  - L4 每个句块后原子落盘的进度树。
- 执行：OpenAI/vLLM 兼容接口或本地 NLLB-200。
- 约束与反思：术语 prompt 软约束、逐块缺失审计、定向重试、全文一致性审计。
- 数据：ALT 的 `my-zh`、`en-zh` 各 10,000 条目标构建，以及通过相同中文句精确连接的 `my-zh-en` 三语数据。

当前版本采用“软约束 + 审计重试”作为稳定基线。Outlines/vLLM Trie 硬约束解码可在不改变 Agent、记忆和数据接口的情况下作为后续 backend 加入。

## 安装与测试

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

本地 NLLB 基线需要额外依赖：

```bash
python -m pip install -e '.[nllb]'
```

文本型 PDF ingestion 需要额外依赖；DOCX/HTML 转换复用系统 `pandoc`：

```bash
python -m pip install -e '.[documents]'
```

## 构建语料

一条命令完成下载、校验、过滤、去重、确定性切分、三语对齐和术语种子构建：

```bash
translation-agent corpus bootstrap \
  --max-records 10000 \
  --raw-dir data/raw \
  --output-dir data/processed
```

主要输出：

```text
data/processed/
├── ALT.my-zh.{train,validation,test}.jsonl
├── ALT.en-zh.{train,validation,test}.jsonl
├── ALT.my-zh-en.{train,validation,test}.jsonl
├── ALT.my-zh.manifest.json
├── ALT.en-zh.manifest.json
├── glossary.jsonl
└── bootstrap.summary.json
```

每个双语 manifest 记录下载 URL、OPUS 元数据、压缩包 SHA-256、原始/接收/去重/各类拒绝数量。中缅两个数据集按照中文枢纽句的哈希切分，防止三语对齐后跨训练集和测试集泄漏。

`data/processed/` 成品语料（约 211MB）与 `data/seeds/` 已入库，克隆后即可直接训练评测，无需重跑数小时的采集与蒸馏；`data/raw/` 的原始下载与蒸馏缓存（约 129MB）不入库，可按命令离线重建。各文件的数量、体积与重建命令见 [data/README.md](data/README.md)。

默认选择 ALT，是因为它同时覆盖英语、简体中文和缅甸语，语料翻译部分由 NICT 以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 发布。数据不是本仓库代码许可证的一部分；使用时仍应保留 manifest 中的来源和署名信息。

单独构建某个 OPUS 语料：

```bash
translation-agent corpus collect-opus \
  --corpus ALT --pair my-zh --max-records 10000 \
  --license CC-BY-4.0
```

语料准入规则包括 NFC 归一化、控制字符/HTML 清理、缅甸文脚本比例、中文脚本比例、长度比、重复句对和高置信 Zawgyi 检查。疑似 Zawgyi 且没有可靠转换器的记录会被拒绝，避免污染 Unicode-only 下游数据。

## 实习领域语料（科技 / 国际 / 财经）

中文单语已按任务书凑齐：三个领域各 10,000 条，每条不少于 100 个汉字，来源为中文维基百科（CC BY-SA 4.0），保留标题和 URL。

```bash
# 只采集中文
translation-agent corpus fill-domains --collect-only --per-domain 10000

# 蒸馏中英 3 万 + 中缅 2 万（可中断续跑）
python -m pip install -e '.[nllb]'
translation-agent corpus fill-domains --distill-backend nllb --glossary-terms 0
```

输出：

```text
data/processed/zh.{tech,intl,finance}.jsonl
data/processed/glossary.zh.candidates.jsonl
data/raw/distill/{domain}.zh-en.jsonl   # 蒸馏缓存
data/raw/distill/{domain}.zh-my.jsonl
```

中英使用 `Helsinki-NLP/opus-mt-zh-en`，中缅使用 NLLB-200 distilled 600M。蒸馏支持断点续跑。

术语库分为多层：`glossary.domain.jsonl` 是 raw 自动术语，只归档不默认使用；`glossary.silver.jsonl` 从真实 Wikipedia 标题出发并经英/缅双路回译校验；`glossary.safe.jsonl` 合并双路相似度 ≥0.90 的 strict silver 与 18 条人工术语；`glossary.gold.jsonl` 是 232 条 model-assisted reviewed 术语（177 ACCEPT / 37 FIX / 0 REJECT，含 18 条 curated）。默认仍用 18 条 curated；`GLOSSARY=data/processed/glossary.gold.jsonl` 可显式开启复核术语消融。

## Qwen-only HPC 自动训练

仓库提供一个单模型、单命令的 Qwen LoRA 训练入口，不训练 Gemma：

```bash
bash scripts/run_qwen_hpc.sh
```

SLURM 用户修改 `scripts/qwen_lora_sbatch.sh` 中的 partition/module 后提交：

```bash
sbatch scripts/qwen_lora_sbatch.sh
```

默认使用 `Qwen/Qwen2.5-7B-Instruct`、四个双语方向、18 条人工术语、2 epochs LoRA，
并自动断点续跑。训练数据直接来自入库的 `data/processed/`，克隆后即可开跑；
`artifacts/*.tar.gz` 打包快照只是可选的搬运方式（本身不入库，SHA-256 记在同名 JSON
manifest 里）。详见 [docs/qwen_hpc_training.md](docs/qwen_hpc_training.md)。

## 翻译文档

### 前置文档归一化

Agent 的内部契约是 canonical GFM Markdown。可以单独把 DOCX/HTML/TXT/PDF/Markdown 转成该契约，并检查 fenced code、语言脚本、DOCX tracked changes、PDF 文本覆盖率等：

```bash
translation-agent ingest \
  --input inputs/report.docx \
  --output work/report.md \
  --report work/report.ingestion.json \
  --source en
```

`translate` 也会自动执行同一层 ingestion，并把 canonical Markdown 落盘为 `<output stem>.source.md`，便于复核原始转换结果。PDF 目前是启发式文本抽取，适合文本型 PDF；扫描 PDF 需要先 OCR。

### vLLM / OpenAI 兼容服务

```bash
export TRANSLATION_BASE_URL=http://localhost:8000/v1
export TRANSLATION_API_KEY=optional

translation-agent translate \
  --input examples/technical_en.md \
  --output outputs/technical_en.zh.md \
  --source en --target zh \
  --backend openai \
  --model Qwen/Qwen2.5-7B-Instruct \
  --domain technology
```

### NLLB-200 本地基线

```bash
translation-agent translate \
  --input path/to/input.my.md \
  --output outputs/input.zh.md \
  --source my --target zh \
  --backend nllb \
  --model facebook/nllb-200-distilled-600M
```

译文旁会生成 `*.report.json`，包含 ingestion 摘要、句块进度、术语命中、修订次数、未解决问题和 backend 元数据。`.translation-agent/` 中保存可恢复的情景记忆与进度快照。

## 代码结构

```text
src/translation_agent/
├── agent.py          # 规划、执行、反思、渲染
├── backends.py       # vLLM/OpenAI 与 NLLB backend
├── corpus.py         # 下载、过滤、切分、枢纽对齐
├── ingestion.py      # DOCX/PDF/HTML/TXT → canonical Markdown 与校验
├── memory.py         # L1-L4 记忆
├── models.py         # 稳定的数据契约
├── preprocessing.py # Unicode、Zawgyi 检查、脚本质量、切句
└── cli.py
```

## 质量边界

- 自动构建的平行句对需要按用途进行人工抽检；manifest 的计数不是语义正确性的保证。
- 种子术语表仅用于打通工程链路，生产或论文实验前需由母语译者复核扩充。
- ALT 是新闻通用域，不等同于技术领域语料。领域数据应通过同一 schema 增量加入，并单独记录许可证和来源。
- FLORES+ 更适合作为独立评测集，不应混入训练语料。

## 更多文档

面向新接手同学的讲解见 [wiki/](wiki/README.md)：快速上手、架构总览、翻译主循环、数据管线与语料资产、测试与路线图。本 README 的安装与命令段落是权威出处，wiki 负责原理与现状。
