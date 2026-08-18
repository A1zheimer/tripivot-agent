# 快速上手

> **TL;DR**
> 三步跑起来：装环境（零运行时依赖）→ `pytest` 验证 → 用 OpenAI 兼容服务或本地 NLLB 翻译一篇文档。译文旁会生成 JSON 质量报告，`.translation-agent/` 里保存情景记忆与进度快照——重跑同一文档会自动跳过已完成的句块（断点续跑），个别句块失败不会中断整篇（失败块保留原文占位，退出码非零）。本页末尾附全部 CLI 命令速查表。

[← 返回首页](./README.md) · [下一页：架构总览 →](./02-architecture.md)

## 环境准备

需要 Python ≥ 3.11（代码用了 `StrEnum`、zip 严格模式等特性）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

31 个用例应全部通过。运行时**零第三方依赖**（纯标准库），只有用到本地 NLLB 时才需要额外安装：

```bash
python -m pip install -e '.[nllb]'   # sentencepiece / torch / transformers / datasets
```

文本型 PDF ingestion 需要 `documents` extra；DOCX/HTML 使用系统 `pandoc`：

```bash
python -m pip install -e '.[documents]'
```

## 前置文档归一化

Agent 只吃 canonical GFM Markdown。`ingest` 命令负责把 Markdown/TXT/HTML/DOCX/PDF 转成这个内部契约，并输出 sidecar 校验报告：

```bash
translation-agent ingest \
  --input inputs/report.docx \
  --output work/report.md \
  --report work/report.ingestion.json \
  --source en
```

检查包括：文件魔数、fenced code 成对、源语言脚本比例、Zawgyi、DOCX tracked changes / 批注 / VBA、PDF 文本覆盖率与重复页眉页脚。`translate` 会自动执行同一流程；即使直接传 `.docx` / `.pdf`，也会落盘 `<output stem>.source.md` 和 `.ingestion.json`，方便复核转换边界。PDF 当前为启发式文本抽取，扫描件需先 OCR。

## 翻译第一篇文档

### 路径 A：vLLM / OpenAI 兼容服务（推荐）

任意 `/chat/completions` 兼容服务都可以（vLLM 部署的 Qwen、Gemma 等）：

```bash
export TRANSLATION_BASE_URL=http://localhost:8000/v1   # 必填
export TRANSLATION_API_KEY=optional                     # 服务有鉴权时才需要

translation-agent translate \
  --input examples/technical_en.md \
  --output outputs/technical_en.zh.md \
  --source en --target zh \
  --backend openai \
  --model Qwen/Qwen2.5-7B-Instruct \
  --domain technology
```

`--source` / `--target` 取值 `en | zh | my`，任意语向可用；`--domain` 会影响术语检索的域过滤。

### 路径 B：本地 NLLB 基线（无需服务）

```bash
translation-agent translate \
  --input examples/technical_en.md \
  --output outputs/technical_en.zh.md \
  --source en --target zh \
  --backend nllb \
  --model facebook/nllb-200-distilled-600M
```

把 `--source`/`--target` 换成 `my`/`zh` 即可翻译缅文文档（自备一篇缅文 Markdown 即可，仓库目前未附带示例）。

## 输出产物解读

一次翻译会产出三类东西：

| 产物 | 位置 | 内容 |
|---|---|---|
| 译文 | `--output` 指定路径 | 按原文 Markdown 结构还原的译文 |
| 源文档归一化 | `<output stem>.source.md` | 非 Markdown 输入（DOCX/PDF/HTML/TXT）转换后的 canonical Markdown，可复核 |
| ingestion 报告 | `<output stem>.ingestion.json` | 来源格式/sha256/转换器/结构指标/警告 |
| 质量报告 | `<output>.report.json`（可用 `--report` 改路径） | `document_id`、`total/completed_chunks`、`terminology_hits`（术语命中累计）、`revisions`（重译次数）、`issues`（未解决的反思问题）、`backend_metadata`（每次调用的模型与 token usage） |
| 状态目录 | `.translation-agent/`（可用 `--state-dir` 改路径） | `episodes.sqlite3` 情景记忆 + `progress/{document_id}.progress.json` 进度检查点，长文档可中断后凭检查点排查 |

`document_id` 是「语向 + 归一化全文」的 SHA-256 前 16 位，同一文档同一语向幂等——重复翻译会 UPSERT 同一批记录，不会膨胀。

## CLI 命令速查

入口命令 `translation-agent`，由 [src/translation_agent/cli.py](../src/translation_agent/cli.py) 定义。

### `ingest` — 文档前置归一化

把 DOCX/PDF/HTML/TXT/Markdown 转成 canonical Markdown 并做结构校验（扫描 PDF、未闭合围栏、DOCX 修订追踪、Zawgyi 会被拒绝）；`translate` 会对输入自动执行同一层：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` / `--output` | 必填 | 源文档 / 转出的 Markdown |
| `--report` | `<output>.ingestion.json` | 结构指标（页数/表格/围栏/警告）与来源 sha256 |
| `--source` | 无 | 声明源语言，用于脚本比例校验 |
| `--strict` | 关 | 任何警告都拒绝（默认仅 blocking 项拒绝） |

### `translate` — 翻译长文档

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` / `--output` | 必填 | 输入输出路径 |
| `--source` / `--target` | 必填 | `en` / `zh` / `my`，语向任意组合 |
| `--model` | 必填 | 模型名（传给后端服务或 NLLB） |
| `--backend` | `openai` | `openai`（vLLM/OpenAI 兼容）或 `nllb`（本地） |
| `--glossary` | `data/processed/glossary.jsonl` | 术语表（L3 语义记忆的数据源） |
| `--state-dir` | `.translation-agent` | 记忆与进度目录 |
| `--domain` / `--register` | `general` / `formal` | 写入 StyleGuide，影响术语域过滤与提示词 |
| `--base-url` / `--api-key` | 环境变量 | 覆盖 `TRANSLATION_BASE_URL` / `TRANSLATION_API_KEY` |
| `--timeout` / `--max-retries` | `180` / `2` | 单次请求超时秒数；429/5xx/网络错误指数退避重试次数 |
| `--no-resume` | 关 | 忽略进度快照，从头重译 |
| `--strict-ingestion` | 关 | 输入 ingestion 出现任何警告即拒绝翻译 |
| `--report` | `<output>.report.json` | 报告输出路径 |
| `--ingestion-report` | `<output stem>.ingestion.json` | 前置转换报告路径 |
| `--strict-ingestion` | 关 | 任一 warning 都拒绝继续（blocking warning 默认拒绝） |

### `ingest` — 前置文档归一化

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` / `--output` | 必填 | 支持 `.md` / `.txt` / `.html` / `.docx` / `.pdf`；输出 canonical GFM |
| `--report` | `<output>.ingestion.json` | 源文件哈希、转换器、结构统计与 warnings |
| `--source` | 空 | 按指定语言做脚本比例与 Zawgyi 校验 |
| `--strict` | 关 | 任一 warning 都失败 |

### `corpus` — 语料管线（详见[数据管线](./04-data-pipeline.md)）

| 子命令 | 作用 | 关键参数 |
|---|---|---|
| `collect-opus` | 从 OPUS 下载一个语言对并清洗切分（本地有归档时直接复用，不联网） | `--pair my-zh`、`--corpus ALT`、`--max-records 10000`、`--force-download` |
| `build-archive` | 跳过下载，直接处理已下载的 OPUS Moses zip | `--archive <zip>`、`--corpus`、`--pair` |
| `bootstrap` | 一键：采 ALT 双语 → 三语枢轴对齐 → 种子术语表 → 写 bootstrap.summary.json | `--glossary-seed data/seeds/glossary.tsv` |
| `fill-domains` | 领域语料：采集中文维基 3 域 → 蒸馏 zh-en/zh-my → 挖术语（断点续跑，含退化检测与拒绝原因统计） | `--distill-backend nllb\|openai`、`--collect-only`、`--glossary-only`、`--per-domain 10000` |
| `clean-glossary` | 清洗术语表：删虚词边界碎片、带句末标点/超长英文译名的条目（人工 curated 条目保留） | `--glossary data/processed/glossary.domain.jsonl` |
| `clean-distilled` | 清洗蒸馏语料：删含重复循环（退化输出）的行 | `--output-dir data/processed` |

## 常见问题

- **`translate` 报 `NameError`？** 旧版本 [cli.py](../src/translation_agent/cli.py) 漏了 `from .memory import GlossaryMemory, MemorySystem`，已于 2026-08-17 修复；若你检出的是旧提交，补上这一行即可。
- **没跑通测试就先看报告文件**：`.report.json` 里 `issues` 非空表示术语反思后仍有缺失（`terminology` 类是句块级、`global_terminology` 类是文档级），`revisions` 表示触发过重译。
- **想复现论文里的数据**：先读[数据管线](./04-data-pipeline.md)的 manifest 一节，所有构建都有 sha256 与计数可查。

---

[← 返回首页](./README.md) · [下一页：架构总览 →](./02-architecture.md)
