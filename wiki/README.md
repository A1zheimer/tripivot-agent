# TriPivot-Agent Wiki

> **TL;DR**
> TriPivot-Agent（包名 `translation-agent`）是一个面向**中英缅三语**、以**中文为枢纽语言（pivot language）**、以**长文档跨段落术语一致性**为核心目标的翻译智能体框架，配套「语料构建 → 蒸馏 → 术语挖掘」的离线数据管线。名字由来：**Tri** = 三种语言（zh / en / my），**Pivot** = 以中文作为枢纽——先在中文侧固定领域表述，再经由中文分别产出英、缅平行文本。
>
> 本 wiki 面向新接手的同学，目标是 30 分钟内对项目建立完整认知。安装与命令的**权威出处是[主 README](../README.md)**，wiki 负责讲清原理与现状，两者分工避免内容双维护。

## 页面导航

| 页面 | 回答的问题 | 什么时候读 |
|---|---|---|
| [快速上手](./01-getting-started.md) | 我怎么把它跑起来？翻译一篇文档会产出什么？ | 第 1 步，边读边操作 |
| [架构总览](./02-architecture.md) | 代码是怎么组织的？模块之间怎么调用？ | 跑通之后，改代码之前 |
| [翻译主循环详解](./03-translation-loop.md) | 系统的灵魂——一个句块从进入到输出经历了什么？ | 想理解/修改翻译质量逻辑时 |
| [数据管线与语料资产](./04-data-pipeline.md) | 数据从哪来、长什么样、能不能直接用？ | 接手数据或训练任务时 |
| [测试、已知问题与路线图](./05-testing-and-status.md) | 现状边界在哪？接下来要做什么？ | 任何时候都值得先扫一眼 |

## 5 分钟速览

```bash
# 1. 装环境（零运行时依赖，纯标准库）
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'

# 2. 跑测试（应全过）
pytest

# 3. 翻译一篇文档（需要一个 OpenAI 兼容服务，如 vLLM）
export TRANSLATION_BASE_URL=http://localhost:8000/v1
translation-agent translate \
  --input examples/technical_en.md \
  --output outputs/technical_en.zh.md \
  --source en --target zh --model Qwen/Qwen2.5-7B-Instruct
```

细节、参数与输出解读见[快速上手](./01-getting-started.md)。

## 核心概念小词典

| 概念 | 含义 |
|---|---|
| 枢纽语言（pivot language） | 以中文为中介：中文先固定表述，英、缅两侧经由中文生成，避免两侧各自发明译名 |
| 长程循环（long-horizon loop） | 规划 → 逐句块翻译 → 反思 → 修订 → 记忆 → 全文审计的主循环，见[翻译主循环](./03-translation-loop.md) |
| 句块（chunk） | 翻译的最小调度单元：标题单独成块，正文每 6 句一块，带 `heading:` / `paragraph:` 注记用于还原结构 |
| 四层记忆 L1–L4 | L1 工作记忆（滑窗上下文）、L2 情景记忆（SQLite 句对档案）、L3 语义记忆（术语检索）、L4 进度记忆（原子检查点，支持断点续跑） |
| TerminologyReflector | 术语反思器：句块级检查「源术语出现而目标译名缺失」，缺失则带修订说明重译 |
| 软约束 + 审计重试 | 当前基线策略：术语通过提示词软性约束，靠事后审计与定向重试兜底；硬约束解码（Outlines/Trie）留作后续 backend |
| 蒸馏（distillation） | 用 Opus-MT / NLLB 把中文单语批量翻译成英/缅平行语料的过程，产物即「蒸馏语料」 |
| Zawgyi / Unicode | 缅甸语两种编码体系；本管线强制 Unicode，高置信 Zawgyi 且无法可靠转换的记录直接拒绝 |
| TCR（术语译准率） | 规划评测指标之一：规定译名的命中率 + 全文译法唯一性，与 BLEU / COMET / chrF++ 并列 |

## 仓库地图

| 位置 | 内容 | 备注 |
|---|---|---|
| [src/translation_agent/](../src/translation_agent/) | 核心包，10 个模块（含 ingestion 前置归一化） | 各模块职责见[架构总览](./02-architecture.md) |
| [tests/test_core.py](../tests/test_core.py) | 32 个 pytest 用例 | 覆盖面与空白见[测试与现状](./05-testing-and-status.md) |
| [data/seeds/](../data/seeds/) | 人工种子术语表 `glossary.tsv`（18 条） | confidence 1.0，仅用于打通链路 |
| [data/raw/](../data/raw/) | 原始与中间缓存：ALT 归档、维基段落、蒸馏缓存 | 可删除后离线重建 |
| [data/processed/](../data/processed/) | 成品数据（2026-08-18 重蒸馏后约 17.7 万行 JSONL）+ manifest | **唯一权威数据源** |
| [deliverables/](../deliverables/) | 2026-08-14 飞书汇报打包快照 + zip | **清洗前**的数据快照（根 `data/` 此后已清洗，两者不再相同），勿在此改数据 |
| [docs/](../docs/)、[examples/](../examples/) | 阶段研究报告、示例文档 | 深度背景读物 |
| [pyproject.toml](../pyproject.toml) | 构建与依赖配置 | 运行时零依赖；`nllb` / `eval` / `dev` 为可选 extra |

---

上一页：无 · 下一页：[快速上手](./01-getting-started.md)
