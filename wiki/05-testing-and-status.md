# 测试、已知问题与路线图

> **TL;DR**
> 32 个 pytest 用例守住预处理、语料构建、本地复用、端到端主路径，以及 2026-08-17 修复批次的行为（代码围栏、断点续跑、执法口径、词界匹配与符号术语、后端重试、双侧歧义、退化检测、术语清洗、空译文拒绝、词表冲突告警），并覆盖前置 ingestion 与 GFM 列表/表格保结构。设计评审发现的四个高危问题已全部修复，对抗式复审发现的次生问题（检索排序、围栏符号混用、快照损坏路径、缩进代码泄漏等）也已修复。自动评测（BLEU/COMET 等）仍未接入。下一步：HPC 微调 → GRPO 强化学习 → 向量 RAG → 三条件对照评测。

[← 返回首页](./README.md) · [上一页：数据管线](./04-data-pipeline.md)

## 测试覆盖

[test_core.py](../tests/test_core.py) 共 32 个用例，`pytest` 一条命令全跑：

| 用例 | 验证什么 | 手法亮点 |
|---|---|---|
| `test_text_normalization_and_quality_filter` | 归一化去 BOM/NBSP/压缩空白；干净 en-zh 对被接受；Zawgyi 输入抛 `ValueError` | 直接锁「脏数据宁可拒绝」的行为契约 |
| `test_build_parallel_corpus_deduplicates_and_filters` | 伪造 Moses zip：正常对接受、重复对去重、非缅文源/含 HTML 目标被拒，manifest 计数与输出一致 | 内存构造归档，不依赖网络 |
| `test_collect_opus_reuses_local_archive_without_api` | 本地归档存在时**绝不调 OPUS API**（`local_reuse=True`） | monkeypatch 让下载函数一被调用就 `AssertionError` |
| `test_long_horizon_agent_plans_remembers_and_enforces_terms` | 端到端：规划/渲染结构、术语命中计数、无 issue、SQLite 写入 2 条、进度文件存在 | `MappingBackend` 查表替身，确定性输出 |
| `test_pack_passages_enforces_hundred_chinese_chars` | 维基段落打包后每段 ≥100 汉字 | 任务书硬指标的回归测试 |
| `test_classify_domain_prefers_distinct_keywords` | 三域判别 + 近乎并列时弃权；长术语「计算机科学」可整词挖出、无虚词碎片 | 覆盖分类边际与术语挖掘新规则 |
| `test_conflicting_glossary_terms_reported_not_double_enforced` | 同源词多译法：检索去重只执法一条 + 文档级 `glossary_conflict` 告警 | 词表冲突可见且不产生不可满足执法 |
| `test_symbol_edge_terms_match` | `C++`/`.NET`/`C#` 符号边缘术语命中，`AI` 不误命中 `said` | lookaround 词界 |
| `test_mixed_fence_markers_do_not_leak_code` | ``` 围栏内的 ~~~ 行不提前关栏、代码不外泄 | 围栏符号感知 |
| `test_corrupt_progress_snapshot_is_ignored` | 快照为合法 JSON 但非对象时安全忽略 | 恢复路径容错 |
| `test_empty_backend_content_is_rejected` | 空译文记 `chunk_error`，失败标题渲染源文占位 | 拒绝假成功 |
| `test_code_fence_preserved_verbatim` | 代码围栏字节级保留、围栏内 `#` 不算标题、代码不送模型 | 围栏感知规划的回归测试 |
| `test_resume_skips_completed_chunks` | 单块失败→占位继续；重跑只重译失败块 | 断点续跑的行为契约 |
| `test_enforcement_matches_injected_glossary` | 25 个术语命中时只执法注入的 20 个 | 锁死「执法=注入」口径 |
| `test_word_boundary_and_casefold_target_matching` | `AI` 不误命中 `said`；目标侧大小写不敏感 | 词界与对称归一化 |
| `test_audit_reports_variant_inconsistency` | 同源术语多译法触发 inconsistent 报告 | 审计判据拆分 |
| `test_openai_backend_retries_then_reports_truncation` | HTTP 503 退避重试后成功；`finish_reason=length` 报截断错误 | monkeypatch 假 urlopen，不依赖服务 |
| `test_align_on_pivot_rejects_first_side_ambiguity` | first 侧重复枢轴被跳过并计数 | 双侧歧义检测 |
| `test_degenerate_and_boundary_detectors` | 退化循环检出、正常枚举不误伤；术语边界过滤 | 两档阈值的行为锚点 |
| `test_clean_glossary_file_drops_unusable_entries` | 碎片/标点伪术语被删，curated 保留，统计准确 | 清洗函数契约 |
| `test_agent_preserves_frontmatter_lists_tables_and_code` | frontmatter、列表、表格、代码围栏翻译后结构不变 | GFM 保结构回归 |
| `test_ingestion_normalizes_markdown_and_writes_report` | 换行归一化、结构统计、sidecar report | canonical Markdown 契约 |
| `test_ingestion_rejects_unclosed_fence` | 未闭合 fenced code 为 blocking warning | 转换失败前置拦截 |
| `test_docx_ingestion_uses_pandoc_and_reports_structure` | DOCX 走 pandoc adapter，段落/表格统计进入报告 | 外部转换边界 |
| `test_docx_tracked_changes_are_blocking` | tracked changes 阻止进入 Agent | 防止翻译未定稿内容 |
| `test_pdf_ingestion_removes_repeated_edges` | 文本型 PDF 页眉/页脚重复行剔除，页数与覆盖率入报告 | PDF 启发式边界 |
| `test_cli_translate_auto_ingests_before_agent` | TXT 自动转 canonical Markdown，sidecar 与翻译报告贯通 | CLI ingestion 集成 |
| `test_training_data_mirrors_directions_and_injects_gold_terms` | 双语对自动镜像四语向；仅注入 18 条人工术语中的命中项 | Qwen SFT 数据构造 |

**仍未覆盖**（改到相关代码时请补）：更多 CLI 参数与错误分支、真实 pandoc/pdfplumber 端到端转换、`NllbBackend` 真实调用、维基采集与蒸馏的真实网络路径、繁简转换（未实现，见下）。

## 已知问题

| 状态 | 问题 | 位置与说明 |
|---|---|---|
| ✅ 已修复（2026-08-17） | `translate` 子命令 `NameError` | [cli.py](../src/translation_agent/cli.py) 曾引用未导入的 `GlossaryMemory`/`MemorySystem`，已补 `from .memory import ...` |
| ✅ 已修复（2026-08-17） | 翻译侧无断点续跑 | `ProgressMemory` 补 `load`；`translate_document` 重跑跳过已完成块并重建 L1 上下文；单块失败记 `chunk_error` 后继续，CLI 写出部分产物并以退出码 1 结束。README 旧版「可恢复」的说法现在属实 |
| ✅ 已修复（2026-08-17） | 术语注入/执法口径分裂（top20 vs top100） | 反思器改为只检查注入提示词的同一批术语；`plan.glossary` 全局检索已随之移除（无消费者的死计算） |
| ✅ 分层复核（2026-08-19） | raw 术语库仍有语义噪声 | `glossary.domain.jsonl` 保持 raw/quarantine，不默认用于强约束或 RL reward。214 条 silver 已逐条复核：177 ACCEPT / 37 FIX / 0 REJECT，生成 232 条 model-assisted Gold（含 18 条 curated），casefold 冲突 0。该层仍需母语缅文译者最终签核；默认翻译/训练仍可用 18 条 curated，Gold 用于显式消融 |
| ✅ 已修复（2026-08-17/18） | zh-my 蒸馏语料 24% 退化循环 | 管线内置两档退化检测（枚举不误伤）+ 存量清洗，随后本机重蒸馏扩池（每域用满 1 万段）至 **22,233** 条，全量退化扫描 0 残留 |
| ✅ 已修复（2026-08-17） | 后端无重试、截断静默、大小写不对称、代码围栏被切碎、修订盲重译、NLLB 无效重试 | 分别对应：退避重试 + `TranslationBackendError`、`finish_reason` 检查、目标侧 casefold、围栏感知规划、修订带旧稿与 expected 指令、后端能力声明（`revision_capable`） |
| ⚠️ 待办 | 无自动评测指标 | `TranslationReport` 只有过程指标（命中/修订/issue）；`pyproject.toml` 的 `eval` extra（sacrebleu）尚未在代码中使用，BLEU/COMET/chrF++/TCR 对照实验在路线图 |
| ⚠️ 待办 | 繁简不统一 | 中文语料约半数条目含繁体（维基遗留），术语按字形分裂。`domain.summary.json` 现已记录每域繁体占比；彻底解决需引入 opencc（会破坏零依赖原则，放 `[nllb]` extra 或采集侧转换） |
| ℹ️ 行为说明 | L2 情景记忆仍只写不读 | 定位是审计日志；模糊匹配复用（TM）在路线图 |
| ℹ️ 行为说明 | 换术语表重跑不放弃恢复 | resume 只比对语向与风格；换了 `--glossary` 会复用旧译文，由 `audit_document` 事后报告新术语缺失。需要强制重译用 `--no-resume` |
| ℹ️ 待办 | `--api-key` 走命令行会进 shell 历史 | 生产建议只用 `TRANSLATION_API_KEY` 环境变量 |
| ✅ 已修复（2026-08-18） | ingestion 对抗式复审的 5 个高危 | 中文/缅文开头的合法 UTF-8 被误判二进制（16 字节前缀截断多字节字符）、UTF-16 BOM 死代码、PDF 重复边缘句误删正文（现仅删边缘行 + 80% 阈值 + 删除告警）、无闭合 frontmatter 静默吞全文、```` 围栏含 ``` 行提前闭合泄漏代码 |
| ℹ️ 已知限制 | 手写 GFM 长尾结构不保形 | setext 标题（下划线式）、blockquote、短分隔符表格（`\|--\|`）、正文紧邻列表（无空行）、松散列表空行——这些结构会退化为普通段落送模型。pandoc（DOCX/HTML）产物已被归一化不受影响；纯手写 md 建议先经 `ingest` 走 pandoc html 路径或避免这些写法 |
| ℹ️ 已知限制 | 表格逐行分块：第 5 行起失去表头上下文 | L1 窗口=4；且模型返回行数与列数不符时整行原样回填。改进方向：行块提示词附带表头（在路线图「结构感知分块」） |
| ℹ️ 已知限制 | GBK 等非 UTF-8 编码被拒（消息已统一并提示转码），但「恰好是合法 UTF-8 的 GBK 字节」仍可能以乱码穿透 | 拒绝大多数场景；乱码穿透需引入编码启发检测（未做）。建议源头转 UTF-8 |
| ✅ 已修复（2026-08-18） | 训练评测把 4 个语向混在一个 BLEU/chrF 里 | `evaluate_generation` 现在同时输出 overall 与 `by_direction`，报告优先看 zh-en/en-zh/zh-my/my-zh 分组指标 |
| ✅ 已修复（2026-08-18） | `--load-in-4bit` 依赖未声明 | 新增 `[qlora]` extra（`bitsandbytes`）；A800 默认 LoRA 路径仍不需要安装它 |
| ℹ️ 已知限制 | `python -m translation_agent.training_data` 导出含 split=test 行 | 训练脚本自身的 `prepared_examples.jsonl` 已正确排除 test；仅模块 CLI 的导出包含——勿直接拿它喂 SFT |
| ℹ️ 已知限制 | 训练评测 tokenizer 被改为左填充后随 final_adapter 一起保存 | 下游加载适配器会继承 `padding_side=left`，通常无害 |
| ℹ️ 已知限制 | max_length=1536 下 zh-my 训练样本约丢 7% | A800 上可提到 2048 捞回；launcher 的 GLOSSARY 默认值已跟随 DATA_DIR |
| ℹ️ 已知限制 | 英文按 `。！？!?` 断句，ASCII `.` 不切分 | 避免 `3.14`/`U.S.` 误断的保守取舍；英文长段可能整段成块 |

## 路线图

按[阶段研究报告](../deliverables/feishu_export/TriPivot-Agent_阶段研究报告.md)的口径：

1. **LoRA/QLoRA 监督微调**（进行中）：在港科广 HPC 单卡 A800 上对 Qwen2.5-7B-Instruct 与 Gemma2 训练低秩适配器；本机 Apple M4 Pro 仅做数据与训练格式校验。
2. **Agentic RL（GRPO）**：把「分块翻译 → 术语检查 → 是否重试」写成多步决策，训练信号为术语一致率、COMET/BLEU 奖励与重试代价惩罚。
3. **向量 RAG**：在 L3 现有精确子串匹配之上加向量召回，覆盖同义表述。
4. **对照评测**：同一测试划分、三条件（裸 Qwen / 裸 Gemma2 / 完整智能体），指标 BLEU、COMET、chrF++（缅甸语加报）与 TCR；FLORES+ 仅作独立评测集。COMET 未就绪前先报 BLEU 与 TCR。
5. **硬约束解码**：Outlines / vLLM Trie 作为新 backend 平行接入，不改动 Agent、记忆与数据接口。
6. **交付物**：Demo 接入与环境演示录屏（实习答辩项）。

## 深度背景读物

- [TriPivot-Agent 阶段研究报告](../deliverables/feishu_export/TriPivot-Agent_阶段研究报告.md) —— 研究问题、方法论、数据与训练计划的全貌（wiki 各页的许多结论出处）。
- [主 README](../README.md) —— 安装与命令的权威出处。

---

[← 返回首页](./README.md)
