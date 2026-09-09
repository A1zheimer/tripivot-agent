# 答辩前待补充事项

> 生成时间：2026-09-08。基于对 `artifacts/eval/`、`scripts/pilot_lib.py`、`docs/conclusions.md`
> 的复核。按优先级排列，P0 为会被评委直接问到、且当前答不上来的项。
>
> 落脚点是**翻译智能体**，因此下列缺口中「文档级证据」（§3）是最关键的一项。

---

## P0-1 指标口径：BLEU 用错分词器，两处结论符号相反

**现状**：`scripts/pilot_lib.py` 的 `score_bleu()` 使用 sacrebleu 默认 `13a` 分词器：

```python
def score_bleu(hyps, refs):
    bleu = BLEU()          # 默认 tokenize='13a'
    return bleu.corpus_score(hyps, [refs]).score / 100.0
```

`13a` 不切分中文与缅甸语。中文无空格，整个小句会被当作单个 token，匹配退化为长字符串精确匹配，
对输出长度极度敏感。

**影响**：用已落盘的 `artifacts/eval/hyps_*.jsonl` 重算（中文 `tokenize='zh'`，缅甸语用字符级近似）：

| 项 | 已发布 | 修正后 |
|---|---|---|
| my→zh BLEU | −18% | **+80%**（9.15 → 16.48） |
| OPD en-zh BLEU | −57% | **+4%**（24.78 → 25.82） |
| GRPO en-zh BLEU | +27% | **+5%**（24.78 → 26.03） |
| zh→my BLEU | ×6.7 | 约 **×1.8**（24.42 → 42.98，字符级） |
| en→zh BLEU | +5% | +8%（22.93 → 24.78） |
| zh→en BLEU | +16% | +16%（不变，英文用 13a 本就正确） |

`OPD BLEU −57%` 基本是分词器伪影：OPD 输出偏短（长度比 0.737），在 13a 下长块匹配崩得最厉害。
**COMET 与 chrF++ 不受影响**（chrF 为字符级，对中缅文天然正确），结论 1 的主体不变。

**风险**：en→zh 的 BLEU 4.9 配 COMET 0.79 是自相矛盾的组合——BLEU 不到 5 意味着「基本不会翻译」。
做过机器翻译的评委会一眼看出异常并追问算法。

**需要补**：
- [ ] `score_bleu()` 增加目标语言参数，按语言选分词器（zh→`zh`，my→`char` 并注明为近似，en→默认）
- [ ] `eval_pilot.py` 调用处传入 `args.direction` 的目标语；输出 JSON 增加 `bleu_tokenizer` 字段
- [ ] 用已保存的 hyps 重算 12 个 `artifacts/eval/*.json` 的 BLEU（**不需重新解码、不需 GPU**）；
      旧值保留为 `bleu_tok13a` 以便审计
- [ ] 回填 `conclusions.md` 结论 1 表格、`training_engineering_report.md` §3.2、
      `defense_ppt_material.md` P5/P10

---

## P0-2 显著性检验：设计文档承诺过，但没做

**现状**：`docs/opd_post_training_design.md` §10.6 明确写了「显著性：配对 bootstrap（1000 次，p<0.05）」，
但 `conclusions.md` 结论 2 直接以点估计给出排序（「GRPO 综合最优」「OPD 单句质量最优」）。

**实测**（配对 bootstrap，n=252，1000 次重采样，vs baseline）：

```
BLEU-zh    baseline 24.78 | opd 25.82  p=0.230 | grpo 26.03  p=0.127
chrF2++    baseline 22.06 | opd 21.73  p=0.211 | grpo 22.42  p=0.098
```

三个系统两两之间**均不显著**。COMET 无法检验（只落盘了均值，未保存逐句分数），
但 0.7921 / 0.7984 / 0.7927 的跨度仅 0.006，几乎可以确定同样不显著。

**需要补**：
- [ ] 新增 `scripts/significance_test.py`，把配对 bootstrap 固化为可重跑脚本，
      产出 `artifacts/eval/significance.json`
- [ ] 结论 2 改为：**试点规模下三者无显著差异**；唯一稳健的差异是 OPD 的行为特征
      （长度比 0.737、退化率 0.00%），符合反向 KL 的 mode-seeking 预期
- [ ] 补充空结果的机理解释：en→zh 恰是四方向中 SFT 增益最小的（COMET 0.794→0.792 持平），
      句级空间已近饱和，后训练边际收益有限——空结果本身是有信息量的
- [ ] 逐句 COMET 分数在后续评测中落盘，以便做显著性检验

**附带**：报告中的 chrF（0.2579）是**逐句 chrF 求平均**，标准 sacrebleu corpus chrF 为 0.2206。
需在文档注明算法，否则他人复现对不上。

---

## P0-3 文档级证据缺失（落脚点是 Agent，这是最大缺口）

**现状**：`scripts/` 下所有评测均为句级（`eval_pilot.py` 比较 raw/sft/opd/grpo 的单句翻译）。
**没有任何 Agent 层面的测量。**

**问题**：汇报落脚于翻译智能体，讲了完整的规划 → 检索 → 翻译 → 反思 → 修订 → 记忆 → 审计循环，
却没有一个数字证明这套机制有用。评委问「你的 Agent 相比直接调模型好在哪」时无法回答。
句级指标在结构上也不可能回答这个问题。

**通路已经具备，不需要写新的推理代码**：
- CLI 已支持 `translate --backend openai --model X --base-url ...`
- vLLM 自带 OpenAI 兼容服务（`vllm serve`）
- `audit_document` 已经产出所需指标：术语缺失、译法冲突（同一术语跨 chunk 出现不同译法）

**需要补**：三组对照，同一篇文档，指标取自 `audit_document` 加结构完整率与失败块数

| 组 | 后端 | Agent 机制 | 说明 |
|---|---|---|---|
| A | 原版 Qwen2.5-7B-Instruct | 完整 | 训练对 Agent 的贡献（A→B） |
| B | SFT 模型 | 完整 | 完整系统 |
| C | SFT 模型 | 关闭（逐块直调，不注入术语、不反思、不修订） | Agent 本身的贡献（C→B） |

- [ ] HPC 上 `vllm serve` 起合并后的 SFT 模型
- [ ] 写 C 组旁路脚本（约几十行：复用 `HierarchicalPlanner` 切块后直调后端）
- [ ] 写 `scripts/eval_document.py` 汇总三组的术语缺失数、译法冲突数、结构完整率、失败块数
- [ ] 结果进 `conclusions.md` 作为新的结论（Agent 层面的证据）

**优先级说明**：若时间只够做一件事，做 **C→B**（同模型，Agent 开/关），因为它直接证明落脚点。

---

## P1-1 定位表述：RL 的作用范围要讲清楚

**现状**：项目落脚于「Long-Horizon 文档翻译 Agent」，但 GRPO 是**单步**的——
一个 prompt 采 8 条译文，各打一个标量分，组内标准化后更新，horizon = 1，
无状态转移、无环境交互、无跨步信用分配。这属于 contextual bandit，
与 RLHF / DeepSeek-R1 式 GRPO 同一范畴，不是 agentic RL。

OPD 则**根本不是 RL**：损失为 `0.6·反向KL(教师) + 0.4·NLL(参考)`，无奖励、无策略梯度。
「on-policy」仅指训练数据采自当前学生策略。设计文档写作「教师蒸馏 vs 指标奖励 RL」是准确的，
PPT 上不要退化成「两种 RL 方法对比」。

**需要补**：
- [ ] `conclusions.md` 「诚实边界」增加一条：本项目的 RL 作用于 Agent 调用的**句级翻译策略**，
      Agent 自身的规划与修订决策仍为规则驱动
- [ ] PPT P15 预设 QA 增加「这算 agentic RL 吗」，答案如上，并给出下一步动机
- [ ] 展望部分用**译法冲突**做论据：该失败模式是纯文档级属性——单看任一 chunk 译文都正确，
      冲突只在跨 chunk 比较时出现——句级奖励在原理上无法覆盖，必须由 episode 级奖励 +
      跨步信用分配解决。这使「下一步做 agentic RL」由自有数据中的失败模式驱动，而非追概念

---

## P1-2 zh→en 回退：建议改为「发现」而非「待解释的麻烦」

**现状**：四方向中 zh→en 出现 COMET −5%（0.766→0.727）、退化率翻倍（11%→22%），
文档目前将其作为引出后训练的动机一笔带过。

**建议表述**：四方向混合语料的单一 LoRA 存在**方向间容量竞争**——低资源方向大幅获益
（zh→my COMET +42%）的同时高资源方向回退。改进方向为分方向 adapter 或语向重采样。

这条「下一步」完全由自有数据推出，比 agentic RL 更难被质疑为追概念，建议作为展望的第一条。

- [ ] 写入 `conclusions.md` 与 PPT P14

---

## P1-3 试点方向选择：需要提前准备说法

**现状**：后训练试点选在 en-zh，而按结论 1，该方向 SFT 增益最小（COMET 持平）。
即把后训练实验放在了 headroom 最小的方向上，叠加 n=252 的规模，是结论 2 真正的软肋。

- [ ] PPT P15 增加预设 QA：诚实答「受限于试点算力，选了数据最充分的方向做通路验证；
      事后看该方向 headroom 最小，全量版应优先 zh→my」

---

## P2-1 演示形态：改为文档级

**现状**：`demo/` 当前为句级三方对比（SFT / OPD / GRPO 并排），服务于「后训练方法对比」这一落脚点。

**问题**：落脚点是 Agent，则应演示**它翻一整篇文档**：原文/译文对照、术语命中高亮、
Markdown 结构保持、结束后弹出审计报告（如「12 个术语全部命中、0 处译法冲突」）。
vLLM 服务复用同一个，改的是前端展示内容。

- [ ] 前端改为文档级视图
- [ ] 展示 `audit_document` 的输出

**附**：`examples/technical_en.md` 仅 357 字节，太短，跑不出跨 chunk 的译法冲突效果。
需准备一篇至少数十句、含标题与列表结构、且术语重复出现的文档。

- [ ] 准备演示文档

---

## P2-2 设计文档与代码漂移

`docs/opd_post_training_design.md` §10 与实现不一致，答辩被追问时容易露怯：

| 项 | 设计文档 §10 | 代码实际 |
|---|---|---|
| GRPO 奖励 | `0.6·COMET_QE + 0.2·chrF + 0.1·术语命中 + 0.1·格式` | `0.75·chrF + 0.25·格式 − 长度惩罚`；**术语命中未实现** |
| KL 锚 β | 0.04 | 0.02（工程报告写的 0.02 是对的） |
| OPD 教师 | 32B（en-zh） | 14B |
| 长度惩罚阈值 | 1.3 | 1.4 |

工程报告基本准确，设计文档是旧版。

- [ ] 对齐设计文档 §10，或在开头标注「§10 为设计稿，实测配置以工程报告 §3.1 为准」

---

## P2-3 `reward_bundle()` 中的哑弹

`scripts/pilot_lib.py` 调用 COMET 时传二元组，而 `score_comet()` 要拆三元组：

```python
comet = score_comet(list(zip(srcs, hyps)), ...)   # 二元组
...
data = [{"src": s, "mt": h, "ref": r} for s, h, r in triples]   # 需要三元组 -> ValueError
```

当前未暴露，因为 COMET 仓库 401 下载失败、`use_comet` 恒为 False 走了兜底分支——
即这行代码从未被执行过。一旦 COMET 下载成功，GRPO 第一次算奖励即崩溃。

另：存在性检查看的是 `.cache/huggingface/hub/models--Unbabel--wmt22-comet-qe-da`（免参考 QE 模型），
而 `score_comet()` 实际加载 `.cache/comet/wmt22-comet-da`（有参考模型），路径与模型均不一致。
`eval_pilot.py` 那边传的是三元组，是正确的。

- [ ] 修正调用签名与路径检查

---

## 补充说明：SFT 产物本身已校验通过

以下为独立核验结果，可在答辩中作为产物可靠性的依据：

- 权重文件 SHA-256 与魔搭记录一致，161,533,192 字节，无截断
- 392 个张量 = 28 层 × 7 个投影 × (A, B)，形状全部匹配 Qwen2.5-7B 几何，无缺失、无多余、无 NaN/Inf
- PEFT 期望的 key 集合与文件精确相等，196 个 Linear 层全部正确包裹
- 196 组 `lora_B` **无一为零矩阵**（PEFT 初始化时 B 恒为零，非零即证明优化器确实步进过）
- 逐层更新幅度 ‖BA‖_F 稳定分布于 31–41，无塌缩或爆炸；权重增量 RMS 约 1e-3，
  与 10,772 步 @ lr 1e-4 的配置吻合

**注意**：魔搭仓库权重位于 `adapter/` 子目录，`PeftModel.from_pretrained` 需指向该子目录而非仓库根；
模型卡引用的 `training_summary.json` 尚未上传。
