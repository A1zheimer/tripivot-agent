# 后训练对比实验：OPD vs GRPO（同基座双分支）

> 状态：设计稿 v3。前置条件：SFT LoRA（作业 12645536，runs/qwen-lora）训练完成并通过评测。
> **实验目标：回答"同一 SFT 基座上，OPD（教师蒸馏）与 GRPO（指标奖励 RL）哪种后训练更适合本翻译任务"。**
> 结构：双分支平行对比——
>
> ```
>                ┌─ 分支A：SFT基座 → OPD-R1  ─┐
> SFT 合并基座 ──┤                           ├─→ 统一评测台（同 held-out、同指标）→ 对比报告
>                └─ 分支B：SFT基座 → GRPO-R1 ─┘
> ```
>
> 两分支**同一初始化（SFT 合并基座）、同一数据分布、同一评测集**，只变后训练方法，保证归因干净；各自独立 adapter/仓库/日志。对比胜出者（或 OPD→GRPO 串行组合）作为后续 P6 扩展，不进本轮主实验。
> 方法学：OPD 稠密教师信号、样本效率高但受教师上限约束；GRPO 稀疏奖励直接优化指标、可能超教师但贵且需防 hacking——正好是教科书级对照。

## 1. 为什么是 On-Policy Distillation

SFT 只在人工/蒸馏参考译文（教师分布外的旧分布）上做极大似然，学生推理时的自回归分布与训练分布不一致。OPD 的做法：

1. 从**当前学生策略**采样译文（on-policy）；
2. 教师模型对学生采样出的序列做前向，给出逐步 logits；
3. 学生在**自己的采样分布**上向教师做 **反向 KL**（mode-seeking）对齐。

相对优势：
- 对比 SFT/离线蒸馏：容量不再浪费在学生已经会 tokens 上；
- 反向 KL 防止学生平均化多峰译文分布（模糊/吞术语的根源之一）；
- 教师在学生的错误上给信号，等于定向纠错。

方法参考：GKD（Agarwal et al., 2023）与 on-policy distillation 的公开实践（Thinking Machines, 2025）。

## 2. 教师选择（受同词表约束）

逐步 logit 蒸馏要求教师与学生**同 tokenizer**（Qwen2.5 系全家族共享 151,936 词表）。

| 方向 | 教师 | 理由 |
|---|---|---|
| zh-en / en-zh | **Qwen2.5-32B-Instruct**（bf16 ≈66G，独占一张 A800-80G） | 英中能力显著强于 7B；同词表可逐步蒸馏 |
| zh-my / my-zh | **Qwen2.5-14B-Instruct**（bf16 ≈28G） | 32B 对缅文提升有限，14B 显存友好；若 COMET 显示教师弱于学生（`t2s` 检验，见 §6），该方向退回 α=0 只用参考 NLL |

GLM-4 等其它家族词表不同，只能做句子级偏好信号，不进本轮设计。

## 3. 损失函数

对每个 prompt 采 k=4 条学生译文，逐条计算：

```
L = α · KL(p_student ‖ p_teacher | y~student) + β · NLL(y_ref) + γ · R_glossary
```

- **KL 项**：reverse KL，仅在学生采样 token 上计算；温度 τ=1（蒸馏温度教师/学生各 1/τ，先不做温度搜索，v2 再调）；
- **NLL 项**：同一 prompt 的参考译文做标准 SFT 损失，防止策略漂移出数据流形（μ=JKD 思想的混合）；
- **R_glossary**：术语惩罚的软化——对 18 条 curated 术语命中 span，若学生 token 与教师 top-1 不一致则该 span loss 上调（λ=2）；缺失术语沿用 Agent 审计逻辑的判定，不额外造硬约束；
- 初始 α=0.6, β=0.4, γ=0.2；每 1000 步在 1k held-out 上跑 COMET，若下降则 α×0.5。

超参（起点，非圣旨）：

| 项 | 值 |
|---|---|
| lr | 5e-6（SFT 的 1/20），cosine，warmup 50 步 |
| batch | 有效 16（per-device 2 × grad-acc 8，与 SFT 一致） |
| epochs | 1 轮 prompt 子集 20k×4 采样 ≈ 5,000 步 |
| 采样 | temperature 0.7, top_p 0.95, max_len 1536 |
| LoRA | 在 **SFT 合并后的基座**上新开 adapter（r=16 同 SFT），归因干净、可回滚 |

## 4. On-Policy 的工程妥协：chunked 采样

严格逐步 on-policy 需要每步用最新策略采样，vLLM 与训练循环同进程冲突。采用**块级 on-policy**（实践上与逐步差异小，GKD 论文的 on-policy 分数即在此设定下取得）：

```
for chunk in prompts[::2000]:            # 每 2k prompts 一块
    adapters = merge(current_lora)       # 或 vLLM --enable-lora 热载
    samples  = vllm.generate(chunk, k=4, temp=0.7)   # GPU0，~8 min/块
    teacher_logits = teacher.forward(chunk × samples) # GPU1，逐块前向缓存
    train(samples, teacher_logits, refs, steps=500)  # GPU0
```

若 TRL ≥0.9 的 `GKDTrainer` + vLLM live-sampling 在环境里可用（vllm 0.9.2 ↔ torch 2.7.1 已对齐），则替换为该实现，减自研代码。

## 5. 资源与队列（沿用现有 HPC2 打法）

- 申请 `--gres=gpu:2`（GPU0 学生采样+训练，GPU1 教师）。emergency_gpu 配额 8 卡可容纳；拿不到 2 卡时的降级：单卡 14B 教师 + 学生分时复用（采样/训练交替），时长 ×1.8；
- 环境：复用 `.venv-hpc`（torch 2.7.1+cu126 / transformers 4.46.3 已钉死），追加 `pip install vllm==0.9.2 trl`；模型：Qwen2.5-32B 经 hf-mirror 预下载（~65GB，先在登录节点 nohup 拉好，或第一个作业里下）；
- 时长估算：采样 20k×4 ≈ 1.5h；教师前向 ≈ 3h；训练 5k 步 ×6s ≈ 8h；合计 **13–16h**，emergency 48h / long_gpu 14d 均可兜底；
- 产物：`runs/opd-r1/final_adapter` + `comet.json` + `samples/`（抽检用）。

## 6. 评测与护栏

- 指标（每方向）：**COMET**（wmt22-comet-da，经 hf-mirror）、chrF++、BLEU（sacrebleu，已有依赖）、术语命中率、长度比、格式完整率（复用仓库 ingestion 校验器）；
- held-out：ALT test + FLORES+ devtest（README 已注明 FLORES+ 只做评测不进训练）；
- **教师有效性检验（t2s）**：开跑前先让教师与 SFT 学生在 500 条 held-out 上对评——教师 COMET 必须显著高于学生（配对 bootstrap p<0.05），否则该方向降级（α=0）；
- 停机条件：连续 2 次 eval COMET 下降 / 学生熵坍缩（采样熵 <3.0 nat/token）/ 重复退化率 >2%；
- 发布门槛：四方向 COMET 全面 ≥ SFT，且术语命中率不降；100 条/方向人工抽检后再改公开。

## 7. 阶段规划（对比实验版）

| 阶段 | 内容 | 预计 | 产物 |
|---|---|---|---|
| P0 | SFT 完成 → 合并基座 + baseline 评测表落盘 | +0.5d | runs/qwen-lora + eval_baseline.json |
| P1 | 环境准备：vllm/trl/COMET 模型、教师权重（32B/14B）预下载、t2s 教师检验 | +0.5d | — |
| **P2a（分支A）** | **OPD-R1：四方向**（英中 32B 教师、中缅 14B 视 t2s） | +2d | runs/opd-r1 |
| **P2b（分支B）** | **GRPO-R1：四方向**（奖励与配置见 §10，**从 SFT 合并基座起步**） | +1.5d | runs/grpo-r1 |
| P3 | 统一评测台跑双分支 + SFT baseline 三方对比（§10.6） | +0.5d | 对比报告 |
| P4（扩展，可选） | 胜者再训 R2，或 OPD→GRPO 串行组合 | 另计 | — |

> 计算资源允许时 P2a/P2b 并行（各需 2×A800）；单队列串行时先跑 GRPO（时长短半天，且若 GRPO 已超教师，OPD 分支的解读更有趣）。

---

## 11. 双分支并行执行编排（P2a/P2b 同时开跑）

### 11.1 机制：两个独立 sbatch 作业

```
sbatch scripts/opd_sbatch.sh    # 作业A：--gres=gpu:2（GPU0 学生、GPU1 教师）
sbatch scripts/grpo_sbatch.sh   # 作业B：--gres=gpu:2（GPU0 策略、GPU1 奖励模型+ref logprob）
```

- 两作业**互不依赖、各自排队**：谁先拿到资源谁先跑，拿到的时间可以错开——这不是缺点，是鲁棒性（emergency 队列给 2 卡的时机不可控，独立作业能吃到任何碎片窗口）；
- 配额检查：emergency_gpu 用户上限 8 卡，2+2=4 卡 ✓；CPU 4+4=8 核 ✓；
- 与前面竞速模式的区别：**这里不设裁判取消**——两个作业都要跑到终点。

### 11.2 前置共享资产（P1 一次性备好，两分支只读）

| 资产 | 位置 | 说明 |
|---|---|---|
| SFT 合并基座 | `runs/qwen-lora/merged_base` | `peft merge_and_unload` 后保存，两分支同一初始化 |
| 教师权重 | `.cache/huggingface`（Qwen2.5-32B/14B） | 登录节点 hf-mirror 预下载，32B 约 65GB |
| 奖励模型 | `.cache/huggingface`（wmt22-comet-qe-da） | GRPO 分支用，约 2.3GB |
| 统一评测台 | `scripts/eval_suite.py` | 三方对比同一入口，held-out 固定种子 |
| venv | `.venv-hpc` 追加 `vllm==0.9.2 trl` | 版本与 torch 2.7.1 对齐后钉死 |

### 11.3 降级阶梯（拿不到 2 卡时）

1. **首选**：2 作业 × 2 卡（真并行，~1.5–2 天全部完成）
2. **降级 1**：2 作业 × 1 卡——OPD 换 14B 教师与学生同卡分时；GRPO 用 vLLM colocate（`gpu_memory_utilization=0.35`）+ CPU offload ref logprob；各慢 ~1.8 倍但仍在跑
3. **降级 2**：串行——先 GRPO 后 OPD（或反之），总时长 ≈ 两分支之和
4. 队列策略：两作业可各挂一个队列分身（emergency + long_gpu）提高命中，谁先跑起来 scancel 同分支的另一个分身（沿用已验证的裁判模式）

### 11.4 监控与产物

- 每作业独立日志 `logs/opd-%j.out` / `logs/grpo-%j.out`，训练内 eval 每 1000 步（OPD）/ 100 步（GRPO）写 `runs/{opd,grpo}-r1/eval_history.jsonl`；
- 断点：两脚本均 checkpoint 续跑（沿用 SFT 的自动续训模式）；
- 产物上传：overnight_upload 流水线参数化仓库名后，`runs/opd-r1/final_adapter` → `tripivot-...-opd-r1`、`runs/grpo-r1/final_adapter` → `tripivot-...-grpo-r1`。

---

## 10. GRPO 分支设计（P2b）

### 10.1 定位与前提

- 输入策略 = **SFT 合并后的基座**（对比实验要求与 OPD 分支同一起点，公平对照）；
- 目标：直接优化翻译任务指标，验证"指标奖励 RL"相对"教师蒸馏"的优劣；重点防 reward hacking 与熵坍缩。

### 10.2 奖励函数（可解释、可消融）

对每个 prompt 采 G=8 条译文，逐条打分：

```
R = 0.6·COMET_QE + 0.2·chrF++(vs ref) + 0.1·术语命中 + 0.1·格式完整
    − 0.3·max(0, len_ratio−1.3) − 0.5·重复退化(repetition≥4gram)
```

- **COMET-QE**：`Unbabel/wmt22-comet-qe-da`（免参考，src+hyp 输入，2.2B 模型单卡常驻）；chrF++ 锚定忠实度防 COMET 被流畅空洞文骗过（reward hacking 主防线）；
- 术语命中/格式完整复用仓库现有审计逻辑（glossary + ingestion 校验器），打分器全部本地可跑（集群无外网依赖）；
- 组内标准化：A_i = (r_i − mean(r_G)) / (std(r_G)+ε)，GRPO 标准做法。

### 10.3 训练配置

| 项 | 值 |
|---|---|
| 框架 | TRL `GRPOTrainer` + vLLM colocate 采样（peft LoRA 新 adapter） |
| prompts | 8k（四方向各 2k，从 SFT 训练分布抽 + 领域单语回译句） |
| G / temp | 8 样本，T=1.0，max_len 1536 |
| KL 锚 | β=0.04（对 OPD 合并策略，防坍缩） |
| lr / batch | 1e-6（RL 惯例再降一档），有效 batch = 8 prompts × 8 样本 |
| 步数 | 1 epoch ≈ 1,000 更新步，每 100 步 eval |

### 10.4 资源与时 long

- 2×A800：GPU0 策略（vLLM 采样 + 训练分时）、GPU1 奖励模型常驻 + ref 策略 logprob；
- 估算：rollout 8k×8≈64k 条 ≈ 2h；COMET/chrF 打分 ≈ 1.5h；训练 1k 步 ≈ 6h → **合计 ~12h**，emergency 48h 内宽裕，long_gpu 兜底；
- 显存：7B bf16(15G) + LoRA + vLLM KV(0.35 上限) + logprob 重算 ≈ GPU0 60G 内可控。

### 10.5 护栏（RL 专属）

| 风险 | 监控 | 对策 |
|---|---|---|
| reward hacking（流畅不忠实） | chrF 奖励分量 vs COMET 分量剪刀差 | 剪刀差 >0.1σ → 加大 chrF 权重 |
| 熵坍缩 | 采样熵 <3.5 nat/token | β×2 或停止 |
| 长度爆炸/重复 | len_ratio、4-gram 重复率 | 惩罚项已有 + eval 门 |
| 灾难性漂移 | KL(π‖π_ref) > 0.2 | 回滚 checkpoint |

### 10.6 对比实验的评测设计（主实验交付物）

- **三方对比**：SFT baseline / SFT+OPD / SFT+GRPO，同一 held-out（ALT test + FLORES+ devtest）、同一推理设置（greedy + T=0.7 各跑一遍）；
- 主指标：四方向 COMET；副指标：chrF++、术语命中率、格式完整率、长度比；
- 报告维度：质量分 × 方向 × 域（tech/intl/finance）× 训练成本（GPU·h）——回答"每 GPU 小时买到多少 COMET 提升"；
- 显著性：配对 bootstrap（1000 次，p<0.05）；
- 附加观察：GRPO 分支是否出现超越 32B 教师的方向（RL 上限证据）、OPD 分支是否在低资源方向（my）更稳（教师弱时的韧性）；
- 人工抽检 100 条/方向/分支（沿用 README 质量边界要求），报告写明自动指标局限。

## 8. 代码落点（对接现有仓库）

```
src/translation_agent/distill.py      # 采样/教师前向/KD 损失（peft + 自研 chunk 循环，或 TRL GKD）
scripts/opd_sbatch.sh                 # 2 卡作业脚本：复用 SLURM_SUBMIT_DIR/anaconda3/pins 修复
scripts/run_opd.sh                    # 环境变量驱动的入口（对齐 run_qwen_hpc.sh 风格）
docs/opd_post_training_design.md      # 本文档
```

魔搭上传：现有 overnight_upload 流水线按 `runs/*/final_adapter` 通配，OPD 产物目录命名 `runs/opd-r1/final_adapter` 即自动被下一轮流水线捕获（注意手动改仓库名或先传 SFT 版）。

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| 32B 教师对缅文弱 | t2s 预检 + 分方向教师（§2） |
| vllm 与训练同卡显存冲突 | 双卡布局；单卡时 vLLM `gpu_memory_utilization=0.35` 采样完释放再训练 |
| KL 对齐导致格式漂移（markdown 结构破坏） | 保留 β·NLL + ingestion 校验器进 eval；格式完整率入停机条件 |
| emergency 队列 2 卡难排 | 降级单卡 14B；或 long_gpu（14 天）长跑 |
| 学生熵坍缩 | 熵监控 + 自动降 α |
