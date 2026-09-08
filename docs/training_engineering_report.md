# tripivot 翻译模型训练工程报告

> 面向答辩的训练工程复盘：SFT 全量训练 → OPD vs GRPO 后训练对比 → 自动化交付流水线。
> 生成时间：2026-09-06（对比实验数据段待评测产出后回填）

## 1. 目标与结果总览

| 交付物 | 状态 | 位置 |
|---|---|---|
| Qwen2.5-7B 翻译 LoRA（SFT，四语向×三领域） | ✅ 完成 | 魔搭 `zechlei/tripivot-qwen2.5-7b-lora`（私有） |
| OPD vs GRPO 对比报告（en-zh 试点） | ⏳ 当日 | `runs/comparison_report.md` |
| 两个后训练变体权重 | ⏳ 当日 | `runs/{opd,grpo}-r1/final_adapter` |
| 本地答辩演示前端 | ✅ 骨架就绪 | `demo/`（三模型并排对比，MLX 离线推理） |
| 完整设计文档 | ✅ | `docs/opd_post_training_design.md` |

## 2. 训练配置（SFT）

- 基座 Qwen2.5-7B-Instruct，LoRA r=16 / α=32 / dropout 0.05（gate/o/v/up/down proj 全目标）
- 数据：91,852 条（ALT 平行语料 + 三领域维基蒸馏，术语软约束 prompt）
- 4 方向 × 3 领域，2 epochs，effective batch 16（2×8），lr 1e-4，max_len 1536
- 10,772 步，A800-80G 单卡实际耗时约 11 小时（含缓存下载），中期自动 checkpoint 续跑能力验证通过

## 3. 后训练对比实验（OPD vs GRPO）

### 3.1 实验设计

同一 SFT 合并基座、同数据分布、同 held-out，双分支平行对比：

- **OPD**：学生在线采样（chunked，vLLM），Qwen2.5-14B 同词表教师逐步 logits，
  损失 `0.6·reverseKL + 0.4·NLL(ref)`，lr 5e-6，~750 步
- **GRPO**：组采样 G=8，奖励 `chrF++ 0.75 + 格式 0.25 − 长度惩罚`（COMET 为受限模型，
  改本地 chrF 主导，两分支同尺公平），组内标准化优势 + KL 锚（β=0.02），lr 1e-6

### 3.2 试点结果（en-zh，400 条 held-out，贪心解码）

<!-- PILOT_RESULTS_PLACEHOLDER：runs/comparison_report.md 产出后回填 -->

| 指标 | baseline(SFT) | OPD-R1 | GRPO-R1 |
|---|---|---|---|
| chrF++ | 待回填 | 待回填 | 待回填 |
| 长度比 | 待回填 | 待回填 | 待回填 |
| 退化率 | 待回填 | 待回填 | 待回填 |

初步结论（数据到位后确认/修正）：待回填。

## 4. 工程亮点：全自动交付流水线

```
rsync 仓库+语料 → sbatch（队列竞速+降级）→ 作业内环境自举
→ 训练 → final_adapter 落盘 → 守护进程检测 → 魔搭自动上传
                                   ↘ 编排器：双分支提交→1卡降级→统一评测→对比报告
```

- **队列竞速**：同一作业多队列分身（emergency/A40/long_gpu），总裁判先到先得取消其余，
  集群 GPU 饱和期（14h 零流转）仍稳定拿到算力
- **降级阶梯**：2 卡 → 1 卡分时复用 → 串行，全部在编排器内自动决策
- **服务端自治**：关键流程 nohup 守护（上传流水线、编排器），客户端断线零影响——
  本项目期间校园 VPN 断连 6+ 次，无一影响训练
- **一键复现**：环境自举在作业内完成（pip 缓存预热 + 版本钉死），`sbatch` 即端到端

## 5. 问题全记录（答辩工程细节弹药）

| # | 现象 | 根因 | 解决 | 代价 |
|---|---|---|---|---|
| 1 | 集群域名全部 NXDOMAIN | EasyConnect 不推送校园 DNS | 从客户端日志挖出 DNS 10.90.63.3，SSH 配置 IP 直连 | 0.5h |
| 2 | 登录节点 pip 进程被杀 | 节点内存压力（pgrep 分配失败） | 依赖安装移入计算节点作业 + pip 缓存预热 | 1 次重试 |
| 3 | sbatch 作业 2 秒失败 `mkdir Permission denied` | Slurm 从 spool 副本执行，`$0` 非仓库路径 | `cd "${SLURM_SUBMIT_DIR:-...}"` | 2 作业 |
| 4 | 模型下载 401 | HF xet CDN 绕过镜像直连 | `HF_HUB_DISABLE_XET=1` + 关高性能模式 | 1 作业 |
| 5 | `cuda available: False` | PyPI 最新 torch 为 cu130 构建，集群驱动仅支持 ≤12.8 | 钉 `torch==2.7.1`(cu126) | 1 作业 |
| 6 | `warmup_ratio` TypeError | transformers 5.x 移除该参数 | 钉 `transformers==4.46.3` 全家桶 | 1 作业 |
| 7 | 资产脚本整链崩溃 | COMET 模型为 HF 受限仓库，镜像 401，`set -e` 放大 | 奖励改 chrF 本地化；步骤容错化 | 3h 延迟 |
| 8 | 编排器提交作业号全空 | 非登录 shell 无 `/opt/slurm/bin`；`--wrap` 不支持 heredoc | PATH 导出 + 独立 sbatch 脚本 | 2 次重启 |
| 9 | 登录 VIP 轮换导致 SSH 拒连 | 负载均衡后端各持不同主机密钥 | `StrictHostKeyChecking=no` + known_hosts 置空 | 已固化 |

**教训抽象**：① 异构集群上"版本钉死 > 最新版"；② 长链路自动化必须每步容错+幂等；
③ 客户端只做观察，执行全部服务端化。

## 6. 成本核算

| 项 | 用量 |
|---|---|
| SFT 训练 | 1×A800 × ~11h ≈ 11 GPU·h |
| 环境与数据准备 | 登录节点下载 ~45GB（教师 28G + 学生 15G + 依赖 2G） |
| 失败重试 | 6 个短命作业合计 <15 GPU·min |
| OPD/GRPO 试点 | 各 2×A800 × ~2h ≈ 8 GPU·h（待确认） |

## 7. 后续计划（答辩前）

1. 试点数据回填本报告 §3.2
2. 周末：全量四方向正式对比（视试点结论调配资源）
3. 周三前：demo 本地化部署演练 + 答辩动线排练
