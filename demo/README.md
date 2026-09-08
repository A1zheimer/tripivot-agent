# 答辩演示前端

三模型并排对比演示页：SFT 基线 vs OPD（教师蒸馏） vs GRPO（指标奖励 RL）。

## 运行（答辩用，全程本地、无需网络）

```bash
# 一次性准备（明早对比权重出来后执行，~10 分钟）
bash demo/prepare_demo.sh        # 合并三个变体 -> MLX 4bit + 生成缓存兜底

# 启动（Mac 本地）
pip install fastapi uvicorn mlx-lm
python demo/server.py --port 7860
# 打开 http://127.0.0.1:7860
```

## 双保险设计

- **mlx 后端**：三个变体常驻内存（4bit 量化，共约 15GB 统一内存），现场实时推理
- **cached 后端**：启动时 mlx 不可用自动切换；演示页顶部徽章会显示当前模式。
  `cache.json` 由 `prepare_demo.sh` 预生成（评测样本 × 三个变体的真实输出），
  答辩现场即使模型加载失败，页面依旧可完整走完流程

## 目录

```
demo/
├── server.py           # FastAPI：/api/translate /api/compare /api/health
├── static/index.html   # 单文件前端（零构建、零 CDN 依赖）
├── prepare_demo.sh     # 合并变体 + mlx-lm convert + 生成 cache.json
├── models/{baseline,opd,grpo}/   # 运行时生成
└── cache.json                     # 运行时生成
```

## 答辩演示建议动线

1. 先讲数据管线（OPUS/ALT 语料 + 三域蒸馏 + 术语库）——README 截图即可
2. 演示页输入科技/财经示例 → 三列对比，讲两种后训练的差异
3. 切换 缅→中 方向演示低资源场景
4. 最后放 `runs/comparison_report.md` 的量化表收尾
