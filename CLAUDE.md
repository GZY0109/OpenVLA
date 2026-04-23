# CLAUDE.md — OpenVLA 项目

> Claude Code 会话启动时自动读取本文件。新开会话直接说任务，无需重复介绍背景。

---

## 项目基本信息

- **项目路径**：`/home/users/ntu/zuyu001/OPENVLA`
- **Conda 环境**：`vla_env`（Python 3.10, torch 2.7.1+cu118）
- **任务**：OpenVLA-7B 推理部署 + LIBERO 零样本评估 + YOLOv8 前置感知集成

每次开终端先运行：
```bash
conda activate vla_env
cd /home/users/ntu/zuyu001/OPENVLA
```

---

## 目录结构

```
/home/users/ntu/zuyu001/OPENVLA/
├── CLAUDE.md                        # 本文件，勿删
├── README.md                        # 项目介绍（GitHub 展示用）
├── .gitignore                       # 自动生成，勿改
├── interview_prep.md                # 面试准备（已 gitignore，不上传）
│
├── inference/
│   ├── run_inference.py             # 单步推理入口，当前延迟 ~320ms
│   └── yolo_preprocess.py           # YOLOv8 前置感知模块
│
├── training/
│   ├── lora_finetune.py             # LoRA 微调脚本
│   ├── dpo_align.py                 # DPO 偏好对齐脚本
│   ├── rlvr_train.py                # RLVR 强化学习训练
│   └── data_prepare.py             # 数据预处理与格式转换
│
├── optimization/
│   ├── quantize_int8.py             # INT8 量化脚本
│   ├── pruning.py                   # 结构化剪枝
│   └── export_onnx.py              # ONNX 导出 + TensorRT
│
├── eval/
│   └── libero_eval.py               # LIBERO benchmark 评估脚本
│
├── configs/
│   └── default.yaml                 # 模型路径、设备配置（路径改动先查这里）
│
├── scripts/
│   ├── init_repo.sh                 # 首次 git 初始化（只跑一次）
│   └── log_run.py                   # 自动日志辅助（可选）
│
└── logs/
    └── YYYYMMDD_实验名称.md          # 手动记录，每次实验写一份
```

---

## 文献参考基线（来源：OpenVLA 原论文，尚未本地复现）

| 指标 | 数值 | 备注 |
|------|------|------|
| 单步推理延迟 | ~320ms | 瓶颈在 VLM 解码阶段 |
| LIBERO Object 成功率 | ~41% | 零样本 |
| LIBERO Goal 成功率 | ~28% | 零样本 |
| 定位模糊失败占比 | ~55% | 主要失败原因 |
| YOLO 集成后 Goal 成功率 | ~39% | 提升约 +11pp |

---

## 已知问题（优先关注）

1. **推理延迟 320ms/step**：瓶颈在 VLM 解码，待量化/批处理优化
2. **指令格式敏感性**：中文指令需标准化预处理
3. **YOLO-VLA 特征对齐**：特征空间未完全对齐，存在 Sim-to-Real gap

---

## 下一步任务（按岗位要求分三条主线）

### 主线一：指令微调与对齐（SFT → DPO）
- [ ] LIBERO 数据集预处理：构建 instruction-image-action 三元组
- [ ] LoRA 微调训练脚本（rank=16/32 对比实验）
- [ ] 微调前后 LIBERO 成功率对比评估
- [ ] DPO 偏好对齐：用 rollout 成功/失败轨迹构造偏好数据
- [ ] DPO 训练 + 评估闭环

### 主线二：RLHF / RLVR 强化学习优化
- [ ] Reward Model 设计：基于 LIBERO task success 的 verifiable reward
- [ ] RLVR 训练流程搭建（PPO / GRPO）
- [ ] RL 微调前后性能对比（成功率 + 轨迹质量）
- [ ] 训练稳定性分析（reward curve、KL divergence）

### 主线三：后训练优化与部署
- [ ] INT8 量化推理加速测试（延迟 & 精度对比）
- [ ] 结构化剪枝实验（attention head / FFN pruning）
- [ ] ONNX 导出 + TensorRT 推理加速
- [ ] 真机部署接口适配
- [ ] 端侧部署验证（延迟、内存占用、成功率）

### 辅助任务
- [ ] Rollout 视频整理归档
- [ ] 各阶段实验日志整理

---

## 日志规范（每次实验手动写一份）

在 `logs/` 下新建文件，命名格式：`YYYYMMDD_实验描述.md`

模板：
```markdown
# 实验日志 — YYYY-MM-DD

## 任务描述
（本次做了什么）

## 运行命令
\`\`\`bash
（完整命令）
\`\`\`

## 结果
（数字、截图描述、现象）

## 问题 & 下一步
（遇到什么问题，下次怎么改）
```

---

## Git 提交规范

```bash
# 每次实验后提交一次，留 GitHub 痕迹
git add -A
git commit -m "exp: 简短描述本次实验内容"
git push origin main

# commit 前缀约定
# exp:  实验记录（最常用）
# feat: 新增功能或脚本
# fix:  bug 修复
# docs: README 或日志更新
```

---

## 常用命令

```bash
# 推理测试
python inference/run_inference.py --test

# LIBERO 评估
python eval/libero_eval.py --task object --episodes 20
python eval/libero_eval.py --task goal --episodes 20

# GPU 状态
nvidia-smi -l 1
```

---

## Claude Code 注意事项

- 修改 `inference/` 核心代码前，确认 `logs/` 有当前基线记录
- 涉及路径配置，优先读 `configs/default.yaml`
- 模型权重文件（*.pt *.bin *.pth）已在 `.gitignore` 中排除，不会上传
- 回答用**中文**，代码注释用**英文**