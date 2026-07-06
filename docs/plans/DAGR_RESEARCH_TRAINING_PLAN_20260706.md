# DAGR 研究训练计划

日期：2026-07-06

本文档用于在启动下一轮训练前，统一当前 DAGR 实验状态、
checkpoint 含义、Ray 训练安排、指标记录方式和后续论文方向。

## 1. 当前 checkpoint 状态

### 1.1 官方 checkpoint：`dagr_s_50.pth`

DAGR 官方仓库提供的 `data/dagr_s_50.pth` 主要用于 DSEC-DET 评测。
官方 README 中的典型评测方式是：

- `scripts/run_test.py`
- `config/dagr-s-dsec.yaml`
- `--use_image`
- `--img_net resnet50`
- `--checkpoint data/dagr_s_50.pth`
- `--batch_size 8`

这个 checkpoint 的含义：

- 它是官方 DAGR-S 预训练权重。
- 它对应论文式的 image + event 设置。
- 它适合用来先验证我们的 DSEC-DET 数据准备、图像/事件对齐、
  `events_2x.h5` 和评测代码是否正常。
- 它不是我们前面 Ray 短跑训练生成的权重。

### 1.2 我们之前训练得到的 detector checkpoint

之前 Ray pipeline 生成了三类 detector 权重：

| 模式 | checkpoint 角色 | 训练条件 |
|---|---|---|
| `none` | 不做事件筛选的 baseline | 从头训练，event-only，短跑 |
| `latest_k5000` | 简单保留最新 K 个事件 | 从头训练，event-only，短跑 |
| `adaptive_k5000` | 使用 Stage1 sampler 的 detector | detector 从头训练；Stage1 sampler 只负责事件筛选 |

关键结论：

- 这些 detector checkpoint 没有从 `dagr_s_50.pth` 初始化。
- Ray wrapper 没有传入 `--checkpoint`、`--resume_checkpoint`、
  `--img_net_checkpoint`、`--use_image` 或 `--img_net resnet50`。
- 所以它们不是官方 DAGR 权重的微调结果，而是当前修改代码路径下
  从头训练出来的短跑实验产物。
- `adaptive` 运行中加载过 Stage1 sampler checkpoint：
  `/home/userroot/training_outputs/dagr-stage1/20260426_seed_highbg_5000step/stage1_seed45_k5000_lr3e-4_bg1.00/sampler_stage1_final.pth`
- 这个 Stage1 checkpoint 只用于 DAGR 图构建前的事件选择，
  不是官方 detector checkpoint。

上一轮短跑结果如下：

| 模式 | 训练步数 | batch | 验证子集 | samples/sec | dry-run best mAP |
|---|---:|---:|---:|---:|---:|
| `none` | 200 | 8 | 100 batches | 5.5810 | 0.0000107163 |
| `latest_k5000` | 200 | 8 | 100 batches | 21.0996 | 0.0000111393 |
| `adaptive_k5000` | 200 | 8 | 100 batches | 19.6664 | 0.0000134504 |

这组结果只能作为 smoke test：

- Ray 调度可以正常工作。
- `event_selection=none/latest/adaptive` 三条链路都能跑通。
- adaptive sampler checkpoint 能正常加载。
- 事件筛选显著提升训练吞吐。
- 当前 mAP 数值太早、太小，不能作为正式论文指标。

## 2. 研究目标

建议把当前研究主线定义为：

> 面向 DAGR 的 budget-aware adaptive event selection，
> 在尽量保持检测精度的同时降低图构建和图网络计算成本。

核心假设：

- DAGR 在使用全部事件时，图构建和异步图处理成本较高。
- 如果能学习一个事件选择器，只保留物体相关、时序上有信息量的事件，
  就有机会降低计算量。
- 如果 adaptive selector 在相同 event budget 下比 `latest-K`、
  `random-K`、`uniform-K` 更好，就具备论文贡献点。

Ray 的作用：

- Ray 不是算法创新点。
- Ray 是实验基础设施，用于多 GPU 并行跑多个独立实验。
- 它可以统一实验矩阵、GPU 资源分配、日志、summary 和 checkpoint 路径。

## 3. 实验阶段

### Phase A：官方 checkpoint 评测 sanity check

目标：

- 先用 `dagr_s_50.pth` 跑通官方 DAGR 评测路径。
- 验证 DSEC-DET 数据、图像、事件、timestamp 和 `events_2x.h5` 没有问题。

运行条件：

- 只做 evaluation，不训练。
- 使用 `--use_image --img_net resnet50`。
- 使用 `--checkpoint data/dagr_s_50.pth`。
- 不使用 adaptive sampler。

预期产出：

- 官方 checkpoint 在当前 P40/DSEC-DET 环境下的评测结果。
- runtime、显存、日志和输出路径。
- 一个本地 report，作为后续所有实验的基准锚点。

建议远端输出：

```text
/home/userroot/training_outputs/dagr-official-eval/
```

建议本地归档：

```text
/home/lz/Documents/Events-Perception/dagr/artifacts/dagr-official-eval/
/home/lz/Documents/Events-Perception/dagr/docs/reports/
```

为什么先做这个：

- 如果官方 checkpoint 都跑不通，后面训练结果无法解释。
- 它能把数据准备问题和模型修改问题分开。
- 后续论文对比 DAGR 时，必须先有官方路径的可复现实验记录。

### Phase B：12 组 controlled short-run 对照

目标：

- 检查 adaptive 方向是否在多 seed 下仍然有信号。
- 这是正式长训练前的筛选实验。

实验矩阵：

| 因素 | 取值 |
|---|---|
| event selection | `none`, `latest`, `adaptive` |
| seeds | `42`, `43`, `44`, `45` |
| batch size | `8` |
| event budget | `5000` |
| train steps | `1000` |
| validation | `dry_run_val_steps=200` |

共 12 个 trial。

Ray 资源安排：

- P40 主机有 4 张 GPU。
- 每个 Ray trial 占用 1 张 GPU。
- 12 个 trial 分批跑完。
- 这不是单模型多卡 DDP，而是多实验并行调度。

建议命令，待确认后再执行：

```bash
cd /home/userroot/dagr-ray
/home/userroot/miniconda3/bin/python run_dagr_detector_ray_pipeline.py \
  --run_name 20260706_none_latest_adaptive_b8_seed42_45_1000step_val200 \
  --exp_prefix compare_b8 \
  --modes none,latest,adaptive \
  --seeds 42,43,44,45 \
  --batch_size 8 \
  --event_budget 5000 \
  --max_train_steps 1000 \
  --dry_run_val_steps 200
```

判断标准：

- 如果 adaptive 在多 seed 下稳定优于 latest-K，进入 Phase C。
- 如果 adaptive 精度接近 latest-K 但速度更慢，需要检查 sampler 成本。
- 如果 adaptive 明显不如 latest-K，先改 Stage1 loss，而不是直接长训练。

### Phase C：event budget sweep

目标：

- 建立 speed-accuracy Pareto 曲线。

候选矩阵：

| 方法 | event budget |
|---|---|
| `latest` | 2000, 5000, 10000, 20000 |
| `adaptive` | 2000, 5000, 10000, 20000 |
| `none` | 不限制事件，baseline |
| `random` / `uniform` | 如果实现，则同样跑各预算 |

需要记录的指标：

- validation mAP
- samples/sec
- steps/sec
- wall-clock time
- GPU memory
- selected events / original events
- graph node count
- GT box 内事件保留率
- 失败 case 可视化

论文需要的理想结果：

- adaptive 在同样 budget 下比 latest/random/uniform mAP 更高；或
- adaptive 在接近 mAP 下显著减少事件量和训练/推理时间；或
- adaptive 在频率变化、事件密度变化时更稳。

### Phase D：正式 metric run

目标：

- 产出可以进入论文表格的正式指标。

候选正式 run：

| run | 作用 |
|---|---|
| 官方 `dagr_s_50.pth` eval | 官方参考 |
| `none` from scratch | 我们代码下的本地 baseline |
| `latest_k5000` | 简单事件预算 baseline |
| best adaptive budget | 我们的方法 |
| adaptive + candidate window | 低成本 adaptive 变体 |

关键问题：

- 最终和官方 DAGR 对比时，是否必须包含 image + event 模式。

建议：

- event-only 结果作为 ablation。
- 如果要严肃对比原论文 DAGR，最终需要补 image + event 设置，
  因为官方评测使用 `--use_image --img_net resnet50`。

### Phase E：Stage1 selector 和 loss 改进

目标：

- 在确认 adaptive 有基本信号后，再改 Stage1 选择器。

当前 Stage1 的作用：

- 在 DAGR 图构建前对事件打分。
- 按预算保留一部分事件。
- detector 对照实验中 Stage1 是 frozen 的。

值得尝试的改进：

1. object coverage loss
   - 鼓励选择 GT box 内或附近事件。
   - 降低背景事件选择比例。

2. budget regularization
   - 训练时随机 K，让 selector 支持多个预算。
   - 避免只适配 `K=5000`。

3. temporal diversity loss
   - 避免退化成只选最新事件。
   - 保留时间窗口中的运动线索。

4. detector-aware distillation
   - 用 detector confidence 或 gradient 作为弱监督。
   - 这个复杂度较高，建议放在 Phase E 后段。

关于 SNN、SSM、Transformer 替换：

- 可以作为后续扩展方向，但不建议现在立刻替换 DAGR 主干。
- 现在先证明 adaptive event selection 对 DAGR 有价值。
- 如果同时替换 GNN 主干，会让变量过多，难以判断贡献来自哪里。

## 4. 路径和命名规范

远端数据集：

```text
/home/userroot/datasets/dsec-det
```

远端 DAGR 仓库：

```text
/home/userroot/dagr-adaptive-sampling
```

远端 Ray 脚本：

```text
/home/userroot/dagr-ray
```

远端训练输出：

```text
/home/userroot/training_outputs/dagr-detector-ray-pipeline/
```

建议 run name：

```text
YYYYMMDD_<modes>_b<BATCH>_seed<SEED_RANGE>_<steps>step_val<VAL_STEPS>
```

示例：

```text
20260706_none_latest_adaptive_b8_seed42_45_1000step_val200
```

本地归档：

```text
/home/lz/Documents/Events-Perception/dagr/artifacts/<run_name>/
/home/lz/Documents/Events-Perception/dagr/docs/reports/<report_name>.md
```

每次正式 run 必须保留：

- `pipeline_config.json`
- `summary.json`
- `summary.md`
- 每个 trial 的 stdout log
- best checkpoint 和 last checkpoint
- 精确命令
- 当时的 git diff 或 commit hash

## 5. 推荐下一步

建议顺序：

1. 先跑 Phase A：官方 `dagr_s_50.pth` evaluation。
2. 把官方 eval 的日志、metric、路径归档到 DAGR repo。
3. 再跑 Phase B：12 组 seed sweep。
4. 归档并总结 Phase B。
5. 决定 adaptive 是否进入 budget sweep。
6. 如果 adaptive 信号稳定，进入 Phase C。
7. 如果 adaptive 不稳定，先改 Stage1 loss，再训练 detector。

暂时不建议直接长训练，直到满足这些条件：

- 官方 checkpoint eval 已确认。
- 12 组 seed sweep 已审核。
- 输出命名和归档路径已确认。
- 决定最终是否补 image + event 设置。

## 6. 主要风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| event-only 不能直接对比官方 DAGR | 论文 baseline 不严谨 | 明确区分 ablation 和 official reproduction |
| adaptive 只接近 latest-K | 创新点较弱 | 增加 coverage、temporal diversity、budget robustness 分析 |
| dry-run mAP 噪声大 | 容易误判方向 | 多 seed、更长验证，再做长训练 |
| P40 显存限制 image + event | batch 受限 | batch size 8，必要时 gradient accumulation |
| 数据对齐问题 | metric 无效 | 先跑官方 checkpoint sanity check |

## 7. 需要确认的问题

启动训练前建议先确认：

1. 下一步是否先执行 Phase A 官方 checkpoint evaluation？
2. Phase B 是否保持 event-only，还是加入 image + event？
3. `event_budget=5000` 是否作为下一轮主预算？
4. Phase C 前是否先补 `random` 和 `uniform` 两个 baseline？
