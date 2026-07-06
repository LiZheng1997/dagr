# DAGR Ray Stage 1 Experiments

This records the first Ray-managed DAGR adaptive-sampler Stage 1 smoke tests on
the P40 host.

## Environment

```text
Host: userroot@192.168.103.2
Ray dashboard: http://192.168.103.2:8265
DAGR code: /home/userroot/dagr-adaptive-sampling
Ray wrapper: /home/userroot/dagr-ray/run_adaptive_stage1_multi.py
Dataset: /home/userroot/datasets/dsec-det
Dataset size: 225G
events_2x.h5 files: 48
train timestamps: 47
Ray resources before run: 12 CPU, 4 GPU, 4 P40 all idle
```

## Single-Process Smoke

Command shape:

```bash
cd /home/userroot/dagr-adaptive-sampling
PYTHONPATH=/home/userroot/dagr-adaptive-sampling/src:/home/userroot/dagr-adaptive-sampling/libs/dsec-det/src:/home/userroot/dagr-adaptive-sampling/libs/YOLOX \
  /home/userroot/anaconda3/envs/dagr/bin/python \
  scripts/train_adaptive_sampler_stage1.py \
  --dataset_directory /home/userroot/datasets/dsec-det \
  --output_dir /home/userroot/training_outputs/stage1_single_smoke_20260425_1647 \
  --split train --epochs 1 --batch_size 1 --num_workers 0 \
  --max_steps 1 --log_every 1
```

Result:

```text
device=cuda
train samples=47996
sampler params=180961
step=1 loss=0.9262 lr=3.00e-04
```

After the event-only sampler update, the checkpoint smoke was rerun:

```text
Output: /home/userroot/training_outputs/stage1_event_only_ckpt_smoke_20260425_1901
device=cuda
train samples=47996
sampler params=38209
step=1 loss=0.9238 lr=3.00e-04
checkpoint: sampler_stage1_final.pth, 157613 bytes
loaded checkpoint: epoch=0, global_step=1, use_image=False
```

## Ray 4-GPU Smoke

Submitted through Ray Jobs:

```bash
cd /home/userroot/dagr-ray
ray job submit \
  --address http://192.168.103.2:8265 \
  --working-dir /home/userroot/dagr-ray \
  -- python run_adaptive_stage1_multi.py
```

Ray cluster resources reported by the job:

```text
CPU=12
GPU=4
P40=4
```

Each Ray task requested:

```python
@ray.remote(num_cpus=2, num_gpus=1, resources={"P40": 1})
```

Results for `max_steps=3`:

| Experiment | GPU | Task resources | Elapsed | Last loss |
|------------|-----|----------------|---------|-----------|
| `stage1_k5000_lr1e-4_bg0.05` | 0 | CPU=2, GPU=1, P40=1 | 8.525s | 0.9927 |
| `stage1_k5000_lr3e-4_bg0.10` | 1 | CPU=2, GPU=1, P40=1 | 8.763s | 0.7554 |
| `stage1_k10000_lr1e-4_bg0.05` | 2 | CPU=2, GPU=1, P40=1 | 8.783s | 0.5949 |
| `stage1_k10000_lr3e-4_bg0.10` | 3 | CPU=2, GPU=1, P40=1 | 8.737s | 0.4874 |

Results for `max_steps=20`:

| Experiment | GPU | Task resources | Elapsed | Last loss |
|------------|-----|----------------|---------|-----------|
| `stage1_k5000_lr1e-4_bg0.05` | 0 | CPU=2, GPU=1, P40=1 | 8.697s | 0.5888 |
| `stage1_k5000_lr3e-4_bg0.10` | 1 | CPU=2, GPU=1, P40=1 | 9.164s | 0.3153 |
| `stage1_k10000_lr1e-4_bg0.05` | 2 | CPU=2, GPU=1, P40=1 | 8.378s | 0.5340 |
| `stage1_k10000_lr3e-4_bg0.10` | 3 | CPU=2, GPU=1, P40=1 | 8.566s | 0.3920 |

Artifacts:

```text
/home/userroot/training_outputs/ray_stage1_smoke/summary.json
/home/userroot/training_outputs/ray_stage1_20step_20260425_1650/summary.json
```

## What This Demonstrates

Ray acted as the GPU resource manager:

- All four experiments ran under one Ray job submission.
- Each task reserved exactly one logical GPU and one custom `P40` resource.
- Ray assigned separate visible devices: `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- The 20-step worker runtimes sum to about 34.8 seconds, while the parallel
  critical path was about 9.2 seconds, giving about 3.8x worker-level
  parallelism for four independent experiments.

The next useful run should increase `max_steps` to a few hundred and add
checkpoint saving for early-stop runs before doing visual score-map inspection.

## Ray 4-GPU Event-Only 500-Step Run

Submitted through Ray Jobs:

```bash
cd /home/userroot/dagr-ray
ray job submit \
  --address http://192.168.103.2:8265 \
  --no-wait \
  --working-dir /home/userroot/dagr-ray \
  -- python run_adaptive_stage1_multi.py \
       --max_steps 500 \
       --out_root /home/userroot/training_outputs/ray_stage1_event_only_500step_20260425_1902
```

Ray job:

```text
raysubmit_GaD8aWrf4FyDJpeu: SUCCEEDED
```

Runtime resource snapshot while the job was running:

```text
Total Usage:
  8.0/12.0 CPU
  4.0/4.0 GPU
  4.0/4.0 P40

GPU memory/utilization sample:
  GPU0: 369 MiB, 5%
  GPU1: 369 MiB, 5%
  GPU2: 369 MiB, 5%
  GPU3: 389 MiB, 5%
```

Each worker wrote its assigned GPU into `train_stdout.log`:

```text
stage1_k5000_lr1e-4_bg0.05:   CUDA_VISIBLE_DEVICES=0
stage1_k5000_lr3e-4_bg0.10:   CUDA_VISIBLE_DEVICES=1
stage1_k10000_lr1e-4_bg0.05:  CUDA_VISIBLE_DEVICES=2
stage1_k10000_lr3e-4_bg0.10:  CUDA_VISIBLE_DEVICES=3
```

Results, sorted by last-100-step mean loss:

| Experiment | GPU | Elapsed | First loss | Last loss | Mean first 50 | Mean last 100 |
|------------|-----|---------|------------|-----------|---------------|---------------|
| `stage1_k5000_lr3e-4_bg0.10` | 1 | 21.149s | 0.7824 | 0.3237 | 0.5927 | 0.3266 |
| `stage1_k10000_lr3e-4_bg0.10` | 3 | 21.236s | 0.9815 | 0.3245 | 0.6708 | 0.3433 |
| `stage1_k10000_lr1e-4_bg0.05` | 2 | 20.361s | 0.7928 | 0.4260 | 0.7380 | 0.4136 |
| `stage1_k5000_lr1e-4_bg0.05` | 0 | 21.299s | 0.9392 | 0.6485 | 1.0474 | 0.4813 |

Best checkpoint by mean last-100-step loss:

```text
/home/userroot/training_outputs/ray_stage1_event_only_500step_20260425_1902/stage1_k5000_lr3e-4_bg0.10/sampler_stage1_final.pth
```

Validation score-map visualization:

```text
/home/userroot/training_outputs/ray_stage1_event_only_500step_20260425_1902/viz_best_stage1_k5000_lr3e-4_bg0.10
```

Generated frames:

```text
frame_000_idx0.png
frame_001_idx700.png
frame_002_idx1400.png
frame_003_idx2100.png
frame_004_idx2800.png
frame_005_idx3500.png
```

Score-map probability ranges on the six visualized validation samples were
roughly `0.53..1.00`. This means the short run already learned a strong positive
prior, but it still needs qualitative inspection and longer training to know
whether the heatmap is spatially selective enough around objects.

Validation selectivity check on 200 sampled val frames:

```text
frames_with_box: 200 / 200
score mean inside bbox:  0.851399
score mean outside bbox: 0.774893
inside - outside:        0.076506
mean frame contrast:     0.051052
top-10% grid precision:  0.107404
```

Interpretation: the 500-step model is learning a positive prior but is not yet
selective enough. The next sweep should increase `bg_weight` to penalize
all-high heatmaps before integrating the sampler into the DAGR detector.

## Ray 4-GPU Event-Only High-Background Sweep

Submitted through Ray Jobs:

```bash
cd /home/userroot/dagr-ray
ray job submit \
  --address http://192.168.103.2:8265 \
  --no-wait \
  --working-dir /home/userroot/dagr-ray \
  -- python run_adaptive_stage1_multi.py \
       --sweep high_bg \
       --max_steps 500 \
       --out_root /home/userroot/training_outputs/ray_stage1_event_only_highbg_500step_20260425_1904
```

Ray job:

```text
raysubmit_unMtBcx2UL5H5Ypt: SUCCEEDED
```

Runtime resource snapshot while the job was running:

```text
Total Usage:
  8.0/12.0 CPU
  4.0/4.0 GPU
  4.0/4.0 P40
```

Validation selectivity check on 200 sampled val frames. The standardized
evaluator added later supersedes the original ad-hoc selectivity numbers.

| Experiment | Last loss | Inside mean | Outside mean | Diff | Top-5% precision | Top-10% precision |
|------------|-----------|-------------|--------------|------|------------------|-------------------|
| `stage1_k10000_lr3e-4_bg0.30` | 0.6408 | 0.731621 | 0.642380 | 0.089241 | 0.109231 | 0.091442 |
| `stage1_k5000_lr3e-4_bg1.00` | 1.1020 | 0.495644 | 0.406816 | 0.088828 | 0.108077 | 0.090433 |
| `stage1_k5000_lr3e-4_bg0.30` | 0.7211 | 0.693295 | 0.612224 | 0.081070 | 0.102212 | 0.086010 |
| `stage1_k10000_lr3e-4_bg1.00` | 1.0939 | 0.511627 | 0.402210 | 0.109417 | 0.098654 | 0.085048 |

Best checkpoint by inside/outside score separation:

```text
/home/userroot/training_outputs/ray_stage1_event_only_highbg_500step_20260425_1904/stage1_k10000_lr3e-4_bg0.30/sampler_stage1_final.pth
```

Validation score-map visualization:

```text
Remote:
/home/userroot/training_outputs/ray_stage1_event_only_highbg_500step_20260425_1904/viz_best_stage1_k10000_lr3e-4_bg0.30

Local copy:
/home/lz/Documents/BluetVPN/artifacts/dagr-ray-stage1-highbg-viz
```

This is a useful direction, but top-10% precision is still low. Do not
integrate into the full DAGR detector yet; first improve the Stage 1 objective
so high-score cells are more concentrated around objects.

## Ray 4-GPU Experimental Loss Sweeps

Two top-k-aware Stage 1 losses were added and tested for 500 steps:

```text
focal_topk = positive BCE + focal background BCE + top background BCE
bce_topk   = positive BCE + weighted background BCE + top background BCE
```

Both ran through Ray with the same resource pattern as before:

```text
CPU=8/12
GPU=4/4
P40=4/4
one experiment per GPU
```

### focal_topk

Output root:

```text
/home/userroot/training_outputs/dagr-stage1/20260425_focal_topk_500step
```

Ray job:

```text
raysubmit_qMj98grL7Zmne3CN: SUCCEEDED
```

Validation selectivity check on 200 sampled val frames:

| Experiment | Last loss | Inside mean | Outside mean | Diff | Top-10% precision |
|------------|-----------|-------------|--------------|------|-------------------|
| `stage1_focal_k10000_lr3e-4_bg0.50_top0.10` | 0.6782 | 0.610632 | 0.552751 | 0.057881 | 0.093798 |
| `stage1_focal_k10000_lr3e-4_bg0.30_top0.10` | 0.7123 | 0.676402 | 0.634213 | 0.042189 | 0.088221 |
| `stage1_focal_k10000_lr3e-4_bg0.30_top0.30` | 0.9002 | 0.588694 | 0.576436 | 0.012259 | 0.083413 |
| `stage1_focal_k10000_lr3e-4_bg0.50_top0.30` | 0.9334 | 0.557561 | 0.551097 | 0.006464 | 0.075337 |

### bce_topk

Output root:

```text
/home/userroot/training_outputs/dagr-stage1/20260425_bce_topk_500step
```

Ray job:

```text
raysubmit_yKujfzsfi8Wzud3V: SUCCEEDED
```

Validation selectivity check on 200 sampled val frames:

| Experiment | Last loss | Inside mean | Outside mean | Diff | Top-10% precision |
|------------|-----------|-------------|--------------|------|-------------------|
| `stage1_bcetopk_k10000_lr3e-4_bg0.50_top0.10` | 0.7439 | 0.648415 | 0.554798 | 0.093617 | 0.093029 |
| `stage1_bcetopk_k10000_lr3e-4_bg0.50_top0.05` | 0.9628 | 0.553411 | 0.460001 | 0.093410 | 0.087308 |
| `stage1_bcetopk_k10000_lr3e-4_bg0.30_top0.05` | 0.6109 | 0.643479 | 0.571347 | 0.072131 | 0.087740 |
| `stage1_bcetopk_k10000_lr3e-4_bg0.30_top0.10` | 0.9668 | 0.682296 | 0.617208 | 0.065088 | 0.090433 |

Conclusion:

```text
Current best 500-step high-bg BCE remains:
/home/userroot/training_outputs/ray_stage1_event_only_highbg_500step_20260425_1904/stage1_k10000_lr3e-4_bg0.30/sampler_stage1_final.pth

inside/outside diff: 0.089241
top-10% precision:   0.091442
```

The top-k-aware losses are useful ablations, but they should not replace
high-background weighted BCE as the active baseline yet. The next stronger
objective should change the target shape itself, for example a CenterNet-style
Gaussian object-center heatmap or a margin/ranking term between object cells and
the highest-scoring background cells.

## Ray 4-GPU Gaussian Target Sweep

Four Gaussian target variants were tested for 500 steps:

```text
output root: /home/userroot/training_outputs/dagr-stage1/20260425_gaussian_500step
Ray job: raysubmit_3CG175Ay7qUUyF6B: SUCCEEDED
```

Ray again scheduled one experiment per GPU:

```text
stage1_gaussbce_k10000_lr3e-4_bg0.30_sig0.30: GPU 0, 21.383s
stage1_gaussbce_k10000_lr3e-4_bg0.30_sig0.50: GPU 1, 20.595s
stage1_centernet_k10000_lr3e-4_sig0.30:       GPU 2, 20.942s
stage1_centernet_k10000_lr3e-4_sig0.50:       GPU 3, 19.919s
```

Validation selectivity check on the same 200 sampled val frames:

```text
Average GT grid fraction: 0.027236
```

| Experiment | Last loss | Inside mean | Outside mean | Diff | Top-5% precision | Top-10% precision |
|------------|-----------|-------------|--------------|------|------------------|-------------------|
| `stage1_gaussbce_k10000_lr3e-4_bg0.30_sig0.30` | 0.1314 | 0.234607 | 0.174869 | 0.059738 | 0.062115 | 0.059087 |
| `stage1_gaussbce_k10000_lr3e-4_bg0.30_sig0.50` | 0.0990 | 0.222175 | 0.179388 | 0.042787 | 0.057885 | 0.054663 |
| `stage1_centernet_k10000_lr3e-4_sig0.30` | 0.1028 | 0.023135 | 0.022612 | 0.000523 | 0.028558 | 0.027356 |
| `stage1_centernet_k10000_lr3e-4_sig0.50` | 0.0583 | 0.033447 | 0.033456 | -0.000009 | 0.028846 | 0.026058 |

Conclusion:

```text
Gaussian/CenterNet target did not beat the high-background BCE baseline.
The 500-step high-bg BCE baseline remains stronger:
inside/outside diff: 0.089241
top-10% precision:   0.091442
```

The useful observation is that the average GT grid fraction is only about
`2.72%`, so the current top-10% precision is still above random. The sampler is
learning a signal, but the selected region is still too diffuse.

## Ray 4-GPU Longer Baseline Sweep

A longer high-background BCE sweep was started to test whether the current best
loss simply needs more optimization steps:

```text
output root: /home/userroot/training_outputs/dagr-stage1/20260425_highbg_5000step
Ray job: raysubmit_XAC7J1bya9b7BUS5: SUCCEEDED
max_steps: 5000
resource snapshot while running: 8/12 CPU, 4/4 GPU, 4/4 P40
```

Validation selectivity check with the standardized evaluator:

| Experiment | Last loss | Inside mean | Outside mean | Diff | Top-5% precision | Top-10% precision |
|------------|-----------|-------------|--------------|------|------------------|-------------------|
| `stage1_k5000_lr3e-4_bg0.30` | 0.4827 | 0.757366 | 0.636035 | 0.121331 | 0.119423 | 0.101154 |
| `stage1_k5000_lr3e-4_bg1.00` | 0.7923 | 0.514209 | 0.342107 | 0.172101 | 0.124135 | 0.100240 |
| `stage1_k10000_lr3e-4_bg1.00` | 0.9881 | 0.563772 | 0.399452 | 0.164321 | 0.123654 | 0.096731 |
| `stage1_k10000_lr3e-4_bg0.30` | 1.1409 | 0.757475 | 0.600679 | 0.156796 | 0.117308 | 0.096298 |

Conclusion:

```text
5000 steps improved the standardized metrics over the 500-step high-bg run.
Best top-10% precision: stage1_k5000_lr3e-4_bg0.30, 0.101154
Best inside/outside diff: stage1_k5000_lr3e-4_bg1.00, 0.172101
```

## Event-Level Selection And Seed Stability

The grid metric above checks whether the dense score map is higher inside GT
boxes. A second evaluator now checks the sampler's actual hard top-K output:

```text
script: /home/userroot/dagr-adaptive-sampling/scripts/eval_adaptive_sampler_events.py
metric file: /home/userroot/training_outputs/dagr-stage1/20260425_highbg_5000step/metrics/event_selection_val200.json
```

For the original 5000-step high-background sweep on 200 validation frames:

| Experiment | Selected event hit rate | Selected/all lift | Selected bbox coverage |
|------------|-------------------------|-------------------|------------------------|
| `stage1_k5000_lr3e-4_bg1.00` | 0.131879 | 1.316011 | 0.190707 |
| `stage1_k10000_lr3e-4_bg1.00` | 0.127308 | 1.405631 | 0.310798 |
| `stage1_k10000_lr3e-4_bg0.30` | 0.119148 | 1.216507 | 0.297154 |
| `stage1_k5000_lr3e-4_bg0.30` | 0.117466 | 1.009382 | 0.136744 |

This changed the active interpretation: `bg0.30` is slightly stronger on
dense grid top-10 precision, but `bg1.00` is cleaner for actual selected
events.

A seed-stability run was then launched through Ray:

```text
output root: /home/userroot/training_outputs/dagr-stage1/20260426_seed_highbg_5000step
sweep: bg0.30/bg1.00 x seed 42/43/44/45
max_steps: 5000
Ray behavior: 8 experiments queued on 4 P40 GPUs, executed in two GPU-saturated waves
```

Mean +/- population std over the four seeds:

| Group | Grid top-10 | Grid diff | Selected hit rate | Selected/all lift | Selected bbox coverage |
|-------|-------------|-----------|-------------------|-------------------|------------------------|
| `bg0.30` | 0.095817 +/- 0.002777 | 0.145015 +/- 0.012782 | 0.122008 +/- 0.002577 | 1.147558 +/- 0.049191 | 0.182426 +/- 0.018852 |
| `bg1.00` | 0.093089 +/- 0.001838 | 0.155169 +/- 0.009054 | 0.127481 +/- 0.004242 | 1.430758 +/- 0.106427 | 0.234772 +/- 0.032431 |

Best individual checkpoints:

```text
Best grid top-10:
/home/userroot/training_outputs/dagr-stage1/20260426_seed_highbg_5000step/stage1_seed44_k5000_lr3e-4_bg0.30/sampler_stage1_final.pth

Best selected-event hit rate:
/home/userroot/training_outputs/dagr-stage1/20260426_seed_highbg_5000step/stage1_seed45_k5000_lr3e-4_bg1.00/sampler_stage1_final.pth
```

Visual inspection of the best 5000-step checkpoints shows that `bg1.00`
suppresses background more aggressively than `bg0.30`, but top-K selection is
still attracted to high-event-density edges such as signs, sky boundaries, and
large vehicle edges. The Stage 1 signal is real, but not yet sufficiently
object-covering for detector integration without another loss iteration.

## Event-Aware Loss Prototypes

Two event-aware Stage 1 loss branches were prototyped after the seed run:

```text
event_topk:
  event-level in-box BCE
  hard-background top-k penalty
  per-box max-score coverage term

hybrid_event_bce:
  dense bbox BCE
  plus weighted event_topk regularizer
```

Implementation:

```text
/home/userroot/dagr-adaptive-sampling/src/dagr/model/adaptive_sampler.py
/home/userroot/dagr-adaptive-sampling/scripts/train_adaptive_sampler_stage1.py
/home/userroot/dagr-ray/run_adaptive_stage1_multi.py
```

The first `event_topk` Ray run exposed an empty-event/empty-target autograd
case. It was fixed by returning `score_map.sum() * 0` for zero-loss frames, and
the 20-step smoke then completed.

### Pure event_topk, 500 steps

```text
output root: /home/userroot/training_outputs/dagr-stage1/20260426_eventtopk_500step
Ray result: 4/4 returncode 0
elapsed: about 22-24s per worker
```

Best results:

| Metric | Best experiment | Value |
|--------|-----------------|-------|
| Grid top-10 | `stage1_eventtopk_bg0.05_top0.10_box0.50` | 0.036202 |
| Selected event hit rate | `stage1_eventtopk_bg0.05_top0.10_box0.50` | 0.084980 |
| Selected/all lift | `stage1_eventtopk_bg0.10_top0.30_box1.00` | 2.081755 |
| Selected bbox coverage | `stage1_eventtopk_bg0.10_top0.30_box1.00` | 0.487711 |

Conclusion:

```text
Pure event_topk strongly improves bbox coverage, but it hurts precision and
dense score-map quality. It is not suitable as the standalone Stage 1 loss.
```

### Hybrid dense BCE + event_topk, 500 steps

```text
output root: /home/userroot/training_outputs/dagr-stage1/20260426_hybrid_event_500step
Ray result: 4/4 returncode 0
elapsed: about 24-25s per worker
```

Best results:

| Metric | Best experiment | Value |
|--------|-----------------|-------|
| Grid top-10 | `stage1_hybrid_bg1.00_event0.25_top0.10_box0.50` | 0.079904 |
| Selected event hit rate | `stage1_hybrid_bg1.00_event0.25_top0.10_box1.00` | 0.112215 |
| Selected/all lift | `stage1_hybrid_bg1.00_event0.50_top0.10_box0.50` | 1.216137 |
| Selected bbox coverage | `stage1_hybrid_bg1.00_event0.50_top0.10_box0.50` | 0.264247 |

Conclusion:

```text
The hybrid branch is more balanced than pure event_topk, but it still does not
beat the high-bg BCE seed-stability baseline:
selected_event_hit_rate: 0.127481 mean for bg1.00 seeds
selected_bbox_coverage: 0.234772 mean for bg1.00 seeds
grid top-10: 0.095817 mean for bg0.30 seeds
```

Do not extend the current event_topk/hybrid configurations to 5000 steps yet.
The next useful loss branch should explicitly balance selected-event precision
and per-box recall, for example by adding a fixed target for bbox-level recall
without letting dense background scores rise.
