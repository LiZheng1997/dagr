# DAGR Training Pipeline

This is the working convention for the current DAGR adaptive-sampling work on
the P40 host.

## Active Host

```text
Host: userroot@192.168.103.2
Conda env: /home/userroot/anaconda3/envs/dagr
DAGR code: /home/userroot/dagr-adaptive-sampling
Stage 1 Ray wrapper: /home/userroot/dagr-ray/run_adaptive_stage1_multi.py
Stage 2 Ray wrapper: /home/userroot/dagr-ray/run_dagr_stage2_event_selection.py
Ray dashboard: http://192.168.103.2:8265
```

## Data Inputs

Current training reads the prepared DSEC-Det tree directly from local P40 NVMe:

```text
/home/userroot/datasets/dsec-det
```

This path is the active dataset root for `train_adaptive_sampler_stage1.py` and
the Ray wrapper. Do not point current DAGR jobs at `/data/nas` or
`/data/p40ssd` unless the prepared `dsec-det` tree is explicitly restaged and
the wrapper constants are changed.

Source archives came from JX NAS cold storage:

```text
/data/nas/Datasets/DSEC/train_events.zip
/data/nas/Datasets/DSEC/test_events.zip
/data/nas/Datasets/DSEC-Det/dsec-det.zip
/data/nas/Datasets/DSEC-Det/dsec-det_left_images_distorted.zip
/data/nas/Datasets/DSEC-Det/dsec-det_left_object_detections.zip
```

The generic long-term storage rule is still:

```text
/data/nas      cold archive
/data/p40ssd   hot staged NAS SSD data
/scratch       local current-run cache
```

For this specific DAGR run, the already-prepared local dataset is preferred.

## Output Convention

Old exploratory outputs are under:

```text
/home/userroot/training_outputs
```

Going forward, use one run root per stage:

```text
/home/userroot/training_outputs/dagr-stage1/<run_id>
/home/userroot/training_outputs/dagr-detector-stage2/<run_id>
```

Use run IDs with timestamp, variant, and key hyperparameters:

```text
20260425_1904_event_only_highbg_500step
20260425_1930_event_only_focal_topk_500step
```

Each Stage 1 Ray sweep should contain:

```text
summary.json
<experiment_name>/
  train_stdout.log
  stage1_log.txt
  ray_result.json
  sampler_stage1_final.pth
viz_<selection_name>/
  frame_*.png
metrics/
  selectivity_val200.json
```

Only selected figures and summaries should be copied back locally:

```text
/home/lz/Documents/BluetVPN/artifacts/
/home/lz/Documents/BluetVPN/docs/
```

Do not copy full checkpoints back unless they are needed for local inspection.

## Ray Usage

Current Ray usage is multi-experiment scheduling:

```text
@ray.remote(num_cpus=2, num_gpus=1, resources={"P40": 1})
```

This launches independent Stage 1 runs concurrently, one GPU per experiment.
It is useful for quick hyperparameter/loss sweeps because the wall time is close
to the slowest single experiment instead of the sum of all experiments.

This is different from one model using all four GPUs. For single-model 4-GPU
training, use Ray Train `TorchTrainer` or native PyTorch DDP:

```text
num_workers=4
use_gpu=True
one process per GPU
gradient synchronization through DDP/NCCL
```

Stage 1 is small enough that multi-experiment Ray scheduling is currently more
valuable than DDP. Full DAGR detector training is the better candidate for
single-model 4-GPU distributed training.

## Current Stage 1 Loss

The current default Stage 1 objective is a binary coverage mask loss:

```text
target=1 inside any GT bbox grid cell
target=0 outside GT bboxes
loss = BCE(positive cells) + bg_weight * BCE(background cells)
```

This is implemented as `coverage_loss()` in:

```text
/home/userroot/dagr-adaptive-sampling/src/dagr/model/adaptive_sampler.py
```

It is useful for a first sanity check, but it optimizes the average heatmap
instead of the top-K selected region. Our 500-step results show that higher
`bg_weight` improves inside/outside contrast, but top-10% precision is still
low. That means the sampler is learning "objects tend to be positive", but
background peaks are not yet suppressed enough.

## Current Best Stage 1 Loss

Use the high-background weighted BCE result as the current baseline. Since
Stage 1 currently trains only `score_map`, `K` does not affect the loss and
should be treated as a deployment setting unless the sampler forward path is
used in the objective.

```text
loss_type=bce
lr=3e-4
bg_weight=0.30 or 1.00
seed=42 for comparable sweeps
```

Current best 5000-step checkpoints:

```text
best top-10% precision:
/home/userroot/training_outputs/dagr-stage1/20260425_highbg_5000step/stage1_k5000_lr3e-4_bg0.30/sampler_stage1_final.pth

best inside/outside gap:
/home/userroot/training_outputs/dagr-stage1/20260425_highbg_5000step/stage1_k5000_lr3e-4_bg1.00/sampler_stage1_final.pth
```

Validation selectivity on 200 val frames:

```text
GT grid fraction=0.027236
best top10_precision=0.101154
best inside_minus_outside=0.172101
```

I tested top-k-aware and Gaussian alternatives, but none beat the high-bg BCE
baseline in this 500-step/5000-step evidence set:

```text
focal_topk best: inside_minus_outside=0.057881, top10_precision=0.093798
bce_topk best:   inside_minus_outside=0.093617, top10_precision=0.093029
gaussian best:   inside_minus_outside=0.059738, top10_precision=0.059087
high-bg 5000:    inside_minus_outside=0.172101, top10_precision=0.101154
```

So the active recommendation is:

```text
Keep high-bg weighted BCE as the default Stage 1 baseline.
Keep focal_topk, bce_topk, and gaussian/centernet as experimental ablations.
Do not integrate Stage 1 into the full DAGR detector until visual heatmaps and
top-k metrics are acceptable.
```

## Experimental Stage 1 Losses

The new experimental losses are available for sweeps:

```text
focal_topk = BCE(positive cells)
           + bg_weight * focal_BCE(background cells)
           + topk_bg_weight * BCE(top background cells, target=0)

bce_topk = BCE(positive cells)
         + bg_weight * BCE(background cells)
         + topk_bg_weight * BCE(top background cells, target=0)

gaussian_bce = BCE(logits, soft Gaussian object target)

centernet = CenterNet-style modified focal loss on Gaussian object centers
```

Reasoning:

- Stage 1 is a dense heatmap problem with strong foreground/background
  imbalance, so focal loss is worth testing.
- The sampler later selects top-K events, so the loss should directly punish
  high-scoring background cells, not only average background score.
- This is a Stage 1 pretraining loss. The final detector still needs DAGR's
  normal YOLOX-style detection loss after sampler integration.

Example Ray sweeps:

```bash
cd /home/userroot/dagr-ray
ray job submit \
  --address http://192.168.103.2:8265 \
  --no-wait \
  --working-dir /home/userroot/dagr-ray \
  -- python run_adaptive_stage1_multi.py \
       --sweep focal_topk \
       --max_steps 500 \
       --out_root /home/userroot/training_outputs/dagr-stage1/20260425_focal_topk_500step

ray job submit \
  --address http://192.168.103.2:8265 \
  --no-wait \
  --working-dir /home/userroot/dagr-ray \
  -- python run_adaptive_stage1_multi.py \
       --sweep bce_topk \
       --max_steps 500 \
       --out_root /home/userroot/training_outputs/dagr-stage1/20260425_bce_topk_500step
```

Compare all experimental losses against the current best standardized baseline:

```text
/home/userroot/training_outputs/dagr-stage1/20260425_highbg_5000step/stage1_k5000_lr3e-4_bg0.30/sampler_stage1_final.pth
/home/userroot/training_outputs/dagr-stage1/20260425_highbg_5000step/stage1_k5000_lr3e-4_bg1.00/sampler_stage1_final.pth
```

Primary Stage 1 metrics:

```text
inside_mean
outside_mean
inside_minus_outside
mean_frame_contrast
top10_precision
selected_event_hit_rate
selected_vs_all_lift
selected_bbox_coverage
```

Promotion rule before DAGR detector integration:

```text
inside_minus_outside should improve over 0.172101
top10_precision should improve over 0.101154
selected_event_hit_rate should improve over 0.133942
selected_bbox_coverage should improve over 0.276735 at K=5000
visual heatmaps should concentrate near annotated objects
```

Current Stage 1 baseline checkpoints:

```text
Best grid checkpoint:
/home/userroot/training_outputs/dagr-stage1/20260426_seed_highbg_5000step/stage1_seed44_k5000_lr3e-4_bg0.30/sampler_stage1_final.pth

Best event-selection checkpoint:
/home/userroot/training_outputs/dagr-stage1/20260426_seed_highbg_5000step/stage1_seed45_k5000_lr3e-4_bg1.00/sampler_stage1_final.pth
```

Do not promote the current `event_topk` or `hybrid_event_bce` 500-step
branches yet. They improved selected bbox coverage in some cases, but did not
beat the high-background BCE baseline on selected-event precision or grid
selectivity.

## Stage 2 Detector Event Selection

The detector integration is intentionally a hard event-selection ablation, not
yet a learned end-to-end adaptive sampler. Selection happens in:

```text
/home/userroot/dagr-adaptive-sampling/scripts/train_dsec.py
```

after `data.cuda()` and before `format_data(data)`. This keeps the DAGR graph
construction and detector unchanged while swapping the event subset.

Supported selector arguments:

```text
--event_selection none|latest|adaptive
--event_budget <events per graph>
--event_candidate_budget <optional recent candidate cap for adaptive>
--adaptive_sampler_checkpoint <sampler_stage1_final.pth>
```

Ray wrapper:

```bash
cd /home/userroot/dagr-ray
/home/userroot/miniconda3/bin/python run_dagr_stage2_event_selection.py \
  --experiment_set baseline \
  --max_train_steps 1000 \
  --dry_run_val_steps 200 \
  --out_root /home/userroot/training_outputs/dagr-detector-stage2/ray_runs/<run_id> \
  --detector_output_root /home/userroot/training_outputs/dagr-detector-stage2/<run_id>
```

Ray resources are reserved as one trial per GPU:

```text
@ray.remote(num_cpus=3, num_gpus=1, resources={"P40": 1})
```

This is multi-experiment GPU scheduling. It used all four P40 GPUs concurrently
in the 2026-04-26 runs, with `summary.json` recording `ray_gpu_ids`,
`cuda_visible_devices`, assigned resources, elapsed time, and best checkpoint.

Stage 2 quick evidence, 1000 train steps and 200 validation batches:

```text
Run: 20260426_k5k_k10k_4way_1000step_val200
latest_k5000                   mAP=0.00013482124689955443  GPU0  135.135s
latest_k10000                  mAP=0.000009718266029251583 GPU1  170.142s
adaptive_bg1_seed45_k5000       mAP=0.000014487455017561648 GPU2  150.991s
adaptive_bg030_seed44_k5000     mAP=0.000013234815965354773 GPU3  150.081s

Run: 20260426_recent_adaptive_4way_1000step_val200
latest_k5000                            mAP=0.00013464488856473035 GPU0  142.359s
adaptive_bg1_seed45_recent10k_k5000     mAP=0.000013039854695327338 GPU1  151.849s
adaptive_bg1_seed45_recent20k_k5000     mAP=0.0000129819483471163   GPU2  146.217s
adaptive_bg030_seed44_recent20k_k5000   mAP=0.000017914822327875592 GPU3  149.509s
```

Interpretation:

```text
latest_k5000 is the current Stage 2 event-selection baseline.
The current Stage 1 score-map sampler should not be promoted as a hard detector
front-end yet.
Recent-window adaptive selection improves slightly over full-window adaptive in
one bg0.30 variant, but it is still far behind latest_k5000.
K=10000 latest is slower and worse in this short-run evidence, so increasing
the hard event budget is not an automatic win.
```

Current next research direction:

```text
Keep latest_k5000 as the detector baseline.
Treat Stage 1 as an auxiliary prior, not as a hard replacement filter.
Next useful ablations are soft score features, time-decayed adaptive scoring,
or a hybrid union that always preserves a latest-event backbone.
```

## Official DAGR Relation

Official DAGR does not include this adaptive sampler Stage 1. Its released DSEC
training path uses the DAGR detector with a YOLOX-style detection head and
standard checkpointing through `last_model.pth` and `best_model_mAP_*.pth`.

For full detector training, use:

```text
/home/userroot/dagr-adaptive-sampling/scripts/train_dsec.py
```

with dataset directory:

```text
--dataset_directory /home/userroot/datasets
--dataset dsec-det
```

and output directory:

```text
--output_directory /home/userroot/training_outputs/dagr-detector
```

The script resolves the actual dataset as:

```text
/home/userroot/datasets/dsec-det
```

## References

- Official DAGR repository: https://github.com/uzh-rpg/dagr
- DAGR Nature paper: https://www.nature.com/articles/s41586-024-07409-w
- Focal Loss paper: https://arxiv.org/abs/1708.02002
- 1Mpx event detection dataset/paper: https://arxiv.org/abs/2009.13436
