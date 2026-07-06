# DAGR Ray Training Pipeline Guide - 2026-04-27

## Goal

Use Ray to schedule reproducible DAGR detector experiments across the P40 GPUs.
The standard comparison is:

| Mode | Meaning |
|---|---|
| `none` | Official DAGR detector baseline. No event selection is applied. |
| `latest` | Keep the latest K events before the detector. |
| `adaptive` | Use a frozen Stage-1 adaptive sampler, then train the detector. |

Ray is used as a multi-experiment GPU scheduler. It does not make one detector
training process use multiple GPUs. For one model on multiple GPUs, add DDP or
Ray Train separately.

## Main Wrapper

Local repo path:

```bash
/home/lz/Documents/Events-Perception/dagr/scripts/ray/run_dagr_detector_ray_pipeline.py
```

Remote P40 target path:

```bash
/home/userroot/dagr-ray/run_dagr_detector_ray_pipeline.py
```

The wrapper writes:

- `pipeline_config.json`: exact trial configs
- `summary.json`: machine-readable aggregate and per-trial results
- `summary.md`: compact human-readable report
- per-trial `train_stdout.log` and `ray_result.json`

## Recommended P40 Command

Run a 3-way comparison with the official baseline, latest-K, and adaptive
Stage1:

```bash
cd /home/userroot/dagr-ray
/home/userroot/miniconda3/bin/python run_dagr_detector_ray_pipeline.py \
  --run_name 20260427_none_latest_adaptive_b8_s42_200step \
  --exp_prefix compare_b8_s42 \
  --modes none,latest,adaptive \
  --seeds 42 \
  --batch_size 8 \
  --event_budget 5000 \
  --max_train_steps 200 \
  --dry_run_val_steps 100
```

For a pure throughput run without detector validation:

```bash
cd /home/userroot/dagr-ray
/home/userroot/miniconda3/bin/python run_dagr_detector_ray_pipeline.py \
  --run_name 20260427_throughput_none_latest_adaptive_b8_s42_200step \
  --exp_prefix throughput_b8_s42 \
  --modes none,latest,adaptive \
  --seeds 42 \
  --batch_size 8 \
  --event_budget 5000 \
  --max_train_steps 200 \
  --skip_epoch_eval
```

## Key Parameters

| Parameter | Purpose |
|---|---|
| `--modes` | Comma-separated event selection modes, e.g. `none,latest,adaptive`. |
| `--seeds` | Comma-separated detector seeds, e.g. `42,43,44,45`. |
| `--batch_size` | Detector training batch size. |
| `--event_budget` | Event budget K for `latest` and `adaptive`. |
| `--event_candidate_budget` | Optional recent-event candidate cap before adaptive scoring. |
| `--adaptive_sampler_checkpoint` | Frozen Stage1 sampler checkpoint. |
| `--max_train_steps` | Detector optimizer steps per trial. |
| `--dry_run_val_steps` | Validation batches for quick metric smoke tests. |
| `--skip_epoch_eval` | Skip validation and measure training throughput only. |
| `--run_name` | Output directory name under the Ray pipeline output root. |
| `--exp_prefix` | Prefix for individual detector experiment names. |

Default adaptive checkpoint on P40:

```bash
/home/userroot/training_outputs/dagr-stage1/20260426_seed_highbg_5000step/stage1_seed45_k5000_lr3e-4_bg1.00/sampler_stage1_final.pth
```

## Output Layout

Default remote output root:

```bash
/home/userroot/training_outputs/dagr-detector-ray-pipeline/
```

Subdirectories:

| Path | Contents |
|---|---|
| `ray_runs/<run_name>/` | Ray logs, per-trial metadata, `summary.json`, `summary.md`. |
| `detector_runs/<run_name>/` | Native DAGR detector outputs and checkpoints. |

## Local Archival Rule

After a remote run completes, copy back the full Ray run directory or at least:

- `summary.json`
- `summary.md`
- `pipeline_config.json`

Recommended local location:

```bash
/home/lz/Documents/Events-Perception/dagr/artifacts/<run_name>/
```

Then add a short Markdown report under:

```bash
/home/lz/Documents/Events-Perception/dagr/docs/reports/
```

## Existing Benchmark Record

The official baseline Ray benchmark from 2026-04-27 is archived locally:

```bash
/home/lz/Documents/Events-Perception/dagr/docs/reports/DAGR_OFFICIAL_RAY_BENCHMARK_20260427.md
/home/lz/Documents/Events-Perception/dagr/artifacts/dagr-official-ray-benchmark-20260427/
```

It measured:

| Run | Aggregate throughput |
|---|---:|
| 1 x P40, one official DAGR trial | 6.5818 samples/sec |
| 4 x P40, four official DAGR trials | 26.2055 samples/sec |

## Next Standard Experiment

For a meaningful detector comparison, run:

| Trial | Mode | Event budget | Seed |
|---|---|---:|---:|
| official baseline | `none` | `-1` | 42 |
| latest-K baseline | `latest` | 5000 | 42 |
| adaptive Stage1 | `adaptive` | 5000 | 42 |

If this quick comparison is stable, scale to four seeds:

```bash
--modes none,latest,adaptive --seeds 42,43,44,45
```

That produces 12 detector trials. Ray will queue and schedule them across the
four P40 GPUs.
