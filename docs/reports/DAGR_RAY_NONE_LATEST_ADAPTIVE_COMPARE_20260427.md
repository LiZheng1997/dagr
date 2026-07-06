# DAGR Ray None/Latest/Adaptive Comparison - 2026-04-27

## Run

| Item | Value |
|---|---|
| Remote host | `userroot@192.168.103.2` |
| Wrapper | `/home/userroot/dagr-ray/run_dagr_detector_ray_pipeline.py` |
| Local wrapper | `/home/lz/Documents/Events-Perception/dagr/scripts/ray/run_dagr_detector_ray_pipeline.py` |
| Run name | `20260427_none_latest_adaptive_b8_s42_200step_val100` |
| Dataset | `/home/userroot/datasets/dsec-det` |
| Batch size | `8` |
| Steps per trial | `200` |
| Seed | `42` |
| Event budget | `5000` |
| Validation | `dry_run_val_steps=100` |

This is a controlled short-run comparison. The mAP values are from a dry-run
validation subset and should be treated as relative smoke metrics, not final
official DSEC-DET metrics.

## Results

| Trial | Mode | Wall sec | Steps/sec | Samples/sec | Speedup vs none | dry-run best mAP |
|---|---|---:|---:|---:|---:|---:|
| `compare_b8_s42_none` | `none` | 286.685 | 0.6976 | 5.5810 | 1.00x | 0.0000107163 |
| `compare_b8_s42_latest_k5000` | `latest` | 75.831 | 2.6374 | 21.0996 | 3.78x | 0.0000111393 |
| `compare_b8_s42_adaptive_k5000` | `adaptive` | 81.357 | 2.4583 | 19.6664 | 3.52x | 0.0000134504 |

Aggregate Ray throughput for the three concurrently scheduled trials was
`16.7431 samples/sec` over `286.685` seconds.

## Interpretation

`latest_k5000` and `adaptive_k5000` both dramatically reduced detector training
time compared with the official no-selection path. The frozen Stage1 adaptive
sampler adds overhead relative to latest-K, but it still ran about `3.52x`
faster than no event selection in this short run.

The dry-run mAP ordering was:

1. `adaptive_k5000`: `0.0000134504`
2. `latest_k5000`: `0.0000111393`
3. `none`: `0.0000107163`

Because the validation is only `100` batches and the detector was trained for
only `200` steps, this result should be used to decide which longer experiment
to run next, not as a publishable metric.

## Artifacts

Local archive:

```bash
/home/lz/Documents/Events-Perception/dagr/artifacts/20260427_none_latest_adaptive_b8_s42_200step_val100/
```

Files:

- `summary.json`
- `summary.md`
- `pipeline_config.json`

Remote Ray output:

```bash
/home/userroot/training_outputs/dagr-detector-ray-pipeline/ray_runs/20260427_none_latest_adaptive_b8_s42_200step_val100/
```

Remote detector checkpoints:

```bash
/home/userroot/training_outputs/dagr-detector-ray-pipeline/detector_runs/20260427_none_latest_adaptive_b8_s42_200step_val100/
```

Best checkpoints:

| Trial | Checkpoint |
|---|---|
| `none` | `/home/userroot/training_outputs/dagr-detector-ray-pipeline/detector_runs/20260427_none_latest_adaptive_b8_s42_200step_val100/dsec-det/detection/compare_b8_s42_none/best_model_mAP_1.071634461003825e-05.pth` |
| `latest_k5000` | `/home/userroot/training_outputs/dagr-detector-ray-pipeline/detector_runs/20260427_none_latest_adaptive_b8_s42_200step_val100/dsec-det/detection/compare_b8_s42_latest_k5000/best_model_mAP_1.1139343285590204e-05.pth` |
| `adaptive_k5000` | `/home/userroot/training_outputs/dagr-detector-ray-pipeline/detector_runs/20260427_none_latest_adaptive_b8_s42_200step_val100/dsec-det/detection/compare_b8_s42_adaptive_k5000/best_model_mAP_1.34504354789795e-05.pth` |

## Next Experiment

Run a longer seed sweep:

```bash
cd /home/userroot/dagr-ray
/home/userroot/miniconda3/bin/python run_dagr_detector_ray_pipeline.py \
  --run_name 20260427_none_latest_adaptive_b8_seed42_45_1000step_val200 \
  --exp_prefix compare_b8 \
  --modes none,latest,adaptive \
  --seeds 42,43,44,45 \
  --batch_size 8 \
  --event_budget 5000 \
  --max_train_steps 1000 \
  --dry_run_val_steps 200
```

That schedules `12` trials across the four P40 GPUs. It is the next useful
step before spending time on a full official metric run.
