# DAGR Official Training Ray Benchmark - 2026-04-27

## Purpose

Benchmark the official DAGR detector training path on the P40 host, then compare
single-trial throughput with Ray-managed 4-GPU multi-trial throughput.

This run is a train-set throughput benchmark, not a full validation metric run.

## Environment

| Item | Value |
|---|---|
| Remote host | `userroot@192.168.103.2` |
| GPU | 4 x NVIDIA Tesla P40 24GB |
| DAGR repo | `/home/userroot/dagr-adaptive-sampling` |
| Dataset | `/home/userroot/datasets/dsec-det` |
| DAGR Python | `/home/userroot/anaconda3/envs/dagr/bin/python` |
| Ray driver Python | `/home/userroot/miniconda3/bin/python` |
| Ray dashboard | `http://192.168.103.2:8265` |

## Training Definition

The benchmark used the official DAGR detector training path:

- Entry point: `scripts/train_dsec.py`
- Config: `config/dagr-s-dsec.yaml`
- Event selection: `none`
- Batch size: `8`
- Steps per trial: `200`
- Initial and epoch evaluation were skipped for throughput measurement.

`event_selection=none` means this benchmark does not use the adaptive Stage1
sampler. It measures the baseline detector training path.

## Ray Configuration

Each Ray task reserved:

- `num_gpus=1`
- `num_cpus=3`
- custom resource: `P40=1`

The 4-GPU benchmark launched four independent DAGR training trials in parallel,
one trial per GPU.

## Results

| Run | Trials | GPUs used | Batch size | Steps per trial | Total samples | Wall time sec | Aggregate samples/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single Ray trial | 1 | 1 | 8 | 200 | 1600 | 243.093 | 6.5818 |
| Four Ray trials | 4 | 4 | 8 each | 200 each | 6400 | 244.224 | 26.2055 |

## Per-Trial Details

| Trial | GPU | Wall time sec | Steps/sec | Samples/sec | Return code |
|---|---:|---:|---:|---:|---:|
| single: `official_none_b8_trial0` | 0 | 243.093 | 0.8227 | 6.5818 | 0 |
| four: `official_none_b8_trial0` | 0 | 244.224 | 0.8189 | 6.5514 | 0 |
| four: `official_none_b8_trial1` | 1 | 242.442 | 0.8249 | 6.5995 | 0 |
| four: `official_none_b8_trial2` | 2 | 239.443 | 0.8353 | 6.6822 | 0 |
| four: `official_none_b8_trial3` | 3 | 242.479 | 0.8248 | 6.5985 | 0 |

## Interpretation

Ray did not make one DAGR training process faster. Its benefit here is GPU
resource scheduling across multiple independent experiments.

The 4-trial Ray run reached `26.2055 samples/sec`, compared with
`6.5818 samples/sec` for the single-trial run. That is an aggregate speedup of
about `3.98x`, with almost no per-trial slowdown.

This indicates the P40 host can efficiently run four independent DAGR training
experiments in parallel. For a single model to use all four GPUs, the next step
would be a DDP or Ray Train implementation, which is a different target from
this multi-experiment scheduler benchmark.

## Local Artifacts

| Artifact | Path |
|---|---|
| Local benchmark script | `/home/lz/Documents/Events-Perception/dagr/scripts/ray/run_dagr_official_ray_benchmark.py` |
| Single-trial summary JSON | `/home/lz/Documents/Events-Perception/dagr/artifacts/dagr-official-ray-benchmark-20260427/summary_1trial_b8_200step.json` |
| Four-trial summary JSON | `/home/lz/Documents/Events-Perception/dagr/artifacts/dagr-official-ray-benchmark-20260427/summary_4trial_b8_200step.json` |

## Remote Artifacts

| Artifact | Path |
|---|---|
| Remote benchmark script | `/home/userroot/dagr-ray/run_dagr_official_ray_benchmark.py` |
| Single-trial summary JSON | `/home/userroot/training_outputs/dagr-official-benchmark/ray_runs/20260427_official_b8_1trial_200step/summary.json` |
| Four-trial summary JSON | `/home/userroot/training_outputs/dagr-official-benchmark/ray_runs/20260427_official_b8_4trial_200step/summary.json` |
| Detector outputs | `/home/userroot/training_outputs/dagr-official-benchmark/detector_runs/` |

## Next Pipeline Step

Use the same Ray wrapper to run controlled comparisons:

| Experiment | Event selection | Purpose |
|---|---|---|
| Official baseline | `none` | Baseline DAGR detector throughput and loss |
| Latest-event baseline | `latest` | Compare simple event selection with official baseline |
| Adaptive Stage1 | `adaptive` | Measure adaptive sampler cost and potential benefit |

The recommended next benchmark is four parallel trials with different seeds or
event selection modes, all using the same batch size and step budget.
