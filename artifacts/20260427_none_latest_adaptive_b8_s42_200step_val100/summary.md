# DAGR Ray Pipeline Summary - 20260427_none_latest_adaptive_b8_s42_200step_val100

## Aggregate

| Trials | Successful | Wall time sec | Samples | Samples/sec |
|---:|---:|---:|---:|---:|
| 3 | 3 | 286.685 | 4800 | 16.7431 |

## Trials

| Name | Mode | Seed | Budget | Candidate | Batch | Steps | Wall sec | Samples/sec | best_mAP | Return |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `compare_b8_s42_none` | none | 42 | 5000 | -1 | 8 | 200 | 286.685 | 5.581 | 1.0716345e-05 | 0 |
| `compare_b8_s42_latest_k5000` | latest | 42 | 5000 | -1 | 8 | 200 | 75.831 | 21.0996 | 1.1139343e-05 | 0 |
| `compare_b8_s42_adaptive_k5000` | adaptive | 42 | 5000 | -1 | 8 | 200 | 81.357 | 19.6664 | 1.3450435e-05 | 0 |

## Artifacts

- Summary JSON: `/home/userroot/training_outputs/dagr-detector-ray-pipeline/ray_runs/20260427_none_latest_adaptive_b8_s42_200step_val100/summary.json`
- Ray run root: `/home/userroot/training_outputs/dagr-detector-ray-pipeline/ray_runs/20260427_none_latest_adaptive_b8_s42_200step_val100`
- Detector output root: `/home/userroot/training_outputs/dagr-detector-ray-pipeline/detector_runs/20260427_none_latest_adaptive_b8_s42_200step_val100`

## Notes

- `event_selection=none` is the official detector baseline path.
- `event_selection=latest` keeps the latest K events per graph.
- `event_selection=adaptive` applies a frozen Stage-1 sampler before DAGR.
- Ray manages GPU allocation across independent trials; it does not turn one detector training process into multi-GPU DDP.
