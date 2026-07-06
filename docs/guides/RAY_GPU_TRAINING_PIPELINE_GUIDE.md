# Ray GPU Training Pipeline Guide

This guide records the reusable GPU training workflow for the P40 host and the
current DAGR adaptive-sampling experiments.

## 1. Current GPU Setup

```text
Host: userroot@192.168.103.2
GPU: 2 x Tesla V100 32GB + 4 x Tesla P40 24GB
Ray dashboard: http://192.168.103.2:8265
Ray CLI: /home/userroot/miniconda3/bin/ray
DAGR env: /home/userroot/anaconda3/envs/dagr
DAGR repo: /home/userroot/dagr-adaptive-sampling
Ray wrapper dir: /home/userroot/dagr-ray
Dataset: /home/userroot/datasets/dsec-det
Output root: /home/userroot/training_outputs
```

Current Ray startup after the V100 expansion:

```bash
/home/userroot/miniconda3/bin/ray start --head \
  --dashboard-host=0.0.0.0 \
  --num-gpus=6 \
  --resources='{"P40":4,"V100":2}'
```

Verified Ray resources:

```text
CPU: 56
GPU: 6
P40: 4
V100: 2
```

Important limitation:

```text
Ray custom resources such as P40=1 or V100=1 are node-level logical
resources. On a single mixed-GPU node, they do not by themselves bind a task to
a physical P40 or V100 CUDA device.
```

For current DAGR multi-experiment sweeps, use `--gpu_resource any` to let Ray
schedule across all six GPUs. Use `--gpu_resource P40` or `--gpu_resource V100`
only as a logical throttle, not as a strict physical GPU selector.

Storage rule:

```text
/data/nas      cold archive
/data/p40ssd   hot staged NAS SSD tier
/scratch       local current-run cache
/home/userroot/datasets/dsec-det  current prepared DAGR dataset on local NVMe
```

For the current DAGR work, read from:

```text
/home/userroot/datasets/dsec-det
```

Do not point current jobs at `/data/nas` directly.

## 2. What We Already Proved

Ray can already schedule all four P40 GPUs on this host.

Original P40-only execution mode:

```python
@ray.remote(num_cpus=2, num_gpus=1, resources={"P40": 1})
```

This means:

```text
one independent experiment per GPU
4 experiments in one Ray job
Ray sets CUDA_VISIBLE_DEVICES per worker
each worker writes train_stdout.log, stage1_log.txt, ray_result.json
the sweep writes summary.json
```

Current wrapper execution mode:

```python
@ray.remote(num_cpus=2, num_gpus=1)
```

The wrapper adds an optional runtime resource request:

```bash
--gpu_resource any   # default; use all Ray-visible GPUs
--gpu_resource P40   # consume logical P40 resource
--gpu_resource V100  # consume logical V100 resource
```

On the current mixed single-node host, `any` is the recommended setting for
throughput. It allows six concurrent one-GPU experiments when the workload can
run on either P40 or V100.

Observed 5000-step high-background BCE sweep:

```text
output root: /home/userroot/training_outputs/dagr-stage1/20260425_highbg_5000step
Ray job: raysubmit_XAC7J1bya9b7BUS5
worker elapsed: about 136-137s each
parallel wall time: about 137s
serial equivalent: about 547s
effective speedup: about 4x for independent sweeps
```

Best standardized Stage 1 metrics so far:

```text
best top-10% precision:
stage1_k5000_lr3e-4_bg0.30 -> 0.101154

best inside/outside gap:
stage1_k5000_lr3e-4_bg1.00 -> 0.172101

GT grid fraction:
0.027236
```

## 3. Choose The Right Ray Mode

### Mode A: Multi-Experiment Scheduling

Use this when comparing many configs.

```text
best for: loss sweeps, learning-rate sweeps, ablations
GPU use: one process per GPU
code change: minimal wrapper around existing script
current DAGR Stage 1 recommendation: use this
```

Command shape:

```bash
cd /home/userroot/dagr-ray
/home/userroot/miniconda3/bin/ray job submit \
  --address http://192.168.103.2:8265 \
  --no-wait \
  --working-dir /home/userroot/dagr-ray \
  -- python run_adaptive_stage1_multi.py \
       --sweep high_bg \
       --max_steps 5000 \
       --gpu_resource any \
       --out_root /home/userroot/training_outputs/dagr-stage1/<run_id>
```

### Mode B: Single-Model Multi-GPU Training

Use this when one model should use all 4 GPUs.

```text
best for: large detector training, large batch, expensive single run
GPU use: DDP, one worker process per GPU
code change: training loop must be DDP/Ray Train aware
current DAGR recommendation: use later for full detector training, not Stage 1
```

Ray Train shape:

```python
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

trainer = TorchTrainer(
    train_loop_per_worker,
    train_loop_config={
        "dataset_directory": "/home/userroot/datasets",
        "dataset": "dsec-det",
        "output_directory": "/home/userroot/training_outputs/dagr-detector",
    },
    scaling_config=ScalingConfig(
        num_workers=4,
        use_gpu=True,
        resources_per_worker={"CPU": 2, "GPU": 1, "P40": 1},
    ),
)
result = trainer.fit()
```

## 3.1 Official Ray References

Useful official documentation:

```text
Ray Core accelerator support:
https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html

Ray Core resource scheduling and custom resources:
https://docs.ray.io/en/latest/ray-core/scheduling/resources.html

Ray Jobs submission:
https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html

Ray Train accelerator configuration:
https://docs.ray.io/en/latest/train/user-guides/using-accelerators.html
```

### Mode C: Ray Tune Search

Use this when the search space is larger than hand-picked sweeps.

```text
best for: many learning rates, loss weights, early stopping
GPU use: one or more GPUs per trial
code change: report metrics through Ray Tune
current DAGR recommendation: useful after selectivity metric is stable
```

Start with ASHA-style early stopping only when the validation metric is reliable.

## 4. Standard Run Directory

Use one root per sweep:

```text
/home/userroot/training_outputs/<project>/<run_id>
```

For DAGR Stage 1:

```text
/home/userroot/training_outputs/dagr-stage1/<run_id>
```

Each run root should contain:

```text
summary.json
metrics/
  selectivity_val200.json
<experiment_name>/
  train_stdout.log
  stage1_log.txt
  ray_result.json
  sampler_stage1_final.pth
viz_<experiment_name>/
  frame_*.png
```

Required metadata in every worker result:

```text
name
returncode
elapsed_sec
ray_gpu_ids
cuda_visible_devices
assigned_resources
run_dir
train_log
last_loss_line
```

## 5. Preflight Checklist

On P40:

```bash
hostname
nvidia-smi
du -sh /home/userroot/datasets/dsec-det
find /home/userroot/datasets/dsec-det -path '*/events/left/events_2x.h5' | wc -l
find /home/userroot/datasets/dsec-det/train -name timestamps.txt | wc -l
/home/userroot/miniconda3/bin/ray status --address 192.168.103.2:6379
```

Expected current DAGR dataset checks:

```text
events_2x.h5 files: 48
train timestamps: 47
```

## 6. Submit, Monitor, Evaluate

Submit a Ray job:

```bash
cd /home/userroot/dagr-ray
/home/userroot/miniconda3/bin/ray job submit \
  --address http://192.168.103.2:8265 \
  --no-wait \
  --working-dir /home/userroot/dagr-ray \
  -- python run_adaptive_stage1_multi.py \
       --sweep high_bg \
       --max_steps 5000 \
       --out_root /home/userroot/training_outputs/dagr-stage1/<run_id>
```

Check status:

```bash
/home/userroot/miniconda3/bin/ray job status <job_id> \
  --address http://192.168.103.2:8265

/home/userroot/miniconda3/bin/ray status --address 192.168.103.2:6379
```

Expected active 4-GPU sweep resource usage:

```text
8/12 CPU
4/4 GPU
4/4 P40
```

Tail logs:

```bash
/home/userroot/miniconda3/bin/ray job logs <job_id> \
  --address http://192.168.103.2:8265 | tail -100
```

Run standardized Stage 1 selectivity evaluation:

```bash
cd /home/userroot/dagr-adaptive-sampling
PYTHONPATH=/home/userroot/dagr-adaptive-sampling/src:/home/userroot/dagr-adaptive-sampling/libs/dsec-det/src:/home/userroot/dagr-adaptive-sampling/libs/YOLOX \
  /home/userroot/anaconda3/envs/dagr/bin/python \
  scripts/eval_adaptive_sampler_selectivity.py \
  --dataset_directory /home/userroot/datasets/dsec-det \
  --run_root /home/userroot/training_outputs/dagr-stage1/<run_id> \
  --split val \
  --n_frames 200
```

## 7. DAGR-Specific Policy

Stage 1 currently trains score maps only.

Important consequence:

```text
K is not in the current Stage 1 loss path.
K should be treated as a deployment/sampler setting until the forward sampler
selection path is included in the objective.
```

Use fixed seeds for comparable sweeps:

```text
--seed 42
```

Current active Stage 1 baseline:

```text
loss_type=bce
lr=3e-4
bg_weight=0.30 or 1.00
event_only=True
```

Do not move to full detector integration until:

```text
top10_precision improves materially beyond current 0.101154
inside/outside gap is stable
visual overlays concentrate around annotated objects
```

## 8. P40 Notes

Tesla P40 is Pascal generation and has no Tensor Cores.

Practical policy:

```text
Prefer FP32 unless a test proves AMP is faster.
Keep batch sizes conservative.
Use local NVMe for current prepared data.
Use /data/p40ssd for staged hot datasets.
Avoid reading training batches directly from /data/nas.
```

## 9. How Experienced Teams Usually Structure This

The common professional pattern is layered:

```text
training code:       PyTorch DDP/FSDP, Lightning, Accelerate, etc.
experiment runner:   Ray Jobs / Ray Tune / Hydra submitit
resource scheduler:  Ray, Slurm, Kubernetes, or managed cloud
metadata:            MLflow, W&B, TensorBoard, JSON summaries
data layer:          local NVMe cache, shared object store, NFS, or parallel FS
```

For a small single-host lab, start simple:

```text
Ray Jobs + JSON summaries + fixed run directories
```

For multi-user / multi-node / cloud, graduate to:

```text
KubeRay or Slurm for resource scheduling
Ray Tune or Determined AI for large HPO
SkyPilot for multi-cloud / mixed cloud-on-prem placement
```

## 10. Platform Decision

Recommended for this P40 host now:

```text
Keep Ray.
Use Mode A for Stage 1 sweeps.
Add Mode B Ray Train/DDP only when full DAGR detector training is ready.
Do not add Kubernetes or Slurm yet.
```

When to choose alternatives:

| Platform | Choose when | Avoid when |
|----------|-------------|------------|
| Ray Jobs/Core | single host or small cluster, Python-first sweeps | strict enterprise multi-user quotas are required |
| Ray Train | one model needs multi-GPU DDP/FSDP | tiny model where DDP overhead dominates |
| Ray Tune | many configs need ASHA/PBT/search algorithms | metric is noisy or not yet trusted |
| Slurm | HPC-style shared cluster with many users | one personal GPU box |
| KubeRay/Kubernetes | already operate Kubernetes and need autoscaling/services | no Kubernetes ops capacity |
| Kubeflow Trainer | Kubernetes-native ML platform with TrainJobs | single-node lab workflows |
| SkyPilot | heterogeneous clouds/on-prem/spot GPU placement | everything runs on one fixed machine |
| Determined AI | managed HPO + experiment tracking platform | you only need a lightweight wrapper |

## 11. References

- Ray TorchTrainer: https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html
- Ray Tune: https://docs.ray.io/en/latest/tune/index.html
- Ray accelerator support: https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html
- KubeRay GPUs: https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/gpu.html
- Kubernetes GPU scheduling: https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
- Kubeflow PyTorch Trainer: https://www.kubeflow.org/docs/components/trainer/user-guides/pytorch/
- Slurm GRES GPU scheduling: https://slurm.schedmd.com/gres.html
- SkyPilot overview: https://docs.skypilot.co/en/stable/docs/
