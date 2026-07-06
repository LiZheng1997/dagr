"""Run short DAGR adaptive-sampler Stage 1 experiments with Ray.

This wrapper is intentionally small and observable: each Ray task reserves one
P40 GPU, runs the existing Stage 1 trainer for a few steps, and writes a JSON
summary that records Ray's GPU assignment and the final loss line.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import argparse
from pathlib import Path

import ray


DAGR_DIR = Path("/home/userroot/dagr-adaptive-sampling")
DATASET_DIR = Path("/home/userroot/datasets/dsec-det")
DEFAULT_OUT_ROOT = Path("/home/userroot/training_outputs/ray_stage1_smoke")
PYTHON = "/home/userroot/anaconda3/envs/dagr/bin/python"
GPU_RESOURCE_CHOICES = ("any", "P40", "V100")
PYTHONPATH = ":".join(
    [
        str(DAGR_DIR / "src"),
        str(DAGR_DIR / "libs/dsec-det/src"),
        str(DAGR_DIR / "libs/YOLOX"),
        os.environ.get("PYTHONPATH", ""),
    ]
)


def ray_remote_options(gpu_resource: str) -> dict:
    if gpu_resource == "any":
        return {}
    if gpu_resource not in GPU_RESOURCE_CHOICES:
        raise ValueError(f"Unsupported gpu_resource: {gpu_resource}")
    return {"resources": {gpu_resource: 1}}


EXPERIMENTS = [
    {
        "name": "stage1_k5000_lr1e-4_bg0.05",
        "K": 5000,
        "lr": 1e-4,
        "bg_weight": 0.05,
        "seed": 42,
    },
    {
        "name": "stage1_k5000_lr3e-4_bg0.10",
        "K": 5000,
        "lr": 3e-4,
        "bg_weight": 0.10,
        "seed": 42,
    },
    {
        "name": "stage1_k10000_lr1e-4_bg0.05",
        "K": 10000,
        "lr": 1e-4,
        "bg_weight": 0.05,
        "seed": 42,
    },
    {
        "name": "stage1_k10000_lr3e-4_bg0.10",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 0.10,
        "seed": 42,
    },
]


HIGH_BG_EXPERIMENTS = [
    {
        "name": "stage1_k5000_lr3e-4_bg0.30",
        "K": 5000,
        "lr": 3e-4,
        "bg_weight": 0.30,
        "seed": 42,
    },
    {
        "name": "stage1_k5000_lr3e-4_bg1.00",
        "K": 5000,
        "lr": 3e-4,
        "bg_weight": 1.00,
        "seed": 42,
    },
    {
        "name": "stage1_k10000_lr3e-4_bg0.30",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 0.30,
        "seed": 42,
    },
    {
        "name": "stage1_k10000_lr3e-4_bg1.00",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 1.00,
        "seed": 42,
    },
]


SEED_HIGHBG_EXPERIMENTS = [
    {
        "name": f"stage1_seed{seed}_k5000_lr3e-4_bg0.30",
        "K": 5000,
        "lr": 3e-4,
        "bg_weight": 0.30,
        "seed": seed,
    }
    for seed in (42, 43, 44, 45)
] + [
    {
        "name": f"stage1_seed{seed}_k5000_lr3e-4_bg1.00",
        "K": 5000,
        "lr": 3e-4,
        "bg_weight": 1.00,
        "seed": seed,
    }
    for seed in (42, 43, 44, 45)
]


FOCAL_TOPK_EXPERIMENTS = [
    {
        "name": "stage1_focal_k10000_lr3e-4_bg0.30_top0.10",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 0.30,
        "loss_type": "focal_topk",
        "focal_gamma": 2.0,
        "topk_bg_weight": 0.10,
        "topk_bg_frac": 0.10,
        "seed": 42,
    },
    {
        "name": "stage1_focal_k10000_lr3e-4_bg0.30_top0.30",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 0.30,
        "loss_type": "focal_topk",
        "focal_gamma": 2.0,
        "topk_bg_weight": 0.30,
        "topk_bg_frac": 0.10,
        "seed": 42,
    },
    {
        "name": "stage1_focal_k10000_lr3e-4_bg0.50_top0.10",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 0.50,
        "loss_type": "focal_topk",
        "focal_gamma": 2.0,
        "topk_bg_weight": 0.10,
        "topk_bg_frac": 0.10,
        "seed": 42,
    },
    {
        "name": "stage1_focal_k10000_lr3e-4_bg0.50_top0.30",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 0.50,
        "loss_type": "focal_topk",
        "focal_gamma": 2.0,
        "topk_bg_weight": 0.30,
        "topk_bg_frac": 0.10,
        "seed": 42,
    },
]


BCE_TOPK_EXPERIMENTS = [
    {
        "name": "stage1_bcetopk_k10000_lr3e-4_bg0.30_top0.05",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 0.30,
        "loss_type": "bce_topk",
        "topk_bg_weight": 0.05,
        "topk_bg_frac": 0.10,
        "seed": 42,
    },
    {
        "name": "stage1_bcetopk_k10000_lr3e-4_bg0.30_top0.10",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 0.30,
        "loss_type": "bce_topk",
        "topk_bg_weight": 0.10,
        "topk_bg_frac": 0.10,
        "seed": 42,
    },
    {
        "name": "stage1_bcetopk_k10000_lr3e-4_bg0.50_top0.05",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 0.50,
        "loss_type": "bce_topk",
        "topk_bg_weight": 0.05,
        "topk_bg_frac": 0.10,
        "seed": 42,
    },
    {
        "name": "stage1_bcetopk_k10000_lr3e-4_bg0.50_top0.10",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 0.50,
        "loss_type": "bce_topk",
        "topk_bg_weight": 0.10,
        "topk_bg_frac": 0.10,
        "seed": 42,
    },
]


GAUSSIAN_EXPERIMENTS = [
    {
        "name": "stage1_gaussbce_k10000_lr3e-4_bg0.30_sig0.30",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 0.30,
        "loss_type": "gaussian_bce",
        "gaussian_sigma_scale": 0.30,
        "seed": 42,
    },
    {
        "name": "stage1_gaussbce_k10000_lr3e-4_bg0.30_sig0.50",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 0.30,
        "loss_type": "gaussian_bce",
        "gaussian_sigma_scale": 0.50,
        "seed": 42,
    },
    {
        "name": "stage1_centernet_k10000_lr3e-4_sig0.30",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 0.30,
        "loss_type": "centernet",
        "gaussian_sigma_scale": 0.30,
        "seed": 42,
    },
    {
        "name": "stage1_centernet_k10000_lr3e-4_sig0.50",
        "K": 10000,
        "lr": 3e-4,
        "bg_weight": 0.30,
        "loss_type": "centernet",
        "gaussian_sigma_scale": 0.50,
        "seed": 42,
    },
]


EVENT_TOPK_EXPERIMENTS = [
    {
        "name": "stage1_eventtopk_bg0.05_top0.10_box0.50",
        "K": 5000,
        "lr": 3e-4,
        "bg_weight": 0.05,
        "loss_type": "event_topk",
        "topk_bg_weight": 0.10,
        "topk_bg_frac": 0.10,
        "box_cover_weight": 0.50,
        "seed": 42,
    },
    {
        "name": "stage1_eventtopk_bg0.05_top0.30_box0.50",
        "K": 5000,
        "lr": 3e-4,
        "bg_weight": 0.05,
        "loss_type": "event_topk",
        "topk_bg_weight": 0.30,
        "topk_bg_frac": 0.10,
        "box_cover_weight": 0.50,
        "seed": 42,
    },
    {
        "name": "stage1_eventtopk_bg0.10_top0.10_box1.00",
        "K": 5000,
        "lr": 3e-4,
        "bg_weight": 0.10,
        "loss_type": "event_topk",
        "topk_bg_weight": 0.10,
        "topk_bg_frac": 0.10,
        "box_cover_weight": 1.00,
        "seed": 42,
    },
    {
        "name": "stage1_eventtopk_bg0.10_top0.30_box1.00",
        "K": 5000,
        "lr": 3e-4,
        "bg_weight": 0.10,
        "loss_type": "event_topk",
        "topk_bg_weight": 0.30,
        "topk_bg_frac": 0.10,
        "box_cover_weight": 1.00,
        "seed": 42,
    },
]


HYBRID_EVENT_EXPERIMENTS = [
    {
        "name": "stage1_hybrid_bg1.00_event0.25_top0.10_box0.50",
        "K": 5000,
        "lr": 3e-4,
        "bg_weight": 1.00,
        "loss_type": "hybrid_event_bce",
        "topk_bg_weight": 0.10,
        "topk_bg_frac": 0.10,
        "box_cover_weight": 0.50,
        "event_loss_weight": 0.25,
        "seed": 42,
    },
    {
        "name": "stage1_hybrid_bg1.00_event0.50_top0.10_box0.50",
        "K": 5000,
        "lr": 3e-4,
        "bg_weight": 1.00,
        "loss_type": "hybrid_event_bce",
        "topk_bg_weight": 0.10,
        "topk_bg_frac": 0.10,
        "box_cover_weight": 0.50,
        "event_loss_weight": 0.50,
        "seed": 42,
    },
    {
        "name": "stage1_hybrid_bg1.00_event0.25_top0.10_box1.00",
        "K": 5000,
        "lr": 3e-4,
        "bg_weight": 1.00,
        "loss_type": "hybrid_event_bce",
        "topk_bg_weight": 0.10,
        "topk_bg_frac": 0.10,
        "box_cover_weight": 1.00,
        "event_loss_weight": 0.25,
        "seed": 42,
    },
    {
        "name": "stage1_hybrid_bg0.30_event0.50_top0.10_box1.00",
        "K": 5000,
        "lr": 3e-4,
        "bg_weight": 0.30,
        "loss_type": "hybrid_event_bce",
        "topk_bg_weight": 0.10,
        "topk_bg_frac": 0.10,
        "box_cover_weight": 1.00,
        "event_loss_weight": 0.50,
        "seed": 42,
    },
]


def parse_loss(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    lines = [line.strip() for line in log_path.read_text().splitlines()]
    loss_lines = [line for line in lines if "loss=" in line]
    return loss_lines[-1] if loss_lines else None


@ray.remote(num_cpus=2, num_gpus=1)
def run_stage1_experiment(config: dict) -> dict:
    run_dir = Path(config["out_root"]) / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log = run_dir / "train_stdout.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["OMP_NUM_THREADS"] = "2"

    gpu_ids = ray.get_gpu_ids()
    assigned = ray.get_runtime_context().get_assigned_resources()
    started = time.time()

    cmd = [
        PYTHON,
        str(DAGR_DIR / "scripts/train_adaptive_sampler_stage1.py"),
        "--dataset_directory",
        str(DATASET_DIR),
        "--output_dir",
        str(run_dir),
        "--split",
        "train",
        "--epochs",
        "1",
        "--batch_size",
        "1",
        "--num_workers",
        "0",
        "--max_steps",
        str(config.get("max_steps", 3)),
        "--log_every",
        "1",
        "--K",
        str(config["K"]),
        "--lr",
        str(config["lr"]),
        "--bg_weight",
        str(config["bg_weight"]),
        "--seed",
        str(config.get("seed", 42)),
    ]
    if config.get("event_only", True):
        cmd.append("--event_only")
    if config.get("loss_type"):
        cmd.extend(["--loss_type", str(config["loss_type"])])
    if config.get("focal_gamma") is not None:
        cmd.extend(["--focal_gamma", str(config["focal_gamma"])])
    if config.get("topk_bg_weight") is not None:
        cmd.extend(["--topk_bg_weight", str(config["topk_bg_weight"])])
    if config.get("topk_bg_frac") is not None:
        cmd.extend(["--topk_bg_frac", str(config["topk_bg_frac"])])
    if config.get("box_cover_weight") is not None:
        cmd.extend(["--box_cover_weight", str(config["box_cover_weight"])])
    if config.get("event_loss_weight") is not None:
        cmd.extend(["--event_loss_weight", str(config["event_loss_weight"])])
    if config.get("gaussian_sigma_scale") is not None:
        cmd.extend([
            "--gaussian_sigma_scale", str(config["gaussian_sigma_scale"])
        ])
    if config.get("gaussian_min_sigma") is not None:
        cmd.extend(["--gaussian_min_sigma", str(config["gaussian_min_sigma"])])

    with train_log.open("w") as fh:
        fh.write(f"[ray-meta] pid={os.getpid()}\n")
        fh.write(f"[ray-meta] ray_gpu_ids={gpu_ids}\n")
        fh.write(f"[ray-meta] assigned_resources={assigned}\n")
        fh.write(
            f"[ray-meta] CUDA_VISIBLE_DEVICES="
            f"{env.get('CUDA_VISIBLE_DEVICES', '')}\n"
        )
        fh.write(f"[ray-meta] cmd={' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(DAGR_DIR),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )

    elapsed = round(time.time() - started, 3)
    result = {
        "name": config["name"],
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "gpu_resource": config.get("gpu_resource", "any"),
        "ray_gpu_ids": gpu_ids,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "assigned_resources": assigned,
        "last_loss_line": parse_loss(run_dir / "stage1_log.txt"),
        "run_dir": str(run_dir),
        "train_log": str(train_log),
    }
    (run_dir / "ray_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--use_image", action="store_true",
                        help="Condition the sampler on RGB images. Default is event-only.")
    parser.add_argument("--sweep",
                        choices=["baseline", "high_bg", "focal_topk",
                                 "bce_topk", "gaussian", "seed_highbg",
                                 "event_topk", "hybrid_event"],
                        default="baseline")
    parser.add_argument("--gpu_resource", choices=GPU_RESOURCE_CHOICES,
                        default="any",
                        help="Logical Ray resource per trial. Use 'any' to "
                             "schedule on any GPU, or P40/V100 to consume the "
                             "corresponding custom resource.")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    ray.init(address="auto")
    print("[ray-stage1] cluster_resources=", ray.cluster_resources())
    print("[ray-stage1] available_resources=", ray.available_resources())
    if args.sweep == "high_bg":
        experiments = HIGH_BG_EXPERIMENTS
    elif args.sweep == "seed_highbg":
        experiments = SEED_HIGHBG_EXPERIMENTS
    elif args.sweep == "focal_topk":
        experiments = FOCAL_TOPK_EXPERIMENTS
    elif args.sweep == "bce_topk":
        experiments = BCE_TOPK_EXPERIMENTS
    elif args.sweep == "gaussian":
        experiments = GAUSSIAN_EXPERIMENTS
    elif args.sweep == "event_topk":
        experiments = EVENT_TOPK_EXPERIMENTS
    elif args.sweep == "hybrid_event":
        experiments = HYBRID_EVENT_EXPERIMENTS
    else:
        experiments = EXPERIMENTS
    futures = [
        run_stage1_experiment.options(
            **ray_remote_options(args.gpu_resource)
        ).remote(
            {
                **exp,
                "max_steps": args.max_steps,
                "out_root": str(args.out_root),
                "event_only": not args.use_image,
                "gpu_resource": args.gpu_resource,
            }
        )
        for exp in experiments
    ]
    results = ray.get(futures)
    summary_path = args.out_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print("[ray-stage1] summary=", summary_path)
    for item in results:
        print(json.dumps(item, sort_keys=True))


if __name__ == "__main__":
    main()
