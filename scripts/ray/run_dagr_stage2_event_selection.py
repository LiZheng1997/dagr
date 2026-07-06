"""Run DAGR detector event-selection comparisons with Ray.

Each Ray task reserves one P40 GPU and launches the existing DAGR detector
trainer with a fixed event budget. This compares the original latest-K cap
against the frozen Stage-1 adaptive sampler.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import ray


DAGR_DIR = Path("/home/userroot/dagr-adaptive-sampling")
PYTHON = "/home/userroot/anaconda3/envs/dagr/bin/python"
DATASET_DIR = Path("/home/userroot/datasets")
DEFAULT_OUT_ROOT = Path("/home/userroot/training_outputs/dagr-detector-stage2")
ADAPTIVE_BG1_CKPT = Path(
    "/home/userroot/training_outputs/dagr-stage1/20260426_seed_highbg_5000step/"
    "stage1_seed45_k5000_lr3e-4_bg1.00/sampler_stage1_final.pth"
)
ADAPTIVE_BG030_CKPT = Path(
    "/home/userroot/training_outputs/dagr-stage1/20260426_seed_highbg_5000step/"
    "stage1_seed44_k5000_lr3e-4_bg0.30/sampler_stage1_final.pth"
)
PYTHONPATH = ":".join(
    [
        str(DAGR_DIR / "src"),
        str(DAGR_DIR / "libs/dsec-det/src"),
        str(DAGR_DIR / "libs/YOLOX"),
        os.environ.get("PYTHONPATH", ""),
    ]
)


BASELINE_EXPERIMENTS = [
    {
        "name": "latest_k5000",
        "event_selection": "latest",
        "event_budget": 5000,
    },
    {
        "name": "latest_k10000",
        "event_selection": "latest",
        "event_budget": 10000,
    },
    {
        "name": "adaptive_bg1_seed45_k5000",
        "event_selection": "adaptive",
        "event_budget": 5000,
        "adaptive_sampler_checkpoint": str(ADAPTIVE_BG1_CKPT),
    },
    {
        "name": "adaptive_bg030_seed44_k5000",
        "event_selection": "adaptive",
        "event_budget": 5000,
        "adaptive_sampler_checkpoint": str(ADAPTIVE_BG030_CKPT),
    },
]

RECENT_EXPERIMENTS = [
    {
        "name": "latest_k5000",
        "event_selection": "latest",
        "event_budget": 5000,
    },
    {
        "name": "adaptive_bg1_seed45_recent10k_k5000",
        "event_selection": "adaptive",
        "event_budget": 5000,
        "event_candidate_budget": 10000,
        "adaptive_sampler_checkpoint": str(ADAPTIVE_BG1_CKPT),
    },
    {
        "name": "adaptive_bg1_seed45_recent20k_k5000",
        "event_selection": "adaptive",
        "event_budget": 5000,
        "event_candidate_budget": 20000,
        "adaptive_sampler_checkpoint": str(ADAPTIVE_BG1_CKPT),
    },
    {
        "name": "adaptive_bg030_seed44_recent20k_k5000",
        "event_selection": "adaptive",
        "event_budget": 5000,
        "event_candidate_budget": 20000,
        "adaptive_sampler_checkpoint": str(ADAPTIVE_BG030_CKPT),
    },
]


def tail_lines(path: Path, n: int = 80) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-n:]


def best_checkpoint(detector_output_root: Path, name: str) -> tuple[str, float | None]:
    run_dir = detector_output_root / "dsec-det" / "detection" / name
    matches = sorted(run_dir.glob("best_model_mAP_*.pth"))
    if not matches:
        return "", None

    best_path = matches[-1]
    try:
        best_map = float(best_path.stem.replace("best_model_mAP_", ""))
    except ValueError:
        best_map = None
    return str(best_path), best_map


@ray.remote(num_cpus=3, num_gpus=1, resources={"P40": 1})
def run_detector_experiment(config: dict) -> dict:
    run_dir = Path(config["out_root"]) / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log = run_dir / "train_stdout.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["WANDB_MODE"] = "offline"
    env["WANDB_SILENT"] = "true"
    env["OMP_NUM_THREADS"] = "2"

    gpu_ids = ray.get_gpu_ids()
    assigned = ray.get_runtime_context().get_assigned_resources()
    started = time.time()

    cmd = [
        PYTHON,
        str(DAGR_DIR / "scripts/train_dsec.py"),
        "--config",
        str(DAGR_DIR / "config/dagr-s-dsec.yaml"),
        "--dataset_directory",
        str(DATASET_DIR),
        "--dataset",
        "dsec-det",
        "--output_directory",
        str(config["detector_output_root"]),
        "--exp_name",
        config["name"],
        "--batch_size",
        str(config.get("batch_size", 2)),
        "--tot_num_epochs",
        "1",
        "--event_selection",
        config["event_selection"],
        "--event_budget",
        str(config["event_budget"]),
        "--max_train_steps",
        str(config.get("max_train_steps", 500)),
        "--dry_run_val_steps",
        str(config.get("dry_run_val_steps", 100)),
        "--skip_initial_eval",
    ]
    if config.get("skip_epoch_eval", False):
        cmd.append("--skip_epoch_eval")
    if config.get("event_candidate_budget", -1) > 0:
        cmd.extend([
            "--event_candidate_budget",
            str(config["event_candidate_budget"]),
        ])
    if config.get("adaptive_sampler_checkpoint"):
        cmd.extend([
            "--adaptive_sampler_checkpoint",
            config["adaptive_sampler_checkpoint"],
        ])

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
    best_path, best_map = best_checkpoint(
        Path(config["detector_output_root"]),
        config["name"],
    )
    result = {
        "name": config["name"],
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "best_checkpoint": best_path,
        "best_mAP": best_map,
        "ray_gpu_ids": gpu_ids,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "assigned_resources": assigned,
        "run_dir": str(run_dir),
        "train_log": str(train_log),
        "tail": tail_lines(train_log),
    }
    (run_dir / "ray_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--dry_run_val_steps", type=int, default=100)
    parser.add_argument("--out_root", type=Path,
                        default=DEFAULT_OUT_ROOT / "ray_runs")
    parser.add_argument("--detector_output_root", type=Path,
                        default=DEFAULT_OUT_ROOT)
    parser.add_argument("--skip_epoch_eval", action="store_true")
    parser.add_argument("--experiment_set",
                        choices=["baseline", "recent"],
                        default="baseline")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.detector_output_root.mkdir(parents=True, exist_ok=True)

    ray.init(address="auto")
    print("[ray-stage2] cluster_resources=", ray.cluster_resources())
    print("[ray-stage2] available_resources=", ray.available_resources())

    experiments = {
        "baseline": BASELINE_EXPERIMENTS,
        "recent": RECENT_EXPERIMENTS,
    }[args.experiment_set]

    futures = [
        run_detector_experiment.remote(
            {
                **exp,
                "out_root": str(args.out_root),
                "detector_output_root": str(args.detector_output_root),
                "max_train_steps": args.max_train_steps,
                "dry_run_val_steps": args.dry_run_val_steps,
                "skip_epoch_eval": args.skip_epoch_eval,
            }
        )
        for exp in experiments
    ]
    results = ray.get(futures)
    summary_path = args.out_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print("[ray-stage2] summary=", summary_path)
    for item in results:
        print(json.dumps(item, sort_keys=True))


if __name__ == "__main__":
    main()
