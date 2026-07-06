"""Benchmark official DAGR detector training on the P40 Ray cluster.

This intentionally uses the detector path with ``event_selection=none`` so it
measures the DAGR training loop rather than the adaptive sampler experiments.
Each Ray task reserves one P40 GPU and launches one independent training trial.
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
DEFAULT_OUT_ROOT = Path("/home/userroot/training_outputs/dagr-official-benchmark")
PYTHONPATH = ":".join(
    [
        str(DAGR_DIR / "src"),
        str(DAGR_DIR / "libs/dsec-det/src"),
        str(DAGR_DIR / "libs/YOLOX"),
        os.environ.get("PYTHONPATH", ""),
    ]
)


def tail_lines(path: Path, n: int = 80) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-n:]


@ray.remote(num_cpus=3, num_gpus=1, resources={"P40": 1})
def run_trial(config: dict) -> dict:
    name = config["name"]
    run_dir = Path(config["out_root"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log = run_dir / "train_stdout.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["WANDB_MODE"] = "offline"
    env["WANDB_SILENT"] = "true"
    env["OMP_NUM_THREADS"] = str(config.get("omp_threads", 2))

    max_train_steps = int(config["max_train_steps"])
    batch_size = int(config["batch_size"])
    gpu_ids = ray.get_gpu_ids()
    assigned = ray.get_runtime_context().get_assigned_resources()

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
        name,
        "--batch_size",
        str(batch_size),
        "--tot_num_epochs",
        "1",
        "--event_selection",
        "none",
        "--max_train_steps",
        str(max_train_steps),
        "--skip_initial_eval",
        "--skip_epoch_eval",
    ]

    started = time.time()
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
    samples = max_train_steps * batch_size if proc.returncode == 0 else 0
    result = {
        "name": name,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "batch_size": batch_size,
        "max_train_steps": max_train_steps,
        "samples": samples,
        "steps_per_sec": round(max_train_steps / elapsed, 4)
        if proc.returncode == 0 and elapsed > 0 else 0.0,
        "samples_per_sec": round(samples / elapsed, 4)
        if samples > 0 and elapsed > 0 else 0.0,
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
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--out_root", type=Path,
                        default=DEFAULT_OUT_ROOT / "ray_runs")
    parser.add_argument("--detector_output_root", type=Path,
                        default=DEFAULT_OUT_ROOT / "detector_runs")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.detector_output_root.mkdir(parents=True, exist_ok=True)

    ray.init(address="auto")
    print("[dagr-official] cluster_resources=", ray.cluster_resources())
    print("[dagr-official] available_resources=", ray.available_resources())

    futures = [
        run_trial.remote(
            {
                "name": f"official_none_b{args.batch_size}_trial{i}",
                "out_root": str(args.out_root),
                "detector_output_root": str(args.detector_output_root),
                "batch_size": args.batch_size,
                "max_train_steps": args.max_train_steps,
            }
        )
        for i in range(args.num_trials)
    ]
    results = ray.get(futures)
    aggregate = {
        "num_trials": len(results),
        "total_elapsed_sec": round(max(r["elapsed_sec"] for r in results), 3),
        "sum_samples": sum(r["samples"] for r in results),
        "aggregate_samples_per_sec": round(
            sum(r["samples"] for r in results)
            / max(r["elapsed_sec"] for r in results),
            4,
        ),
        "results": results,
    }
    summary_path = args.out_root / "summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print("[dagr-official] summary=", summary_path)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
