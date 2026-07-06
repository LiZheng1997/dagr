"""Run DAGR detector event-selection experiments through Ray.

The default matrix compares the original detector path
(``event_selection=none``), a latest-K event budget, and a frozen Stage-1
adaptive sampler. Each Ray task owns one GPU and launches one independent
``train_dsec.py`` process.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import ray


DEFAULT_DAGR_DIR = Path("/home/userroot/dagr-adaptive-sampling")
DEFAULT_PYTHON = "/home/userroot/anaconda3/envs/dagr/bin/python"
DEFAULT_DATASET_ROOT = Path("/home/userroot/datasets")
DEFAULT_OUTPUT_ROOT = Path("/home/userroot/training_outputs/dagr-detector-ray-pipeline")
DEFAULT_ADAPTIVE_CKPT = Path(
    "/home/userroot/training_outputs/dagr-stage1/20260426_seed_highbg_5000step/"
    "stage1_seed45_k5000_lr3e-4_bg1.00/sampler_stage1_final.pth"
)
GPU_RESOURCE_CHOICES = ("any", "P40", "V100")


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in csv_list(value)]


def ray_remote_options(gpu_resource: str) -> dict:
    if gpu_resource == "any":
        return {}
    if gpu_resource not in GPU_RESOURCE_CHOICES:
        raise ValueError(f"Unsupported gpu_resource: {gpu_resource}")
    return {"resources": {gpu_resource: 1}}


def tail_lines(path: Path, n: int = 80) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-n:]


def last_matching_line_from_file(path: Path, needle: str) -> str:
    if not path.exists():
        return ""
    match = ""
    with path.open(errors="replace") as fh:
        for line in fh:
            if needle in line:
                match = line.rstrip()
    return match


def build_pythonpath(dagr_dir: Path) -> str:
    return ":".join(
        [
            str(dagr_dir / "src"),
            str(dagr_dir / "libs/dsec-det/src"),
            str(dagr_dir / "libs/YOLOX"),
            os.environ.get("PYTHONPATH", ""),
        ]
    )


def detector_run_dir(detector_output_root: Path, dataset: str, name: str) -> Path:
    return detector_output_root / dataset / "detection" / name


def best_checkpoint(detector_output_root: Path, dataset: str,
                    name: str) -> tuple[str, float | None]:
    run_dir = detector_run_dir(detector_output_root, dataset, name)
    best: tuple[Path, float] | None = None
    for path in run_dir.glob("best_model_mAP_*.pth"):
        try:
            metric = float(path.stem.replace("best_model_mAP_", ""))
        except ValueError:
            continue
        if best is None or metric > best[1]:
            best = (path, metric)
    if best is None:
        return "", None
    return str(best[0]), best[1]


def build_experiments(args: argparse.Namespace) -> list[dict]:
    if args.experiments_json:
        loaded = json.loads(args.experiments_json.read_text())
        if not isinstance(loaded, list):
            raise ValueError("--experiments_json must contain a JSON list")
        return loaded

    experiments = []
    for seed in csv_ints(args.seeds):
        for mode in csv_list(args.modes):
            if mode not in {"none", "latest", "adaptive"}:
                raise ValueError(f"Unsupported event selection mode: {mode}")

            suffix = mode
            exp = {
                "event_selection": mode,
                "seed": seed,
                "batch_size": args.batch_size,
                "max_train_steps": args.max_train_steps,
                "dry_run_val_steps": args.dry_run_val_steps,
            }
            if mode != "none":
                exp["event_budget"] = args.event_budget
                suffix += f"_k{args.event_budget}"
            if mode == "adaptive":
                exp["adaptive_sampler_checkpoint"] = str(
                    args.adaptive_sampler_checkpoint
                )
                if args.event_candidate_budget > 0:
                    exp["event_candidate_budget"] = args.event_candidate_budget
                    suffix += f"_recent{args.event_candidate_budget}"

            seed_part = f"_seed{seed}" if len(csv_ints(args.seeds)) > 1 else ""
            exp["name"] = f"{args.exp_prefix}_{suffix}{seed_part}"
            experiments.append(exp)
    return experiments


def command_for_trial(config: dict) -> list[str]:
    dagr_dir = Path(config["dagr_dir"])
    cmd = [
        config["python"],
        str(dagr_dir / "scripts/train_dsec.py"),
        "--config",
        str(config["config"]),
        "--dataset_directory",
        str(config["dataset_root"]),
        "--dataset",
        config["dataset"],
        "--output_directory",
        str(config["detector_output_root"]),
        "--exp_name",
        config["name"],
        "--batch_size",
        str(config["batch_size"]),
        "--tot_num_epochs",
        str(config["tot_num_epochs"]),
        "--seed",
        str(config["seed"]),
        "--event_selection",
        config["event_selection"],
        "--max_train_steps",
        str(config["max_train_steps"]),
        "--dry_run_val_steps",
        str(config["dry_run_val_steps"]),
    ]

    if not config.get("run_initial_eval", False):
        cmd.append("--skip_initial_eval")
    if config.get("skip_epoch_eval", False):
        cmd.append("--skip_epoch_eval")

    event_budget = int(config.get("event_budget", -1))
    if event_budget > 0:
        cmd.extend(["--event_budget", str(event_budget)])

    candidate_budget = int(config.get("event_candidate_budget", -1))
    if candidate_budget > 0:
        cmd.extend(["--event_candidate_budget", str(candidate_budget)])

    checkpoint = config.get("adaptive_sampler_checkpoint")
    if config["event_selection"] == "adaptive":
        if not checkpoint:
            raise ValueError(
                "adaptive experiments require adaptive_sampler_checkpoint"
            )
        cmd.extend(["--adaptive_sampler_checkpoint", checkpoint])

    return cmd


@ray.remote(num_cpus=3, num_gpus=1)
def run_trial(config: dict) -> dict:
    run_dir = Path(config["out_root"]) / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log = run_dir / "train_stdout.log"
    result_path = run_dir / "ray_result.json"

    dagr_dir = Path(config["dagr_dir"])
    detector_output_root = Path(config["detector_output_root"])
    env = os.environ.copy()
    env["PYTHONPATH"] = build_pythonpath(dagr_dir)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["WANDB_MODE"] = config.get("wandb_mode", "offline")
    env["WANDB_SILENT"] = "true"
    env["OMP_NUM_THREADS"] = str(config.get("omp_threads", 2))

    cmd = command_for_trial(config)
    gpu_ids = ray.get_gpu_ids()
    assigned = ray.get_runtime_context().get_assigned_resources()

    started = time.time()
    with train_log.open("w") as fh:
        fh.write(f"[ray-meta] pid={os.getpid()}\n")
        fh.write(f"[ray-meta] ray_gpu_ids={gpu_ids}\n")
        fh.write(f"[ray-meta] assigned_resources={assigned}\n")
        fh.write(
            f"[ray-meta] CUDA_VISIBLE_DEVICES="
            f"{env.get('CUDA_VISIBLE_DEVICES', '')}\n"
        )
        fh.write("[ray-meta] config=" + json.dumps(config, sort_keys=True) + "\n")
        fh.write("[ray-meta] cmd=" + " ".join(shlex.quote(x) for x in cmd) + "\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(dagr_dir),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )

    elapsed = round(time.time() - started, 3)
    max_steps = int(config["max_train_steps"])
    batch_size = int(config["batch_size"])
    samples = max_steps * batch_size if proc.returncode == 0 and max_steps > 0 else 0
    tail = tail_lines(train_log)
    best_path, best_map = best_checkpoint(
        detector_output_root, config["dataset"], config["name"]
    )
    last_model = detector_run_dir(
        detector_output_root, config["dataset"], config["name"]
    ) / "last_model.pth"

    result = {
        "name": config["name"],
        "event_selection": config["event_selection"],
        "event_budget": config.get("event_budget", -1),
        "event_candidate_budget": config.get("event_candidate_budget", -1),
        "adaptive_sampler_checkpoint": config.get("adaptive_sampler_checkpoint", ""),
        "gpu_resource": config.get("gpu_resource", "any"),
        "seed": config["seed"],
        "batch_size": batch_size,
        "max_train_steps": max_steps,
        "dry_run_val_steps": config["dry_run_val_steps"],
        "skip_epoch_eval": config.get("skip_epoch_eval", False),
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "samples": samples,
        "steps_per_sec": round(max_steps / elapsed, 4)
        if proc.returncode == 0 and elapsed > 0 and max_steps > 0 else 0.0,
        "samples_per_sec": round(samples / elapsed, 4)
        if samples > 0 and elapsed > 0 else 0.0,
        "best_checkpoint": best_path,
        "best_mAP": best_map,
        "last_checkpoint": str(last_model) if last_model.exists() else "",
        "last_train_line": last_matching_line_from_file(
            train_log, "[train] step="
        ),
        "ray_gpu_ids": gpu_ids,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "assigned_resources": assigned,
        "run_dir": str(run_dir),
        "train_log": str(train_log),
        "tail": tail,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def write_markdown(summary: dict, path: Path) -> None:
    lines = [
        f"# DAGR Ray Pipeline Summary - {summary['run_name']}",
        "",
        "## Aggregate",
        "",
        "| Trials | Successful | Wall time sec | Samples | Samples/sec |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {summary['num_trials']} | {summary['successful_trials']} | "
            f"{summary['total_elapsed_sec']} | {summary['sum_samples']} | "
            f"{summary['aggregate_samples_per_sec']} |"
        ),
        "",
        "## Trials",
        "",
        (
            "| Name | Mode | Seed | Budget | Candidate | Batch | Steps | "
            "Wall sec | Samples/sec | best_mAP | Return |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["results"]:
        best_map = item["best_mAP"]
        best_map_text = "" if best_map is None else f"{best_map:.8g}"
        lines.append(
            f"| `{item['name']}` | {item['event_selection']} | {item['seed']} | "
            f"{item['event_budget']} | {item['event_candidate_budget']} | "
            f"{item['batch_size']} | {item['max_train_steps']} | "
            f"{item['elapsed_sec']} | {item['samples_per_sec']} | "
            f"{best_map_text} | {item['returncode']} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary JSON: `{summary['summary_json']}`",
            f"- Ray run root: `{summary['out_root']}`",
            f"- Detector output root: `{summary['detector_output_root']}`",
            "",
            "## Notes",
            "",
            "- `event_selection=none` is the official detector baseline path.",
            "- `event_selection=latest` keeps the latest K events per graph.",
            "- `event_selection=adaptive` applies a frozen Stage-1 sampler before DAGR.",
            "- Ray manages GPU allocation across independent trials; it does not turn one detector training process into multi-GPU DDP.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="")
    parser.add_argument("--exp_prefix", default="dagr_compare")
    parser.add_argument("--modes", default="none,latest,adaptive")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--experiments_json", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--tot_num_epochs", type=int, default=1)
    parser.add_argument("--dry_run_val_steps", type=int, default=100)
    parser.add_argument("--event_budget", type=int, default=5000)
    parser.add_argument("--event_candidate_budget", type=int, default=-1)
    parser.add_argument("--adaptive_sampler_checkpoint", type=Path,
                        default=DEFAULT_ADAPTIVE_CKPT)
    parser.add_argument("--skip_epoch_eval", action="store_true")
    parser.add_argument("--run_initial_eval", action="store_true")
    parser.add_argument("--dagr_dir", type=Path, default=DEFAULT_DAGR_DIR)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset", default="dsec-det")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out_root", type=Path, default=None)
    parser.add_argument("--detector_output_root", type=Path, default=None)
    parser.add_argument("--wandb_mode", default="offline")
    parser.add_argument("--omp_threads", type=int, default=2)
    parser.add_argument("--gpu_resource", choices=GPU_RESOURCE_CHOICES,
                        default="any",
                        help="Logical Ray resource per trial. Use 'any' to "
                             "schedule on any GPU, or P40/V100 to consume the "
                             "corresponding custom resource.")
    args = parser.parse_args()

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S_dagr_detector")
    out_root = args.out_root or DEFAULT_OUTPUT_ROOT / "ray_runs" / run_name
    detector_output_root = (
        args.detector_output_root
        or DEFAULT_OUTPUT_ROOT / "detector_runs" / run_name
    )
    config = args.config or args.dagr_dir / "config/dagr-s-dsec.yaml"
    out_root.mkdir(parents=True, exist_ok=True)
    detector_output_root.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(args)
    common = {
        "run_name": run_name,
        "dagr_dir": str(args.dagr_dir),
        "python": args.python,
        "dataset_root": str(args.dataset_root),
        "dataset": args.dataset,
        "config": str(config),
        "out_root": str(out_root),
        "detector_output_root": str(detector_output_root),
        "seed": csv_ints(args.seeds)[0],
        "batch_size": args.batch_size,
        "event_budget": args.event_budget,
        "max_train_steps": args.max_train_steps,
        "dry_run_val_steps": args.dry_run_val_steps,
        "tot_num_epochs": args.tot_num_epochs,
        "skip_epoch_eval": args.skip_epoch_eval,
        "run_initial_eval": args.run_initial_eval,
        "wandb_mode": args.wandb_mode,
        "omp_threads": args.omp_threads,
        "gpu_resource": args.gpu_resource,
    }
    trial_configs = [{**common, **exp} for exp in experiments]
    (out_root / "pipeline_config.json").write_text(
        json.dumps(trial_configs, indent=2, sort_keys=True) + "\n"
    )

    ray.init(address="auto")
    print("[dagr-ray] cluster_resources=", ray.cluster_resources())
    print("[dagr-ray] available_resources=", ray.available_resources())
    print("[dagr-ray] run_name=", run_name)
    print("[dagr-ray] trials=", len(trial_configs))

    results = ray.get([
        run_trial.options(**ray_remote_options(cfg["gpu_resource"])).remote(cfg)
        for cfg in trial_configs
    ])
    total_elapsed = max((item["elapsed_sec"] for item in results), default=0.0)
    sum_samples = sum(item["samples"] for item in results)
    summary_path = out_root / "summary.json"
    markdown_path = out_root / "summary.md"
    summary = {
        "run_name": run_name,
        "num_trials": len(results),
        "successful_trials": sum(1 for item in results if item["returncode"] == 0),
        "total_elapsed_sec": round(total_elapsed, 3),
        "sum_samples": sum_samples,
        "aggregate_samples_per_sec": round(sum_samples / total_elapsed, 4)
        if total_elapsed > 0 else 0.0,
        "out_root": str(out_root),
        "detector_output_root": str(detector_output_root),
        "summary_json": str(summary_path),
        "summary_markdown": str(markdown_path),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(summary, markdown_path)
    print("[dagr-ray] summary_json=", summary_path)
    print("[dagr-ray] summary_markdown=", markdown_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
