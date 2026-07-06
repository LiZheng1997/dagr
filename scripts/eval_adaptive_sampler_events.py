"""Evaluate event-level selection quality for Stage-1 adaptive samplers.

This complements score-map selectivity. It runs the sampler's actual hard
top-K path and measures whether selected events fall inside GT boxes more often
than the full event stream.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from dagr.data.dsec_data import DSEC
from dagr.model.adaptive_sampler import AdaptiveEventSampler, AdaptiveSamplerConfig


def data_to_inputs(data, time_window_us: int, device: str):
    pos = data.pos.to(device)
    events = {
        "x": pos[:, 0].long(),
        "y": pos[:, 1].long(),
        "t": data.t.to(device).float() / float(time_window_us),
        "p": data.x.view(-1).long().to(device),
    }
    rgb = data.image.to(device).float() / 255.0
    if data.bbox.numel() > 0:
        bboxes = data.bbox[:, :4].to(device).float()
    else:
        bboxes = torch.zeros((0, 4), device=device)
    return events, rgb, bboxes


def event_box_stats(events: dict, bboxes: torch.Tensor) -> tuple[torch.Tensor, float]:
    x = events["x"].float()
    y = events["y"].float()
    inside_any = torch.zeros_like(x, dtype=torch.bool)
    covered_boxes = 0

    for row in bboxes:
        bx, by, bw, bh = [float(v.item()) for v in row]
        if bw <= 0 or bh <= 0:
            continue
        inside_box = (
            (x >= bx) & (x < bx + bw) &
            (y >= by) & (y < by + bh)
        )
        inside_any |= inside_box
        if inside_box.any():
            covered_boxes += 1

    box_coverage = covered_boxes / max(1, int(bboxes.shape[0]))
    return inside_any, box_coverage


def evaluate_checkpoint(ckpt_path: Path, ds: DSEC,
                        sample_indices: list[int], device: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = AdaptiveSamplerConfig(**ckpt["cfg"])
    sampler = AdaptiveEventSampler(cfg).to(device).eval()
    sampler.load_state_dict(ckpt["state_dict"])

    all_hit_rates = []
    selected_hit_rates = []
    lifts = []
    selected_box_coverages = []
    all_box_coverages = []
    event_budget_fracs = []
    total_events = []
    selected_events = []
    frames_with_box = 0

    with torch.no_grad():
        for idx in sample_indices:
            data = ds[idx]
            events, rgb, bboxes = data_to_inputs(data, ds.time_window, device)
            if bboxes.numel() == 0 or events["x"].numel() == 0:
                continue

            rgb_arg = rgb if cfg.use_image else None
            sampled, _, _ = sampler(events, rgb_arg, hard=True)

            all_inside, all_box_cov = event_box_stats(events, bboxes)
            selected_inside, selected_box_cov = event_box_stats(sampled, bboxes)

            all_rate = all_inside.float().mean().item()
            selected_rate = selected_inside.float().mean().item()
            lift = selected_rate / max(all_rate, 1e-12)

            frames_with_box += 1
            all_hit_rates.append(all_rate)
            selected_hit_rates.append(selected_rate)
            lifts.append(lift)
            all_box_coverages.append(all_box_cov)
            selected_box_coverages.append(selected_box_cov)
            total_events.append(int(events["x"].numel()))
            selected_events.append(int(sampled["x"].numel()))
            event_budget_fracs.append(
                float(sampled["x"].numel()) / float(events["x"].numel())
            )

    return {
        "name": ckpt_path.parent.name,
        "checkpoint": str(ckpt_path),
        "frames": len(sample_indices),
        "frames_with_box": frames_with_box,
        "K": cfg.K,
        "use_image": cfg.use_image,
        "mean_total_events": float(np.mean(total_events)) if total_events else None,
        "mean_selected_events": (
            float(np.mean(selected_events)) if selected_events else None
        ),
        "event_budget_fraction": (
            float(np.mean(event_budget_fracs)) if event_budget_fracs else None
        ),
        "all_event_hit_rate": (
            float(np.mean(all_hit_rates)) if all_hit_rates else None
        ),
        "selected_event_hit_rate": (
            float(np.mean(selected_hit_rates)) if selected_hit_rates else None
        ),
        "selected_vs_all_lift": float(np.mean(lifts)) if lifts else None,
        "all_bbox_coverage": (
            float(np.mean(all_box_coverages)) if all_box_coverages else None
        ),
        "selected_bbox_coverage": (
            float(np.mean(selected_box_coverages))
            if selected_box_coverages else None
        ),
    }


def find_checkpoints(run_root: Path) -> list[Path]:
    return sorted(run_root.glob("stage1_*/sampler_stage1_final.pth"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory", required=True, type=Path)
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--n_frames", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = DSEC(root=args.dataset_directory, split=args.split, transform=None,
              debug=False, min_bbox_height=0, min_bbox_diag=0,
              only_perfect_tracks=False)
    ds.set_num_us(-1)
    sample_indices = np.linspace(0, len(ds) - 1, args.n_frames,
                                 dtype=int).tolist()

    results = [
        evaluate_checkpoint(path, ds, sample_indices, device)
        for path in find_checkpoints(args.run_root)
    ]
    results = sorted(
        results,
        key=lambda item: (
            item["selected_event_hit_rate"] or -999,
            item["selected_vs_all_lift"] or -999,
        ),
        reverse=True,
    )

    out_path = args.out
    if out_path is None:
        out_path = (
            args.run_root / "metrics" /
            f"event_selection_{args.split}{args.n_frames}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"[event-selection] wrote {out_path}")


if __name__ == "__main__":
    main()
