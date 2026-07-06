"""Evaluate Stage-1 adaptive sampler score-map selectivity.

The metric is intentionally simple and fast: for a fixed set of validation
frames, compare mean score inside GT boxes against background and compute how
many top-scoring grid cells fall inside GT boxes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dagr.data.dsec_data import DSEC
from dagr.model.adaptive_sampler import (
    AdaptiveEventSampler,
    AdaptiveSamplerConfig,
)


def data_to_inputs(data, time_window_us: int, device: str):
    pos = data.pos.to(device)
    events = {
        "x": pos[:, 0].long(),
        "y": pos[:, 1].long(),
        "t": data.t.to(device).float() / float(time_window_us),
        "p": data.x.view(-1).long().to(device),
    }
    if data.bbox.numel() > 0:
        bboxes = data.bbox[:, :4].to(device)
    else:
        bboxes = torch.zeros((0, 4), device=device)
    return events, bboxes


def bbox_mask(bboxes: torch.Tensor, H: int, W: int,
              sensor_h: int, sensor_w: int) -> torch.Tensor:
    target = torch.zeros((H, W), device=bboxes.device, dtype=torch.bool)
    scale_x = W / sensor_w
    scale_y = H / sensor_h
    for row in bboxes:
        x, y, w, h = [float(v.item()) for v in row]
        if w <= 0 or h <= 0:
            continue
        gx0 = max(0, int(x * scale_x))
        gy0 = max(0, int(y * scale_y))
        gx1 = min(W, max(gx0 + 1, int((x + w) * scale_x)))
        gy1 = min(H, max(gy0 + 1, int((y + h) * scale_y)))
        target[gy0:gy1, gx0:gx1] = True
    return target


def evaluate_checkpoint(ckpt_path: Path, ds: DSEC,
                        sample_indices: list[int], device: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    sampler = AdaptiveEventSampler(AdaptiveSamplerConfig(**ckpt["cfg"]))
    sampler = sampler.to(device).eval()
    sampler.load_state_dict(ckpt["state_dict"])

    pos_vals = []
    neg_vals = []
    contrasts = []
    top5_precisions = []
    top10_precisions = []
    gt_fracs = []
    frames_with_box = 0

    with torch.no_grad():
        for idx in sample_indices:
            data = ds[idx]
            events, bboxes = data_to_inputs(data, ds.time_window, device)
            score_map = sampler.compute_score_map(events, None)
            prob = torch.sigmoid(score_map)
            target = bbox_mask(bboxes, prob.shape[0], prob.shape[1],
                               ds.height, ds.width)
            if not target.any():
                continue

            frames_with_box += 1
            pos_mean = prob[target].mean()
            neg_mean = prob[~target].mean()
            pos_vals.append(pos_mean.item())
            neg_vals.append(neg_mean.item())
            contrasts.append((pos_mean - neg_mean).item())
            gt_fracs.append(target.float().mean().item())

            flat = prob.flatten()
            target_flat = target.flatten()
            k10 = max(1, int(flat.numel() * 0.10))
            k5 = max(1, int(flat.numel() * 0.05))
            top10 = target_flat[flat.topk(k10).indices].float().mean()
            top5 = target_flat[flat.topk(k5).indices].float().mean()
            top10_precisions.append(top10.item())
            top5_precisions.append(top5.item())

    return {
        "name": ckpt_path.parent.name,
        "checkpoint": str(ckpt_path),
        "frames": len(sample_indices),
        "frames_with_box": frames_with_box,
        "gt_fraction": float(np.mean(gt_fracs)) if gt_fracs else None,
        "inside_mean": float(np.mean(pos_vals)) if pos_vals else None,
        "outside_mean": float(np.mean(neg_vals)) if neg_vals else None,
        "inside_minus_outside": (
            float(np.mean(contrasts)) if contrasts else None
        ),
        "top5_precision": (
            float(np.mean(top5_precisions)) if top5_precisions else None
        ),
        "top10_precision": (
            float(np.mean(top10_precisions)) if top10_precisions else None
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
            item["top10_precision"] or -999,
            item["inside_minus_outside"] or -999,
        ),
        reverse=True,
    )

    out_path = args.out
    if out_path is None:
        out_path = args.run_root / "metrics" / f"selectivity_{args.split}{args.n_frames}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"[selectivity] wrote {out_path}")


if __name__ == "__main__":
    main()
