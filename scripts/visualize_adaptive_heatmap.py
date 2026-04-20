"""Quick visualizer for Adaptive Event Sampler heatmaps.

Loads a Stage-1 checkpoint, runs it on N samples from the DSEC test
split, and writes side-by-side PNGs showing:
  (left)   RGB image with GT bboxes
  (middle) score_map heatmap, upsampled and overlaid on RGB
  (right)  the K events selected by top-K, plotted on RGB

Use this after Stage 1 to eyeball whether the sampler is learning
anything useful before committing to Stage 2.

Example:

    cd /path/to/dagr
    python scripts/visualize_adaptive_heatmap.py \\
        --ckpt log_seqs/adaptive_sampler_stage1/sampler_stage1_epoch4.pth \\
        --dataset_directory data/DSEC_fragment \\
        --num_samples 8 \\
        --output_dir log_seqs/adaptive_heatmap_viz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from dagr.data.dsec_data import DSEC
from dagr.model.adaptive_sampler import (
    AdaptiveEventSampler,
    AdaptiveSamplerConfig,
)


def load_sampler(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_d = ckpt.get("cfg", {})
    cfg = AdaptiveSamplerConfig(**cfg_d) if cfg_d else AdaptiveSamplerConfig()
    sampler = AdaptiveEventSampler(cfg).to(device).eval()
    sampler.load_state_dict(ckpt["state_dict"])
    return sampler, cfg


def render_one(idx, data, sampler, cfg, device, out_dir: Path):
    with torch.no_grad():
        pos = data.pos.to(device)
        events = {
            'x': pos[:, 0].long(),
            'y': pos[:, 1].long(),
            't': data.t.to(device).float() / float(data.time_window),
            'p': data.x.view(-1).long().to(device),
        }
        rgb = data.image.to(device).float() / 255.0  # [1,3,H,W]

        sampled, scores_K, score_map = sampler(events, rgb, hard=True)

    # --- panel 1: RGB + GT bboxes ---------------------------------------
    rgb_u8 = (rgb[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    panel_gt = rgb_u8.copy()
    if data.bbox.numel() > 0:
        for row in data.bbox.cpu().numpy():
            x, y, w, h = row[:4]
            x0, y0 = int(x), int(y)
            x1, y1 = int(x + w), int(y + h)
            cv2.rectangle(panel_gt, (x0, y0), (x1, y1), (0, 255, 0), 1)

    # --- panel 2: heatmap overlay ---------------------------------------
    hm = torch.sigmoid(score_map).cpu().numpy().astype(np.float32)
    hm_resized = cv2.resize(
        hm, (rgb_u8.shape[1], rgb_u8.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    hm_color = cv2.applyColorMap(
        (hm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    panel_heat = cv2.addWeighted(rgb_u8, 0.5, hm_color, 0.5, 0.0)

    # --- panel 3: selected events ---------------------------------------
    panel_sel = np.zeros_like(rgb_u8)
    panel_sel = rgb_u8.copy() // 3  # dim RGB background
    sx = sampled['x'].cpu().numpy()
    sy = sampled['y'].cpu().numpy()
    sp = sampled['p'].cpu().numpy()
    # positive = red, negative = blue
    for (xi, yi, pi) in zip(sx, sy, sp):
        if 0 <= xi < panel_sel.shape[1] and 0 <= yi < panel_sel.shape[0]:
            if pi > 0:
                panel_sel[yi, xi] = (0, 0, 255)   # BGR red
            else:
                panel_sel[yi, xi] = (255, 0, 0)   # BGR blue

    # --- composite ------------------------------------------------------
    panel = np.hstack([panel_gt, panel_heat, panel_sel])
    cv2.putText(panel, f"#{idx} N={len(sx)} K={cfg.K}",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    out_path = out_dir / f"sample_{idx:04d}.png"
    cv2.imwrite(str(out_path), panel)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--dataset_directory", required=True, type=str)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--output_dir", required=True, type=str)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler, cfg = load_sampler(args.ckpt, device)
    ds = DSEC(
        root=Path(args.dataset_directory), split=args.split,
        transform=None, num_us=-1, debug=False,
        min_bbox_height=0, min_bbox_diag=0,
        only_perfect_tracks=False,
    )
    n = min(args.num_samples, len(ds))
    print(f"[viz] {n} samples from {args.split} → {out_dir}")
    stride = max(1, len(ds) // n)
    for i, idx in enumerate(range(0, len(ds), stride)):
        if i >= n:
            break
        data = ds[idx]
        out_path = render_one(i, data, sampler, cfg, device, out_dir)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
