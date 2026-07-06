"""Visualize Adaptive Event Sampler score maps.

Loads a Stage-1 checkpoint, runs it over a few validation frames, and
writes PNG overlays showing:
  - input RGB
  - predicted score_map (sigmoid'd) as heatmap over the RGB
  - GT bboxes for visual check
  - the top-K selected events (for a sanity feel of density)

Usage:

    python scripts/viz_adaptive_sampler_scoremap.py \\
        --checkpoint log_seqs/adaptive_sampler_stage1_smoke/sampler_stage1_epoch4.pth \\
        --dataset_directory data/DSEC_fragment --split test \\
        --out_dir log_seqs/adaptive_sampler_stage1_smoke/viz --n_frames 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from dagr.data.dsec_data import DSEC
from dagr.model.adaptive_sampler import (
    AdaptiveEventSampler,
    AdaptiveSamplerConfig,
)

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def pyg_to_sampler_inputs(data, time_window_us, device):
    pos = data.pos.to(device)
    events = {
        'x': pos[:, 0].long(),
        'y': pos[:, 1].long(),
        't': data.t.to(device).float() / float(time_window_us),
        'p': data.x.view(-1).long().to(device),
    }
    rgb = data.image.to(device).float() / 255.0
    if data.bbox.numel() > 0:
        bboxes = data.bbox[:, :4].cpu().numpy()
    else:
        bboxes = np.zeros((0, 4), dtype=np.float32)
    return events, rgb, bboxes


def render_frame(rgb_np, score_prob, bboxes_xywh, sel_xy, out_path, title):
    H, W, _ = rgb_np.shape
    sp = F.interpolate(
        torch.from_numpy(score_prob)[None, None], size=(H, W),
        mode='bilinear', align_corners=False,
    )[0, 0].numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].imshow(rgb_np)
    ax[0].imshow(sp, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    for (bx, by, bw, bh) in bboxes_xywh:
        ax[0].add_patch(patches.Rectangle(
            (bx, by), bw, bh, linewidth=1.5,
            edgecolor='lime', facecolor='none'))
    ax[0].set_title("RGB + score_map heatmap + GT bbox (green)")
    ax[0].axis('off')

    ax[1].imshow(rgb_np)
    if sel_xy is not None and sel_xy.shape[0] > 0:
        ax[1].scatter(sel_xy[:, 0], sel_xy[:, 1], s=0.6,
                      c='cyan', alpha=0.4, linewidths=0)
    for (bx, by, bw, bh) in bboxes_xywh:
        ax[1].add_patch(patches.Rectangle(
            (bx, by), bw, bh, linewidth=1.5,
            edgecolor='lime', facecolor='none'))
    ax[1].set_title(
        f"RGB + top-K selected events ({0 if sel_xy is None else sel_xy.shape[0]})"
    )
    ax[1].axis('off')

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset_directory", required=True)
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--stride", type=int, default=1,
                    help="Take every Nth frame from the dataset.")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = DSEC(root=Path(args.dataset_directory), split=args.split,
              transform=None, debug=False, min_bbox_height=0,
              min_bbox_diag=0, only_perfect_tracks=False)
    ds.set_num_us(-1)
    print(f"[viz] split={args.split} samples={len(ds)}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = AdaptiveSamplerConfig(**ckpt['cfg'])
    sampler = AdaptiveEventSampler(cfg).to(device).eval()
    sampler.load_state_dict(ckpt['state_dict'])
    print(f"[viz] loaded ckpt epoch={ckpt.get('epoch')} -> K={cfg.K}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idxs = list(range(0, len(ds), args.stride))[:args.n_frames]
    with torch.no_grad():
        for j, idx in enumerate(idxs):
            data = ds[idx]
            events, rgb, bboxes = pyg_to_sampler_inputs(
                data, ds.time_window, device)
            sampled, sel_scores, score_map = sampler(
                events, rgb, hard=True)

            rgb_np = rgb[0].cpu().numpy().transpose(1, 2, 0)
            rgb_np = np.clip(rgb_np, 0, 1)
            sp = torch.sigmoid(score_map).cpu().numpy()

            sel_xy = torch.stack(
                [sampled['x'].float(), sampled['y'].float()], dim=1
            ).cpu().numpy()

            out_path = out_dir / f"frame_{j:03d}_idx{idx}.png"
            render_frame(
                rgb_np, sp, bboxes, sel_xy, out_path,
                title=(f"idx={idx}  N_events={events['x'].shape[0]}  "
                       f"K={sel_xy.shape[0]}  "
                       f"score_map min/max={sp.min():.3f}/{sp.max():.3f}")
            )
            print(f"  -> {out_path}  "
                  f"(score_map {sp.min():.3f}..{sp.max():.3f})")

    print(f"[viz] done. Artifacts at {out_dir}")


if __name__ == "__main__":
    main()
