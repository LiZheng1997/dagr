"""Stage-1 trainer for the Adaptive Event Sampler.

Warm-up pass that trains ONLY the sampler (image encoder + event
voxelizer + importance scorer) with a coverage loss:

    1 inside GT bboxes, 0 outside — weighted BCE on the logit heatmap

The downstream DAGR detector is NOT used in Stage 1. After convergence
the score_map should localize "object regions" well enough for visual
inspection. If that looks reasonable, proceed to Stage 2 (joint
fine-tuning with DAGR detection loss).

Example:

    cd /path/to/dagr
    python scripts/train_adaptive_sampler_stage1.py \\
        --config config/dagr-s-dsec.yaml \\
        --dataset_directory data/DSEC_fragment \\
        --epochs 5 --batch_size 2 \\
        --output_dir log_seqs/adaptive_sampler_stage1

The fragment-only dataset is enough to validate the training loop;
for a full run point `--dataset_directory` at the complete DSEC-DET
dataset root.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from dagr.data.dsec_data import DSEC
from dagr.model.adaptive_sampler import (
    AdaptiveEventSampler,
    AdaptiveSamplerConfig,
    coverage_loss,
    count_parameters,
)


# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------

def pyg_data_to_sampler_inputs(data, time_window_us: int,
                               sensor_h: int, sensor_w: int, device: str):
    """Unpack a single PyG Data object into:
        events_dict : {'x','y','t','p'} CUDA tensors aligned with the sampler
        rgb         : [1, 3, H_img, W_img] float CUDA tensor in [0, 1]
        bboxes      : [M, 4] (x, y, w, h) in model coords on CUDA
    """
    pos = data.pos.to(device)
    events_dict = {
        'x': pos[:, 0].long(),
        'y': pos[:, 1].long(),
        # Normalize time to [0, 1] using the dataset's advertised window.
        't': data.t.to(device).float() / float(time_window_us),
        'p': data.x.view(-1).long().to(device),  # polarity stored as data.x
    }
    rgb = data.image.to(device).float() / 255.0   # [1, 3, H, W]
    # data.bbox columns per preprocess_detections: x, y, w, h, class
    if data.bbox.numel() > 0:
        bboxes = data.bbox[:, :4].to(device).float()
    else:
        bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
    return events_dict, rgb, bboxes


def batch_sampler_inputs(batch: Batch, time_window_us: int,
                         sensor_h: int, sensor_w: int, device: str):
    """Equivalent of the above for a PyG Batch — returns lists keyed
    by batch index. We don't collate events across the batch because
    the sampler operates per-frame."""
    per_frame = []
    for k in range(batch.num_graphs):
        single = batch.get_example(k)
        ev, rgb, bb = pyg_data_to_sampler_inputs(
            single, time_window_us, sensor_h, sensor_w, device,
        )
        per_frame.append((ev, rgb, bb))
    return per_frame


def bboxes_to_batched_tensor(per_frame_bboxes) -> torch.Tensor:
    """Stack (B-list of [M, 4]) into a single [sum M, 5] tensor with
    the leading column being the batch index, matching what
    coverage_loss() expects."""
    rows = []
    for b, bb in enumerate(per_frame_bboxes):
        if bb.numel() == 0:
            continue
        col = torch.full((bb.shape[0], 1), float(b),
                         dtype=bb.dtype, device=bb.device)
        rows.append(torch.cat([col, bb], dim=1))
    if not rows:
        return torch.zeros((0, 5), device=per_frame_bboxes[0].device
                           if per_frame_bboxes else 'cpu')
    return torch.cat(rows, dim=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[train-stage1] device={device}")

    # --- dataset -------------------------------------------------------------
    ds_root = Path(args.dataset_directory)
    train_ds = DSEC(
        root=ds_root, split="train", transform=None,
        num_us=-1, debug=False, min_bbox_height=0, min_bbox_diag=0,
        only_perfect_tracks=False,
    )
    print(f"[train-stage1] train samples: {len(train_ds)}")

    def collate(batch_list):
        return Batch.from_data_list(batch_list,
                                    follow_batch=['bbox', 'bbox0'])

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=True, drop_last=True,
    )

    # --- sampler -------------------------------------------------------------
    cfg = AdaptiveSamplerConfig(
        sensor_h=train_ds.height, sensor_w=train_ds.width,
        grid_stride=args.grid_stride, T=args.T_bins, K=args.K,
    )
    sampler = AdaptiveEventSampler(cfg).to(device).train()
    print(f"[train-stage1] sampler params: {count_parameters(sampler)}")

    opt = optim.AdamW(sampler.parameters(), lr=args.lr,
                      weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(loader), eta_min=1e-5,
    )

    # --- output --------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stage1_log.txt"
    log_f = open(log_path, "w")

    # --- loop ----------------------------------------------------------------
    global_step = 0
    for epoch in range(args.epochs):
        ep_loss = 0.0
        t0 = time.time()
        for batch in loader:
            per_frame = batch_sampler_inputs(
                batch, train_ds.time_window,
                train_ds.height, train_ds.width, device,
            )

            score_maps = []
            for events, rgb, _ in per_frame:
                sm = sampler.compute_score_map(events, rgb)   # [H, W]
                score_maps.append(sm)
            score_maps = torch.stack(score_maps, dim=0)        # [B, H, W]

            bboxes_batched = bboxes_to_batched_tensor(
                [pf[2] for pf in per_frame]
            )
            loss = coverage_loss(
                score_maps, bboxes_batched,
                sensor_h=train_ds.height, sensor_w=train_ds.width,
                bg_weight=args.bg_weight,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sampler.parameters(), 1.0)
            opt.step()
            sched.step()

            ep_loss += loss.item()
            global_step += 1
            if global_step % args.log_every == 0:
                msg = (f"ep={epoch} step={global_step} "
                       f"loss={loss.item():.4f} "
                       f"lr={sched.get_last_lr()[0]:.2e}")
                print(msg, flush=True)
                log_f.write(msg + "\n"); log_f.flush()

        ep_mean = ep_loss / max(1, len(loader))
        dt = time.time() - t0
        msg = (f"[epoch {epoch}] mean_loss={ep_mean:.4f} "
               f"({dt:.0f}s, {len(loader)} steps)")
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

        # Save checkpoint each epoch.
        ckpt_path = out_dir / f"sampler_stage1_epoch{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'state_dict': sampler.state_dict(),
            'cfg': cfg.__dict__,
        }, ckpt_path)
        print(f"  -> saved {ckpt_path}", flush=True)

    log_f.close()
    print(f"[train-stage1] done. Artifacts at {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_directory", required=True, type=str,
                    help="Root of the DSEC-DET dataset "
                         "(contains test/<sequence>/...).")
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grid_stride", type=int, default=8)
    ap.add_argument("--T_bins", type=int, default=8)
    ap.add_argument("--K", type=int, default=10000)
    ap.add_argument("--bg_weight", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
