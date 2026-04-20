"""Adaptive event sampling for DAGR.

The default DAGR pipeline caps raw events at a fixed n_nodes and keeps
the most recent K, which indiscriminately drops older events regardless
of how informative they are. Under high-density scenes (city traffic,
many moving objects) this hurts small-object recall and widens the
per-frame latency distribution.

This module learns a spatial importance heatmap conditioned on both a
voxelized event histogram and an RGB image, then selects K events via
top-K over per-event scores obtained by bilinear lookup. At training
time the selection is Gumbel-softened for differentiability; at
inference it is hard top-K.

Module size is tiny (<300k params). Grid resolution is a design arg;
default stride 8 in model coordinates gives a 40x27 heatmap, which is
coarse enough to regularize well while still localizing GT bboxes.

All coordinates are in DAGR *model space* (scale=2 applied, so width
= dataset_width / 2, height = cropped_height / 2). The caller must
pre-halve sensor-space event coordinates before invoking this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveSamplerConfig:
    # Model-space sensor resolution. DAGR-s-dsec uses scale=2 so these are
    # 640/2=320 and 430/2=215 (cropped from 480). For GenX320 native mode
    # the caller should adjust.
    sensor_h: int = 215
    sensor_w: int = 320
    # Heatmap resolution (grid_h = sensor_h // stride, etc.).
    grid_stride: int = 8
    # Number of temporal voxel bins per polarity.
    T: int = 8
    # Target number of events to keep per frame.
    K: int = 10000
    # Hidden dim for the scorer.
    hidden: int = 64
    # Gumbel temperature for soft top-K during training.
    gumbel_tau: float = 1.0


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class EventVoxelHistogram(nn.Module):
    """Discretize raw events into a (2T, H, W) float histogram. Two
    polarity channels, T temporal bins, spatial resolution matching
    the importance heatmap grid."""

    def __init__(self, cfg: AdaptiveSamplerConfig):
        super().__init__()
        self.T = cfg.T
        self.H = cfg.sensor_h // cfg.grid_stride
        self.W = cfg.sensor_w // cfg.grid_stride
        self.sh = cfg.grid_stride
        self.sw = cfg.grid_stride
        self.sensor_h = cfg.sensor_h
        self.sensor_w = cfg.sensor_w

    @torch.no_grad()
    def forward(self, events: dict) -> torch.Tensor:
        """Returns [2T, H, W].
        events must be a dict of CUDA tensors with 'x','y','t','p'.
          x, y: integer sensor coords in model space (ints in [0, W_sensor) / [0, H_sensor))
          t   : float in [0, 1] (pre-normalized per-frame temporal offset)
          p   : int in {-1, +1}
        """
        x, y, t, p = events['x'], events['y'], events['t'], events['p']
        device = x.device

        xi = (x.long() // self.sw).clamp(0, self.W - 1)
        yi = (y.long() // self.sh).clamp(0, self.H - 1)
        ti = (t.float() * self.T).long().clamp(0, self.T - 1)
        pi = (p.long() > 0).long()  # -1/+1 -> 0/1

        flat = (pi * self.T * self.H * self.W
                + ti * self.H * self.W
                + yi * self.W
                + xi)
        total = 2 * self.T * self.H * self.W
        voxel = torch.zeros(total, dtype=torch.float32, device=device)
        voxel.index_add_(
            0, flat, torch.ones_like(flat, dtype=torch.float32)
        )
        return voxel.view(2 * self.T, self.H, self.W)


class ImageContextEncoder(nn.Module):
    """Tiny CNN that extracts RGB context features aligned to the
    heatmap grid. Independent of the DAGR ResNet50 backbone so the
    sampler can be trained, ablated and deployed independently."""

    def __init__(self, out_ch: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(128, out_ch, 1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb: [B, 3, H_img, W_img] in [0, 1]. Returns [B, out_ch, H/8, W/8]."""
        return self.head(self.stem(rgb))


class ImportanceScorer(nn.Module):
    """Fuse event voxel + RGB context features into a single-channel
    spatial logit heatmap."""

    def __init__(self, event_ch: int, rgb_ch: int, hidden: int,
                 H: int, W: int):
        super().__init__()
        self.H, self.W = H, W
        self.event_proj = nn.Conv2d(event_ch, hidden, 1)
        self.rgb_proj = nn.Conv2d(rgb_ch, hidden, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, event_voxel: torch.Tensor,
                rgb_feat: torch.Tensor) -> torch.Tensor:
        # Event voxel is [B, 2T, H, W]; rgb_feat is [B, C, H', W'].
        if (rgb_feat.shape[-2], rgb_feat.shape[-1]) != (self.H, self.W):
            rgb_feat = F.interpolate(
                rgb_feat, size=(self.H, self.W),
                mode='bilinear', align_corners=False,
            )
        return self.fuse(
            torch.cat([self.event_proj(event_voxel),
                       self.rgb_proj(rgb_feat)], dim=1)
        ).squeeze(1)  # [B, H, W]


# ---------------------------------------------------------------------------
# Top-level sampler
# ---------------------------------------------------------------------------

class AdaptiveEventSampler(nn.Module):
    """Main entry point. forward() takes raw events + RGB and returns
    K selected events (plus score_map + per-selected-event score,
    useful for loss terms and visualization)."""

    def __init__(self, cfg: AdaptiveSamplerConfig = AdaptiveSamplerConfig()):
        super().__init__()
        self.cfg = cfg
        self.H = cfg.sensor_h // cfg.grid_stride
        self.W = cfg.sensor_w // cfg.grid_stride

        self.voxelizer = EventVoxelHistogram(cfg)
        self.img_encoder = ImageContextEncoder(out_ch=64)
        self.scorer = ImportanceScorer(
            event_ch=2 * cfg.T, rgb_ch=64, hidden=cfg.hidden,
            H=self.H, W=self.W,
        )

    # ---- helpers ----------------------------------------------------------

    def _per_event_score(self, events: dict,
                         score_map: torch.Tensor) -> torch.Tensor:
        """score_map: [H, W] logits. Returns [N] per-event score via
        bilinear lookup so the gradient w.r.t. score_map is dense
        (the discrete event x/y indices are treated as constants)."""
        xn = (events['x'].float() / self.cfg.sensor_w) * 2.0 - 1.0
        yn = (events['y'].float() / self.cfg.sensor_h) * 2.0 - 1.0
        grid = torch.stack([xn, yn], dim=-1)[None, None]  # [1,1,N,2]
        sampled = F.grid_sample(
            score_map[None, None], grid,
            mode='bilinear', align_corners=False,
            padding_mode='border',
        )
        return sampled.view(-1)  # [N]

    # ---- forward ----------------------------------------------------------

    def compute_score_map(self, events: dict,
                          rgb: torch.Tensor) -> torch.Tensor:
        """Produce the logit heatmap. Useful as a standalone
        differentiable path for Stage-1 coverage training."""
        voxel = self.voxelizer(events).unsqueeze(0)  # [1, 2T, H, W]
        rgb_f = self.img_encoder(rgb)                # [1, 64, h', w']
        return self.scorer(voxel, rgb_f).squeeze(0)  # [H, W]

    def forward(self, events: dict, rgb: torch.Tensor,
                hard: bool = True):
        """Select K events from `events` using image-conditioned
        importance scoring. Returns (sampled_events, scores_K, score_map).

        When len(events['x']) <= K, returns the events untouched
        (`sampled_events == events`).
        """
        score_map = self.compute_score_map(events, rgb)
        scores = self._per_event_score(events, score_map)  # [N]
        N = scores.shape[0]
        if N <= self.cfg.K:
            return events, scores, score_map

        if hard:
            idx = scores.topk(self.cfg.K).indices
        else:
            # Gumbel-Top-K. Straight-through: forward uses the hard
            # top-K, backward receives the soft perturbed scores.
            gumbel = -torch.log(
                -torch.log(torch.rand_like(scores).clamp(1e-20, 1 - 1e-20))
            )
            perturbed = (scores + gumbel) / self.cfg.gumbel_tau
            idx = perturbed.topk(self.cfg.K).indices

        sampled = {k: v[idx] for k, v in events.items()}
        return sampled, scores[idx], score_map


# ---------------------------------------------------------------------------
# Training losses
# ---------------------------------------------------------------------------

def coverage_loss(score_map: torch.Tensor,
                  bboxes_pixel: torch.Tensor,
                  sensor_h: int, sensor_w: int,
                  bg_weight: float = 0.1) -> torch.Tensor:
    """Stage-1 loss: push score_map high inside GT bboxes, low outside.

    score_map: [B, H, W] logits (or [H, W] if batch dim is collapsed).
    bboxes_pixel: [B*N, 5] array of (batch_idx, x, y, w, h) in model
                  coords, matching DAGR's data.bbox layout extended
                  with a leading batch index column. Pass an empty
                  tensor for frames with no GT.
    bg_weight: weight of the background (outside-box) BCE loss. 0.1
               keeps the network from collapsing to all-high.
    """
    if score_map.dim() == 2:
        score_map = score_map.unsqueeze(0)
    B, H, W = score_map.shape

    # Build a binary target mask: 1 inside any GT bbox, 0 outside.
    target = torch.zeros_like(score_map)
    if bboxes_pixel.numel() > 0:
        # bboxes_pixel columns: batch_idx, x, y, w, h
        scale_x = W / sensor_w
        scale_y = H / sensor_h
        for row in bboxes_pixel:
            b = int(row[0].item())
            x = float(row[1].item())
            y = float(row[2].item())
            w = float(row[3].item())
            h = float(row[4].item())
            if w <= 0 or h <= 0:
                continue
            gx0 = max(0, int(x * scale_x))
            gy0 = max(0, int(y * scale_y))
            gx1 = min(W, max(gx0 + 1, int((x + w) * scale_x)))
            gy1 = min(H, max(gy0 + 1, int((y + h) * scale_y)))
            target[b, gy0:gy1, gx0:gx1] = 1.0

    # Weighted BCE: full weight on fg pixels, bg_weight on bg pixels.
    pos_mask = target > 0.5
    neg_mask = ~pos_mask
    pos_loss = F.binary_cross_entropy_with_logits(
        score_map[pos_mask], target[pos_mask], reduction='mean'
    ) if pos_mask.any() else score_map.new_zeros(())
    neg_loss = F.binary_cross_entropy_with_logits(
        score_map[neg_mask], target[neg_mask], reduction='mean'
    ) if neg_mask.any() else score_map.new_zeros(())
    return pos_loss + bg_weight * neg_loss


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
