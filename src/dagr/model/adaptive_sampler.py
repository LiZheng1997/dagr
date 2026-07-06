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
    # Whether to condition the score map on RGB image features.
    use_image: bool = True
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

        xi = torch.div(x.long(), self.sw, rounding_mode='floor').clamp(0, self.W - 1)
        yi = torch.div(y.long(), self.sh, rounding_mode='floor').clamp(0, self.H - 1)
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
    """Fuse event voxel and optional RGB context features into a
    single-channel spatial logit heatmap."""

    def __init__(self, event_ch: int, rgb_ch: int, hidden: int,
                 H: int, W: int):
        super().__init__()
        self.H, self.W = H, W
        self.event_proj = nn.Conv2d(event_ch, hidden, 1)
        self.rgb_proj = nn.Conv2d(rgb_ch, hidden, 1) if rgb_ch > 0 else None
        fuse_ch = hidden * (2 if self.rgb_proj is not None else 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_ch, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, event_voxel: torch.Tensor,
                rgb_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Event voxel is [B, 2T, H, W]; rgb_feat is [B, C, H', W'].
        parts = [self.event_proj(event_voxel)]
        if self.rgb_proj is not None:
            if rgb_feat is None:
                raise ValueError("rgb_feat is required when RGB conditioning is enabled")
            if (rgb_feat.shape[-2], rgb_feat.shape[-1]) != (self.H, self.W):
                rgb_feat = F.interpolate(
                    rgb_feat, size=(self.H, self.W),
                    mode='bilinear', align_corners=False,
                )
            parts.append(self.rgb_proj(rgb_feat))
        return self.fuse(torch.cat(parts, dim=1)).squeeze(1)  # [B, H, W]


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
        self.img_encoder = ImageContextEncoder(out_ch=64) if cfg.use_image else None
        self.scorer = ImportanceScorer(
            event_ch=2 * cfg.T,
            rgb_ch=64 if cfg.use_image else 0,
            hidden=cfg.hidden,
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
                          rgb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Produce the logit heatmap. Useful as a standalone
        differentiable path for Stage-1 coverage training."""
        voxel = self.voxelizer(events).unsqueeze(0)  # [1, 2T, H, W]
        rgb_f = self.img_encoder(rgb) if self.img_encoder is not None else None
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

def _coverage_target(score_map: torch.Tensor,
                     bboxes_pixel: torch.Tensor,
                     sensor_h: int, sensor_w: int):
    if score_map.dim() == 2:
        score_map = score_map.unsqueeze(0)
    B, H, W = score_map.shape

    target = torch.zeros_like(score_map)
    if bboxes_pixel.numel() > 0:
        scale_x = W / sensor_w
        scale_y = H / sensor_h
        for row in bboxes_pixel:
            b = int(row[0].item())
            if b < 0 or b >= B:
                continue
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
    return score_map, target


def _gaussian_target(score_map: torch.Tensor,
                     bboxes_pixel: torch.Tensor,
                     sensor_h: int, sensor_w: int,
                     sigma_scale: float = 0.30,
                     min_sigma: float = 1.0) -> torch.Tensor:
    score_map, target = _coverage_target(
        score_map, torch.zeros((0, 5), device=score_map.device),
        sensor_h, sensor_w
    )
    B, H, W = score_map.shape
    if bboxes_pixel.numel() == 0:
        return score_map, target

    scale_x = W / sensor_w
    scale_y = H / sensor_h
    yy = torch.arange(H, device=score_map.device).float()[:, None]
    xx = torch.arange(W, device=score_map.device).float()[None, :]
    for row in bboxes_pixel:
        b = int(row[0].item())
        if b < 0 or b >= B:
            continue
        x = float(row[1].item())
        y = float(row[2].item())
        w = float(row[3].item())
        h = float(row[4].item())
        if w <= 0 or h <= 0:
            continue
        cx = (x + 0.5 * w) * scale_x
        cy = (y + 0.5 * h) * scale_y
        sigma_x = max(min_sigma, w * scale_x * sigma_scale)
        sigma_y = max(min_sigma, h * scale_y * sigma_scale)
        g = torch.exp(
            -(((xx - cx) ** 2) / (2 * sigma_x ** 2)
              + ((yy - cy) ** 2) / (2 * sigma_y ** 2))
        )
        target[b] = torch.maximum(target[b], g)
    return score_map, target


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
    score_map, target = _coverage_target(
        score_map, bboxes_pixel, sensor_h, sensor_w
    )

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


def gaussian_bce_coverage_loss(score_map: torch.Tensor,
                               bboxes_pixel: torch.Tensor,
                               sensor_h: int, sensor_w: int,
                               bg_weight: float = 0.3,
                               sigma_scale: float = 0.30,
                               min_sigma: float = 1.0) -> torch.Tensor:
    """Soft Gaussian object target with weighted BCE."""
    score_map, target = _gaussian_target(
        score_map, bboxes_pixel, sensor_h, sensor_w, sigma_scale, min_sigma
    )
    weights = torch.ones_like(target)
    weights = torch.where(target > 0.05, weights, weights * bg_weight)
    loss = F.binary_cross_entropy_with_logits(
        score_map, target, reduction='none'
    )
    return (loss * weights).mean()


def centernet_heatmap_loss(score_map: torch.Tensor,
                           bboxes_pixel: torch.Tensor,
                           sensor_h: int, sensor_w: int,
                           sigma_scale: float = 0.30,
                           min_sigma: float = 1.0,
                           alpha: float = 2.0,
                           beta: float = 4.0) -> torch.Tensor:
    """CenterNet-style modified focal loss on Gaussian object centers."""
    score_map, target = _gaussian_target(
        score_map, bboxes_pixel, sensor_h, sensor_w, sigma_scale, min_sigma
    )
    pred = torch.sigmoid(score_map).clamp(1e-4, 1 - 1e-4)
    pos_inds = target.eq(1.0)
    neg_inds = target.lt(1.0)
    neg_weights = (1.0 - target).pow(beta)

    pos_loss = torch.log(pred) * (1.0 - pred).pow(alpha) * pos_inds
    neg_loss = (
        torch.log(1.0 - pred) * pred.pow(alpha) * neg_weights * neg_inds
    )
    num_pos = pos_inds.float().sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def _topk_background_loss(score_map: torch.Tensor,
                          neg_mask: torch.Tensor,
                          topk_bg_frac: float) -> torch.Tensor:
    topk_losses = []
    if topk_bg_frac > 0:
        for b in range(score_map.shape[0]):
            logits_b = score_map[b][neg_mask[b]]
            if logits_b.numel() == 0:
                continue
            k = max(1, int(logits_b.numel() * topk_bg_frac))
            k = min(k, logits_b.numel())
            top_logits = logits_b.topk(k).values
            topk_losses.append(F.softplus(top_logits).mean())
    return (
        torch.stack(topk_losses).mean()
        if topk_losses else score_map.new_zeros(())
    )


def topk_coverage_loss(score_map: torch.Tensor,
                       bboxes_pixel: torch.Tensor,
                       sensor_h: int, sensor_w: int,
                       bg_weight: float = 0.3,
                       topk_bg_weight: float = 0.1,
                       topk_bg_frac: float = 0.10) -> torch.Tensor:
    """Weighted BCE plus an explicit penalty on highest-scoring background."""
    score_map, target = _coverage_target(
        score_map, bboxes_pixel, sensor_h, sensor_w
    )
    pos_mask = target > 0.5
    neg_mask = ~pos_mask
    pos_loss = F.binary_cross_entropy_with_logits(
        score_map[pos_mask], target[pos_mask], reduction='mean'
    ) if pos_mask.any() else score_map.new_zeros(())
    neg_loss = F.binary_cross_entropy_with_logits(
        score_map[neg_mask], target[neg_mask], reduction='mean'
    ) if neg_mask.any() else score_map.new_zeros(())
    topk_loss = _topk_background_loss(score_map, neg_mask, topk_bg_frac)
    return pos_loss + bg_weight * neg_loss + topk_bg_weight * topk_loss


def focal_topk_coverage_loss(score_map: torch.Tensor,
                             bboxes_pixel: torch.Tensor,
                             sensor_h: int, sensor_w: int,
                             bg_weight: float = 0.3,
                             focal_gamma: float = 2.0,
                             topk_bg_weight: float = 0.2,
                             topk_bg_frac: float = 0.10) -> torch.Tensor:
    """Coverage loss variant for sparse selection.

    The base BCE term learns the GT-box mask. The focal background term
    emphasizes high-scoring false positives, and the top-k background term
    directly suppresses the worst background cells. That matches the sampler's
    downstream top-K behavior better than averaging over all background cells.
    """
    score_map, target = _coverage_target(
        score_map, bboxes_pixel, sensor_h, sensor_w
    )
    pos_mask = target > 0.5
    neg_mask = ~pos_mask

    pos_loss = F.binary_cross_entropy_with_logits(
        score_map[pos_mask], target[pos_mask], reduction='mean'
    ) if pos_mask.any() else score_map.new_zeros(())

    if neg_mask.any():
        neg_logits = score_map[neg_mask]
        neg_bce = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits), reduction='none'
        )
        neg_prob = torch.sigmoid(neg_logits)
        neg_loss = (neg_prob.pow(focal_gamma) * neg_bce).mean()
    else:
        neg_loss = score_map.new_zeros(())

    topk_loss = _topk_background_loss(score_map, neg_mask, topk_bg_frac)

    return pos_loss + bg_weight * neg_loss + topk_bg_weight * topk_loss


def _event_bbox_target(events: dict,
                       bboxes_pixel: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    x = events['x'].float()
    y = events['y'].float()
    target = torch.zeros_like(x, dtype=torch.bool)
    box_masks = []

    if bboxes_pixel.numel() == 0:
        return target, box_masks

    for row in bboxes_pixel:
        bx = float(row[0].item())
        by = float(row[1].item())
        bw = float(row[2].item())
        bh = float(row[3].item())
        if bw <= 0 or bh <= 0:
            continue
        mask = (
            (x >= bx) & (x < bx + bw) &
            (y >= by) & (y < by + bh)
        )
        target |= mask
        if mask.any():
            box_masks.append(mask)
    return target, box_masks


def event_selection_coverage_loss(sampler: AdaptiveEventSampler,
                                  score_map: torch.Tensor,
                                  events: dict,
                                  bboxes_pixel: torch.Tensor,
                                  bg_weight: float = 0.05,
                                  topk_bg_weight: float = 0.5,
                                  topk_bg_frac: float = 0.10,
                                  box_cover_weight: float = 0.5) -> torch.Tensor:
    """Event-level Stage-1 loss aligned with hard top-K selection.

    The dense grid losses train an object heatmap, but deployment selects
    individual events. This loss samples the score map at every event location,
    pushes in-box event scores up, suppresses high-scoring background events,
    and adds a per-box max-score term so each annotated object has at least one
    competitive event.
    """
    scores = sampler._per_event_score(events, score_map)
    target, box_masks = _event_bbox_target(events, bboxes_pixel)
    neg_mask = ~target
    zero = score_map.sum() * 0.0

    pos_loss = (
        F.softplus(-scores[target]).mean()
        if target.any() else zero
    )
    neg_loss = (
        F.softplus(scores[neg_mask]).mean()
        if neg_mask.any() else zero
    )

    topk_loss = zero
    if neg_mask.any() and topk_bg_frac > 0:
        neg_scores = scores[neg_mask]
        k = max(1, int(neg_scores.numel() * topk_bg_frac))
        k = min(k, neg_scores.numel())
        topk_loss = F.softplus(neg_scores.topk(k).values).mean()

    box_cover_loss = zero
    if box_masks:
        box_terms = [F.softplus(-scores[mask].max()) for mask in box_masks]
        box_cover_loss = torch.stack(box_terms).mean()

    return (
        pos_loss
        + bg_weight * neg_loss
        + topk_bg_weight * topk_loss
        + box_cover_weight * box_cover_loss
    )


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
