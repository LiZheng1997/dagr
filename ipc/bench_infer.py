#!/usr/bin/env python3
"""Standalone latency benchmark for the live inference path (no ROS, no socket).

Loads the model exactly like infer_server.main(), then times infer_once() on
synthetic frames in the chosen sensor convention. Reports per-call wall time
with CUDA synchronised, so the number is what the ROS node would see per frame
minus IPC overhead.

Example (inside the dagr-orin container):
  cd /opt/dagr && python3 ipc/bench_infer.py --sensor_mode genx320 \
      --genx320_fit letterbox --iters 60 --events 10000
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer_server as S  # noqa: E402
import torch  # noqa: E402


def synth_frame(w, h, rng):
    """Low-light-ish frame with a few bright blobs so the CNN has structure."""
    img = (rng.normal(28, 6, size=(h, w, 3))).clip(0, 255).astype(np.uint8)
    for _ in range(4):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = int(rng.integers(8, 40))
        yy, xx = np.ogrid[:h, :w]
        m = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
        img[m] = rng.integers(120, 230)
    return img


def synth_events(w, h, n, rng, window_ns=50_000_000):
    t1 = int(time.time_ns())
    t = np.sort(rng.integers(t1 - window_ns, t1, size=n)).astype(np.int64)
    return {
        'x': rng.integers(0, w, size=n).astype(np.int32),
        'y': rng.integers(0, h, size=n).astype(np.int32),
        't_ns': t,
        'p': rng.integers(0, 2, size=n).astype(np.int8),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/dagr-s-dsec.yaml')
    ap.add_argument('--checkpoint', default='data/dagr_s_50.pth')
    ap.add_argument('--img_net', default='resnet50')
    ap.add_argument('--n_nodes', type=int, default=10000)
    ap.add_argument('--dataset_directory', default='data/DSEC_fragment')
    ap.add_argument('--sensor_mode', choices=['dsec', 'genx320'], default='genx320')
    ap.add_argument('--genx320_fit', choices=['letterbox', 'squash', 'crop_top'], default='letterbox')
    ap.add_argument('--events', type=int, default=10000, help='synthetic events per frame')
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--warmup', type=int, default=8)
    ap.add_argument('--json', default='', help='write summary JSON here')
    cli = ap.parse_args()

    S.set_sensor_mode(cli.sensor_mode)
    S.set_genx320_fit(cli.genx320_fit)
    S.set_max_events_per_frame(cli.n_nodes)
    w, h = (640, 480) if cli.sensor_mode == 'dsec' else (320, 320)
    print(f"[bench] sensor_mode={cli.sensor_mode} fit={cli.genx320_fit} frame={w}x{h} "
          f"events/frame={cli.events} fp16={S._FP16} trt_cnn={os.environ.get('DAGR_TRT_CNN', '0')}",
          flush=True)

    args = S.build_args(cli)
    t0 = time.perf_counter()
    model = S.load_model(args)
    print(f"[bench] model loaded in {time.perf_counter() - t0:.1f} s", flush=True)

    rng = np.random.default_rng(0)
    frames = [synth_frame(w, h, rng) for _ in range(8)]
    evs = [synth_events(w, h, cli.events, rng) for _ in range(8)]

    with torch.no_grad():
        for i in range(cli.warmup):
            S.infer_once(model, frames[i % 8], evs[i % 8])
        torch.cuda.synchronize()
        times = []
        ndet = []
        for i in range(cli.iters):
            torch.cuda.synchronize()
            t = time.perf_counter()
            det = S.infer_once(model, frames[i % 8], evs[i % 8])
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000.0)
            ndet.append(int(len(det['boxes'])))
    times = np.array(times)
    q = lambda p: float(np.percentile(times, p))
    summary = {
        'sensor_mode': cli.sensor_mode, 'fit': cli.genx320_fit, 'events_per_frame': cli.events,
        'fp16': bool(S._FP16), 'trt_cnn': os.environ.get('DAGR_TRT_CNN', '0'),
        'iters': cli.iters, 'ms_mean': float(times.mean()), 'ms_p50': q(50), 'ms_p95': q(95),
        'ms_max': float(times.max()), 'fps_at_p50': 1000.0 / q(50), 'mean_detections': float(np.mean(ndet)),
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
    }
    print("[bench] " + json.dumps(summary), flush=True)
    print(f"[bench] per-call ms: mean {summary['ms_mean']:.1f}  p50 {summary['ms_p50']:.1f}  "
          f"p95 {summary['ms_p95']:.1f}  max {summary['ms_max']:.1f}  -> {summary['fps_at_p50']:.1f} fps at p50",
          flush=True)
    if cli.json:
        Path(cli.json).write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
