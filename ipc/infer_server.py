"""DAGR inference daemon.

Runs inside the `dagr` conda env (py3.9 + torch 1.11 + PyG 2.1) and listens
on a Unix Domain Socket. Clients hand over a single-frame (image + event
window); the daemon returns detections.

Launch:
    source /home/lz/miniconda3/etc/profile.d/conda.sh && conda activate dagr
    cd /home/lz/Documents/Events-Perception/dagr
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 WANDB_MODE=offline \
    python ipc/infer_server.py \
        --checkpoint data/dagr_s_50.pth \
        --config config/dagr-s-dsec.yaml \
        --img_net resnet50 \
        --n_nodes 20000
"""
import argparse
import os
import pickle
import socket
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, Batch

# dagr must be importable (installed editable as `dagr`)
from dagr.data.utils import to_data
from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA
from dagr.utils.buffers import format_data
from dagr.utils.args import FLAGS as DAGR_FLAGS

IPC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(IPC_DIR))
from protocol import (  # noqa: E402
    SOCKET_PATH,
    MSG_INFER_REQ, MSG_PING, MSG_PONG,
    DET_DTYPE,
    pack_header, recv_header, _recv_exact,
    decode_infer_req, encode_infer_reply_ok, encode_infer_reply_err,
)


# ---------- arg/config plumbing ----------------------------------------------

def build_args(cli):
    """Produce an args Namespace compatible with the DAGR constructor.

    We reuse dagr.utils.args.FLAGS which parses --config YAML + CLI overrides.
    """
    sys.argv = [
        "infer_server",
        "--config", cli.config,
        "--checkpoint", cli.checkpoint,
        "--output_directory", "/tmp/dagr_infer_server_out",
        "--batch_size", "1",
        "--dataset_directory", cli.dataset_directory,
        "--use_image",
        "--img_net", cli.img_net,
        "--n_nodes", str(cli.n_nodes),
        "--no_eval",
    ]
    os.makedirs("/tmp/dagr_infer_server_out", exist_ok=True)
    return DAGR_FLAGS()


# ---------- input shaping -----------------------------------------------------

# DSEC default dataset dimensions after scale=2: width=320, height=215.
# These are the dims the model's LUT and internal strides are computed for.
MODEL_WIDTH = 320
MODEL_HEIGHT = 215
SCALE = 2                     # image is cropped/resized from (scale*H) rows
CROPPED_HEIGHT = SCALE * MODEL_HEIGHT  # 430
TIME_WINDOW_NS = 1_000_000_000         # 1 second in ns, matches DSEC config

# Runtime sensor mode — set via CLI flag, consumed by preprocess_*.
#   "dsec"     : events/image live in 640x480 sensor space (DSEC training)
#   "genx320"  : events/image live in 320x320 sensor space (GenX320, route a)
_SENSOR_MODE = "dsec"

# Hard cap on events per frame. DSEC bags can push 500k events per 50 ms
# packet which would OOM the 2070 Max-Q during event graph construction
# (~1.3 GiB just for edge tensors). The DSEC dataset loader does the same
# subsampling via args.n_nodes. Override via --n_nodes CLI flag.
_MAX_EVENTS_PER_FRAME = 10000


def set_sensor_mode(mode: str) -> None:
    global _SENSOR_MODE
    if mode not in ("dsec", "genx320"):
        raise ValueError(f"unknown sensor_mode {mode!r}")
    _SENSOR_MODE = mode


def set_max_events_per_frame(n: int) -> None:
    global _MAX_EVENTS_PER_FRAME
    _MAX_EVENTS_PER_FRAME = int(n)


def preprocess_image(bgr: np.ndarray) -> torch.Tensor:
    """Mirror dsec_data.DSEC.preprocess_image for live frames.

    Input:  numpy BGR image, any HxWx3 shape.
    Output: torch.float tensor 1x3xHxW with H=MODEL_HEIGHT, W=MODEL_WIDTH.

    In ``dsec`` mode the input is expected to be 640x480 (or taller) and gets
    top-cropped to 430 rows then bilinearly resized to 320x215.

    In ``genx320`` mode the input is 320x320 already near the model width, so
    we skip the crop-then-downsample and just resize straight to 320x215.
    """
    import cv2
    h, w = bgr.shape[:2]
    if _SENSOR_MODE == "dsec":
        if h < CROPPED_HEIGHT:
            pad = np.zeros((CROPPED_HEIGHT - h, w, 3), dtype=bgr.dtype)
            bgr = np.vstack([bgr, pad])
        bgr = bgr[:CROPPED_HEIGHT]
    resized = cv2.resize(bgr, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_CUBIC)
    t = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
    return t


def preprocess_events(events: dict) -> dict:
    """Mimic DSEC.preprocess_events for a raw event window.

    Input: dict with x, y, t_ns, p numpy arrays.
    Output: dict with x, y, t (int32 relative), p (int8 -1/+1).

    Sensor-mode branches:
      - ``dsec``:    events come in 640x480 sensor coords, we crop y<430
                     and halve xy to 320x215 model coords.
      - ``genx320``: events come in 320x320 sensor coords, we crop y<215
                     (keep top) and pass xy straight through (model is
                     already at native sensor resolution in width).
    """
    x = np.asarray(events['x']).astype(np.int16)
    y = np.asarray(events['y']).astype(np.int16)
    t_ns = np.asarray(events['t_ns']).astype(np.int64)
    p = np.asarray(events['p']).astype(np.int8)

    if _SENSOR_MODE == "dsec":
        mask = y < CROPPED_HEIGHT
        x = x[mask]; y = y[mask]; t_ns = t_ns[mask]; p = p[mask]
    else:  # genx320
        mask = y < MODEL_HEIGHT
        x = x[mask]; y = y[mask]; t_ns = t_ns[mask]; p = p[mask]

    # Enforce the per-frame event cap (same role as args.n_nodes in the
    # training dataset). Keep the most recent events so the temporal
    # window stays tight and graph edges remain meaningful.
    if len(x) > _MAX_EVENTS_PER_FRAME:
        tail = len(x) - _MAX_EVENTS_PER_FRAME
        x = x[tail:]
        y = y[tail:]
        t_ns = t_ns[tail:]
        p = p[tail:]

    # Pack polarity to {-1, +1}
    p = (2 * p - 1).astype(np.int8)

    if len(t_ns) == 0:
        return {'x': x, 'y': y, 't': np.zeros(0, dtype=np.int32), 'p': p}

    # Relative time in integer units of TIME_WINDOW_NS scale.
    t_last = int(t_ns[-1])
    t_rel_ns = t_ns - t_last + TIME_WINDOW_NS
    t_us = (t_rel_ns // 1000).astype(np.int32)

    if _SENSOR_MODE == "dsec":
        x = (x // 2).astype(np.int16)
        y = (y // 2).astype(np.int16)
    # in genx320 mode, events already live in (MODEL_WIDTH, MODEL_HEIGHT)-compatible coords

    return {'x': x, 'y': y, 't': t_us, 'p': p.reshape(-1, 1)}


def build_sample(image_bgr: np.ndarray, events: dict) -> Data:
    """Assemble a torch_geometric Data object that the model can consume."""
    img = preprocess_image(image_bgr)
    ev = preprocess_events(events)
    data = to_data(
        x=ev['x'], y=ev['y'], t=ev['t'], p=ev['p'],
        bbox=np.zeros((0, 5), dtype=np.float32),
        bbox0=np.zeros((0, 5), dtype=np.float32),
        t0=0, t1=TIME_WINDOW_NS // 1000,
        width=MODEL_WIDTH, height=MODEL_HEIGHT,
        time_window=TIME_WINDOW_NS // 1000,
        image=img, sequence="live",
    )
    return data


# ---------- main loop ---------------------------------------------------------

def load_model(args):
    torch.manual_seed(42)
    torch_geometric.seed.seed_everything(42)

    # The DAGR ctor needs height/width; use our fixed model dims.
    model = DAGR(args, height=MODEL_HEIGHT, width=MODEL_WIDTH)
    model = model.cuda()
    ema = ModelEMA(model)

    ckpt = torch.load(args.checkpoint, map_location="cuda")
    ema.ema.load_state_dict(ckpt['ema'])
    ema.ema.cache_luts(radius=args.radius, height=MODEL_HEIGHT, width=MODEL_WIDTH)
    ema.ema.eval()

    # FP16 is driven at inference time via torch.cuda.amp.autocast in
    # infer_once (safer than model.half() since the custom PyG / ev_graph
    # CUDA ops may not implement half dtype). No load-time change needed.

    # TRT drop-in for the CNN branch. When DAGR_TRT_CNN=1 and the engine
    # file exists we replace model.backbone.net with a TrtCnn module
    # exposing the same (features, outputs) contract as HookModule.
    try:
        from dagr.model.networks.trt_cnn import maybe_install_trt_cnn
        maybe_install_trt_cnn(ema.ema)
    except Exception as e:
        print(f"[infer_server] TRT CNN install skipped: {e}", flush=True)

    try:
        from dagr.model.networks.trt_head import maybe_install_trt_head
        # GNNHead.forward reads self.output_sizes on first call; make sure
        # it's populated before the TRT head swap so the caller provides
        # correctly-sized features to the engine.
        ema.ema.head.output_sizes = ema.ema.backbone.get_output_sizes()
        maybe_install_trt_head(ema.ema)
    except Exception as e:
        print(f"[infer_server] TRT head install skipped: {e}", flush=True)

    if os.environ.get("DAGR_PROFILE_SUB", "0") == "1":
        _install_submodule_hooks(ema.ema)

    compile_mode = os.environ.get("DAGR_COMPILE", "0")
    if compile_mode != "0":
        # Several compile backends to pick from:
        #   reduce-overhead (default_mode for speed) — uses inductor +
        #     CUDA graphs + triton. Memory-heavy; OOM'd on the full model.
        #   aot_eager — AOT autograd + eager exec. Modest python-dispatch
        #     savings (~5%), no triton codegen, much less RAM.
        #   default — inductor without CUDA graphs. Needs triton.
        backend_map = {
            "1":              ("reduce-overhead", None),
            "reduce":         ("reduce-overhead", None),
            "aot_eager":      (None,              "aot_eager"),
            "default":        ("default",         None),
        }
        mode_arg, backend_arg = backend_map.get(
            compile_mode, ("default", None)
        )
        print(f"[infer_server] applying torch.compile(mode={mode_arg}, "
              f"backend={backend_arg}) ...", flush=True)
        kwargs = dict(dynamic=False, fullgraph=False)
        if mode_arg is not None:
            kwargs["mode"] = mode_arg
        if backend_arg is not None:
            kwargs["backend"] = backend_arg
        ema.ema = torch.compile(ema.ema, **kwargs)

    return ema.ema


# ---- sub-module profiling via forward hooks --------------------------------
#
# Accumulators for per-submodule CUDA-event timings. Populated by forward
# hooks registered when DAGR_PROFILE_SUB=1. Reported every PROFILE_EVERY
# inferences. We time four slots:
#   cnn     : backbone.net (ResNet50 + FPN-like feat extraction)
#   graph   : backbone.events_to_graph (EV_TGN custom CUDA op)
#   backbone: whole backbone (Net) — so gnn_fusion = backbone - cnn - graph
#   head    : yolox head (CNNHead for the image path, GNNHead for events)
_hook_accum_ms = {"cnn": 0.0, "graph": 0.0, "backbone": 0.0, "head": 0.0}
_hook_accum_n = 0
# Per-call CUDA events held while the submodule is executing, keyed by slot.
_hook_events = {}


def _make_pre_hook(slot):
    def _pre(_module, _inputs):
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_stop = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        _hook_events[slot] = (ev_start, ev_stop)
    return _pre


def _make_post_hook(slot):
    def _post(_module, _inputs, _output):
        ev_start, ev_stop = _hook_events[slot]
        ev_stop.record()
        # Don't sync here — let the outer per-call sync do it once.
        _hook_events[slot] = (ev_start, ev_stop, "pending")
    return _post


def _install_submodule_hooks(model):
    # Four coarse buckets that must always exist so the reporter can compute
    # gnn_fusion = backbone - cnn - graph.
    targets = {
        "cnn":      model.backbone.net,
        "graph":    model.backbone.events_to_graph,
        "backbone": model.backbone,
        "head":     model.head,
    }
    # When DAGR_PROFILE_DEEP=1, also hook every individual GNN layer / pool /
    # attr module inside the backbone. Pricier (more CUDA events per frame)
    # but reveals WHERE the ~15 ms of gnn_fusion is spent.
    if os.environ.get("DAGR_PROFILE_DEEP", "0") == "1":
        bb = model.backbone
        for name in ("edge_attrs", "conv_block1",
                     "pool1", "layer2", "pool2",
                     "layer3", "pool3", "layer4",
                     "pool4", "layer5"):
            mod = getattr(bb, name, None)
            if mod is not None:
                targets[f"bb.{name}"] = mod
    for slot, module in targets.items():
        module.register_forward_pre_hook(_make_pre_hook(slot))
        module.register_forward_hook(_make_post_hook(slot))
        _hook_accum_ms.setdefault(slot, 0.0)
    print(f"[infer_server] sub-module hooks installed: {list(targets)}", flush=True)


def _collect_submodule_timings():
    """After a cuda.synchronize(), read elapsed_time for each slot."""
    global _hook_accum_n
    if not _hook_events:
        return
    for slot, tup in list(_hook_events.items()):
        if len(tup) == 3:
            ev_start, ev_stop, _ = tup
            _hook_accum_ms[slot] += ev_start.elapsed_time(ev_stop)
    _hook_accum_n += 1
    _hook_events.clear()


def _maybe_report_submodule():
    global _hook_accum_n
    if _hook_accum_n < PROFILE_EVERY:
        return
    n = _hook_accum_n
    cnn   = _hook_accum_ms["cnn"]      / n
    graph = _hook_accum_ms["graph"]    / n
    bb    = _hook_accum_ms["backbone"] / n
    head  = _hook_accum_ms["head"]     / n
    gnn_fusion = max(0.0, bb - cnn - graph)
    print(
        f"[infer_server sub-profile] n={n} "
        f"cnn={cnn:6.2f}ms graph={graph:6.2f}ms gnn_fusion={gnn_fusion:6.2f}ms "
        f"head={head:6.2f}ms (backbone_total={bb:6.2f}ms)",
        flush=True,
    )
    # Deep breakdown: list every backbone submodule we registered (bb.*).
    deep_keys = sorted(k for k in _hook_accum_ms if k.startswith("bb."))
    if deep_keys:
        pairs = [f"{k[3:]}={_hook_accum_ms[k]/n:5.2f}" for k in deep_keys]
        print(f"[infer_server deep-profile] n={n} " + " ".join(pairs), flush=True)
    for k in _hook_accum_ms:
        _hook_accum_ms[k] = 0.0
    _hook_accum_n = 0


_INFER_CALL_COUNT = 0

# Accumulators (ms) for coarse stage profiling. Reported every
# PROFILE_EVERY calls via _maybe_report_profile. Enable by setting
# env DAGR_PROFILE=1.
_PROFILE = os.environ.get("DAGR_PROFILE", "0") == "1"
_FP16 = os.environ.get("DAGR_FP16", "0") == "1"
PROFILE_EVERY = 20
_acc_pre_ms = 0.0
_acc_fwd_ms = 0.0
_acc_post_ms = 0.0
_acc_n = 0


def _maybe_report_profile():
    global _acc_pre_ms, _acc_fwd_ms, _acc_post_ms, _acc_n
    if not _PROFILE or _acc_n < PROFILE_EVERY:
        return
    tot = _acc_pre_ms + _acc_fwd_ms + _acc_post_ms
    print(
        f"[infer_server profile] n={_acc_n} "
        f"pre={_acc_pre_ms/_acc_n:6.2f}ms "
        f"fwd={_acc_fwd_ms/_acc_n:6.2f}ms "
        f"post={_acc_post_ms/_acc_n:6.2f}ms "
        f"total={tot/_acc_n:6.2f}ms",
        flush=True,
    )
    _acc_pre_ms = _acc_fwd_ms = _acc_post_ms = 0.0
    _acc_n = 0


@torch.no_grad()
def infer_once(model, image_bgr, events):
    # Profiling splits wall time into three stages:
    #   pre  = CPU side (build_sample + Batch + .cuda() queue + format_data),
    #          measured with perf_counter after a cuda sync (so H2D is done).
    #   fwd  = GPU forward, measured with cuda.Event start/stop.
    #   post = output conversion + del + maybe empty_cache.
    global _INFER_CALL_COUNT, _acc_pre_ms, _acc_fwd_ms, _acc_post_ms, _acc_n

    if _PROFILE:
        t_pre0 = time.perf_counter()

    data = build_sample(image_bgr, events)
    batch = Batch.from_data_list([data], follow_batch=['bbox', 'bbox0'])
    batch = batch.cuda(non_blocking=True)
    batch = format_data(batch)

    if _PROFILE:
        torch.cuda.synchronize()  # finish the H2D + format_data ops
        t_pre1 = time.perf_counter()
        ev_fwd0 = torch.cuda.Event(enable_timing=True)
        ev_fwd1 = torch.cuda.Event(enable_timing=True)
        ev_fwd0.record()

    if _FP16:
        # Mixed-precision inference. Wraps model.forward in autocast so
        # supported ops run in fp16 on Tensor Cores while untouched ops
        # (notably the custom ev_graph / PyG pieces that don't implement
        # half) stay fp32. No input dtype change needed.
        with torch.cuda.amp.autocast(dtype=torch.float16):
            detections, _ = model(batch.clone())
    else:
        detections, _ = model(batch.clone())

    if _PROFILE:
        ev_fwd1.record()
        torch.cuda.synchronize()
        t_post0 = time.perf_counter()

    out = detections[0]

    del batch, data
    _INFER_CALL_COUNT += 1
    if _INFER_CALL_COUNT % 5 == 0:
        torch.cuda.empty_cache()

    if _PROFILE:
        t_post1 = time.perf_counter()
        _acc_pre_ms += (t_pre1 - t_pre0) * 1000
        _acc_fwd_ms += ev_fwd0.elapsed_time(ev_fwd1)
        _acc_post_ms += (t_post1 - t_post0) * 1000
        _acc_n += 1
        _maybe_report_profile()
        # The outer sync already happened before t_post0, so submodule events
        # are all finished — safe to read their elapsed_time.
        _collect_submodule_timings()
        _maybe_report_submodule()
    return out


def detections_to_numpy(det: dict) -> np.ndarray:
    dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'),
        ('class_id', 'u1'), ('class_confidence', '<f4'),
    ])
    boxes = det['boxes'].detach().cpu().numpy()
    labels = det['labels'].detach().cpu().numpy().astype(np.uint8)
    scores = det['scores'].detach().cpu().numpy().astype(np.float32)
    out = np.zeros(len(boxes), dtype=dtype)
    if len(boxes):
        # Boxes are in model coord space (MODEL_WIDTH x MODEL_HEIGHT).
        # In dsec mode the original sensor is 2x the model dims (640x480),
        # so boxes get scaled back up. In genx320 mode the sensor is already
        # at 320x320, so boxes map 1:1 in x. We treat y identically to
        # preprocess_events: the model-space y coord is directly in sensor
        # y space (since preprocess did not halve y for genx320).
        x_scale = SCALE if _SENSOR_MODE == "dsec" else 1
        y_scale = SCALE if _SENSOR_MODE == "dsec" else 1
        out['x'] = boxes[:, 0] * x_scale
        out['y'] = boxes[:, 1] * y_scale
        out['w'] = (boxes[:, 2] - boxes[:, 0]) * x_scale
        out['h'] = (boxes[:, 3] - boxes[:, 1]) * y_scale
        out['class_id'] = labels
        out['class_confidence'] = scores
    return out


def serve(socket_path: str, model):
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(4)
    os.chmod(socket_path, 0o666)
    print(f"[infer_server] listening on {socket_path} (binary protocol v1)", flush=True)

    while True:
        conn, _ = srv.accept()
        print("[infer_server] client connected", flush=True)
        try:
            while True:
                msg_type, req_id, plen, _ = recv_header(conn)
                payload = _recv_exact(conn, plen) if plen > 0 else b""
                if msg_type == MSG_PING:
                    conn.sendall(pack_header(MSG_PONG, req_id, 0))
                    continue
                if msg_type != MSG_INFER_REQ:
                    conn.sendall(encode_infer_reply_err(req_id, f"unknown msg_type {msg_type}"))
                    continue
                t0 = time.time()
                try:
                    t_ns, image, events = decode_infer_req(payload)
                    # adapt event field names for preprocess_events (it expects 't_ns' key)
                    det = infer_once(model, image, events)
                    out = detections_to_numpy(det)
                    elapsed_ms = (time.time() - t0) * 1000
                    reply = encode_infer_reply_ok(req_id, out, elapsed_ms)
                except Exception as e:
                    traceback.print_exc()
                    reply = encode_infer_reply_err(req_id, f"{type(e).__name__}: {e}",
                                                    (time.time() - t0) * 1000)
                conn.sendall(reply)
        except (ConnectionError, ValueError) as e:
            print(f"[infer_server] client disconnected: {e}", flush=True)
        except Exception as e:
            print(f"[infer_server] unexpected error: {e}", flush=True)
            traceback.print_exc()
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/dagr-s-dsec.yaml')
    parser.add_argument('--checkpoint', default='data/dagr_s_50.pth')
    parser.add_argument('--img_net', default='resnet50')
    parser.add_argument('--n_nodes', type=int, default=10000,
                        help='Max events per frame (controls peak GPU memory). '
                             'Excess events are dropped (oldest first).')
    parser.add_argument('--dataset_directory', default='data/DSEC_fragment')
    parser.add_argument('--socket', default=SOCKET_PATH)
    parser.add_argument('--sensor_mode', choices=['dsec', 'genx320'], default='dsec',
                        help='Input sensor spatial convention: '
                             'dsec=640x480 events+image, genx320=320x320')
    cli = parser.parse_args()

    set_sensor_mode(cli.sensor_mode)
    set_max_events_per_frame(cli.n_nodes)
    print(f"[infer_server] sensor_mode={cli.sensor_mode} "
          f"max_events_per_frame={cli.n_nodes}", flush=True)
    print("[infer_server] building args...", flush=True)
    args = build_args(cli)
    print("[infer_server] loading model...", flush=True)
    model = load_model(args)
    print("[infer_server] model ready", flush=True)
    serve(cli.socket, model)


if __name__ == '__main__':
    main()
