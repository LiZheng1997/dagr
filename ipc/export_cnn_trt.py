"""Export DAGR's CNN branch (HookModule wrapping ResNet50) to ONNX, then
compile a TRT engine for the Orin AGX (sm_87).

Run inside the Orin dagr-smoke container:

    docker exec -it dagr-smoke bash -c \\
        'cd /opt/dagr && python3 ipc/export_cnn_trt.py \\
            --checkpoint data/dagr_s_50.pth \\
            --dataset_directory data/DSEC_fragment \\
            --output_dir data \\
            --fp16'

This produces:
    data/cnn_branch.onnx    (FP32, static shape 1x3x215x320)
    data/cnn_branch.engine  (FP16, Orin sm_87, bound to current TRT version)

The engine file is NOT portable across JetPack / TRT major versions;
re-run this script after any such upgrade.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

IPC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(IPC_DIR))
sys.path.insert(0, str(IPC_DIR.parent / "src"))

import torch_geometric  # noqa: F401 — needed so DAGR import path works
from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA
from dagr.model.networks.net_img_exportable import build_from_hookmodule
from dagr.utils.args import FLAGS as DAGR_FLAGS


MODEL_WIDTH = 320
MODEL_HEIGHT = 215


def build_args(cli):
    sys.argv = [
        "export_cnn_trt",
        "--config", cli.config,
        "--checkpoint", cli.checkpoint,
        "--output_directory", "/tmp/dagr_export_out",
        "--batch_size", "1",
        "--dataset_directory", cli.dataset_directory,
        "--use_image",
        "--img_net", "resnet50",
        "--n_nodes", "10000",
        "--no_eval",
    ]
    os.makedirs("/tmp/dagr_export_out", exist_ok=True)
    return DAGR_FLAGS()


def load_trained_model(args):
    model = DAGR(args, height=MODEL_HEIGHT, width=MODEL_WIDTH).cuda()
    ema = ModelEMA(model)
    ckpt = torch.load(args.checkpoint, map_location="cuda")
    ema.ema.load_state_dict(ckpt["ema"])
    ema.ema.cache_luts(radius=args.radius, height=MODEL_HEIGHT, width=MODEL_WIDTH)
    ema.ema.eval()
    return ema.ema


@torch.no_grad()
def validate_parity(orig_hookmodule, wrapper, device="cuda", atol=1e-4):
    """Run both on the same input, check every output tensor matches."""
    x = torch.rand(1, 3, MODEL_HEIGHT, MODEL_WIDTH, device=device)
    orig_feats, orig_outs = orig_hookmodule(x)
    wrapped = wrapper(x)
    # wrapper returns flat 7-tuple: 5 feats + 2 outs
    assert len(wrapped) == 7
    for i, (a, b) in enumerate(zip(orig_feats + orig_outs, wrapped)):
        diff = (a - b).abs().max().item()
        if diff > atol:
            raise RuntimeError(f"parity failure at output {i}: max abs diff {diff}")
    print(f"[export] parity OK across 7 outputs (max abs diff < {atol})")


def export_onnx(wrapper, onnx_path: Path):
    wrapper.eval()
    dummy = torch.rand(1, 3, MODEL_HEIGHT, MODEL_WIDTH, device="cuda")
    input_names = ["image"]
    output_names = [
        "feat_conv1", "feat_layer1", "feat_layer2", "feat_layer3", "feat_layer4",
        "out_layer3", "out_layer4",
    ]
    print(f"[export] torch.onnx.export -> {onnx_path}")
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=16,
        do_constant_folding=True,
        # Static shapes — detector only ever feeds 1x3x215x320.
        dynamic_axes=None,
    )


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool):
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.isfile(trtexec):
        raise RuntimeError(f"trtexec not found at {trtexec}")
    # NOTE: deliberately no --useCudaGraph. CUDA graph replay in a hybrid
    # pipeline (TRT engine + subsequent PyG / custom CUDA ops on the same
    # stream) inflates the downstream op timings in our profiling by
    # 5-8ms per frame — seems like graph-replay side effects impede the
    # subsequent dispatches. We lose ~200us per call of trtexec peak
    # throughput but gain it back 10x downstream.
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--memPoolSize=workspace:2048",
    ]
    if fp16:
        cmd.append("--fp16")
    print(f"[export] {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[export] engine built in {time.time()-t0:.1f}s -> {engine_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dagr-s-dsec.yaml")
    parser.add_argument("--checkpoint", default="data/dagr_s_50.pth")
    parser.add_argument("--dataset_directory", default="data/DSEC_fragment")
    # /opt/dagr/data is a ro mount inside the container — default output to
    # the persistent cache mount instead. Host-side: /home/lz/dagr_cache/dagr/.
    parser.add_argument("--output_dir", default="/root/.cache/dagr")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip_build", action="store_true",
                        help="Export ONNX only, don't run trtexec")
    cli = parser.parse_args()

    out = Path(cli.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "cnn_branch.onnx"
    engine_path = out / "cnn_branch.engine"

    args = build_args(cli)
    model = load_trained_model(args)

    wrapper = build_from_hookmodule(model.backbone.net).cuda().eval()
    validate_parity(model.backbone.net, wrapper)

    export_onnx(wrapper, onnx_path)

    if cli.skip_build:
        return
    build_engine(onnx_path, engine_path, fp16=cli.fp16)


if __name__ == "__main__":
    main()
