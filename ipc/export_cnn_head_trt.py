"""Export DAGR's image-path CNN head (``model.head.cnn_head``, a
``CNNHead`` subclass of YOLOXHead) to ONNX, then compile a TRT engine.

Run inside the Orin dagr-smoke container:

    docker exec dagr-smoke bash -c \\
        'cd /opt/dagr && python3 ipc/export_cnn_head_trt.py --fp16'

Produces:
    /root/.cache/dagr/cnn_head.onnx
    /root/.cache/dagr/cnn_head.engine
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn

IPC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(IPC_DIR))
sys.path.insert(0, str(IPC_DIR.parent / "src"))

import torch_geometric  # noqa: F401
from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA
from dagr.utils.args import FLAGS as DAGR_FLAGS


MODEL_WIDTH = 320
MODEL_HEIGHT = 215


class CnnHeadExportable(nn.Module):
    """Wraps CNNHead to return a flat 6-tensor tuple instead of a dict
    (dicts confuse torch.onnx.export). Output ordering matches
    TrtCnnHead's expectations: cls0, cls1, reg0, reg1, obj0, obj1."""

    def __init__(self, cnn_head):
        super().__init__()
        self.cnn_head = cnn_head

    def forward(self, feat0, feat1):
        out = self.cnn_head([feat0, feat1])
        return (
            out["cls_output"][0], out["cls_output"][1],
            out["reg_output"][0], out["reg_output"][1],
            out["obj_output"][0], out["obj_output"][1],
        )


def build_args(cli):
    sys.argv = [
        "export_cnn_head_trt",
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


def _resolve_head_input_shapes(model):
    """CNN head inputs come from ``F.interpolate(image_feat, output_sizes)``
    in ``GNNHead.forward``. Query the backbone for the resize target sizes
    and the channel count for each scale (from out_channels_cnn=[256, 256]).
    """
    model.head.output_sizes = model.backbone.get_output_sizes()
    channels = model.backbone.out_channels_cnn
    shapes = []
    for c, (h, w) in zip(channels, model.head.output_sizes):
        shapes.append((1, c, h, w))
    return shapes


@torch.no_grad()
def validate_parity(cnn_head, wrapper, in_shapes, device="cuda", atol=1e-4):
    xs = [torch.rand(s, device=device) for s in in_shapes]
    orig = cnn_head(xs)
    wrapped = wrapper(*xs)
    pairs = [
        ("cls_output[0]", orig["cls_output"][0], wrapped[0]),
        ("cls_output[1]", orig["cls_output"][1], wrapped[1]),
        ("reg_output[0]", orig["reg_output"][0], wrapped[2]),
        ("reg_output[1]", orig["reg_output"][1], wrapped[3]),
        ("obj_output[0]", orig["obj_output"][0], wrapped[4]),
        ("obj_output[1]", orig["obj_output"][1], wrapped[5]),
    ]
    for name, a, b in pairs:
        diff = (a - b).abs().max().item()
        if diff > atol:
            raise RuntimeError(f"parity failure at {name}: max abs diff {diff}")
    print(f"[export-head] parity OK across 6 outputs (max abs diff < {atol})")


def export_onnx(wrapper, in_shapes, onnx_path: Path):
    wrapper.eval()
    dummies = tuple(torch.rand(s, device="cuda") for s in in_shapes)
    print(f"[export-head] torch.onnx.export -> {onnx_path}")
    torch.onnx.export(
        wrapper,
        dummies,
        str(onnx_path),
        input_names=["feat0", "feat1"],
        output_names=["cls0", "cls1", "reg0", "reg1", "obj0", "obj1"],
        opset_version=16,
        do_constant_folding=True,
        dynamic_axes=None,
    )


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool):
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--memPoolSize=workspace:1024",
    ]
    if fp16:
        cmd.append("--fp16")
    print(f"[export-head] {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[export-head] engine built in {time.time()-t0:.1f}s -> {engine_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dagr-s-dsec.yaml")
    parser.add_argument("--checkpoint", default="data/dagr_s_50.pth")
    parser.add_argument("--dataset_directory", default="data/DSEC_fragment")
    parser.add_argument("--output_dir", default="/root/.cache/dagr")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip_build", action="store_true")
    cli = parser.parse_args()

    out = Path(cli.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "cnn_head.onnx"
    engine_path = out / "cnn_head.engine"

    args = build_args(cli)
    model = load_trained_model(args)

    in_shapes = _resolve_head_input_shapes(model)
    print(f"[export-head] resolved CNN head input shapes: {in_shapes}")

    wrapper = CnnHeadExportable(model.head.cnn_head).cuda().eval()
    validate_parity(model.head.cnn_head, wrapper, in_shapes)

    export_onnx(wrapper, in_shapes, onnx_path)

    if cli.skip_build:
        return
    build_engine(onnx_path, engine_path, fp16=cli.fp16)


if __name__ == "__main__":
    main()
