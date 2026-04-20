"""TRT runtime wrapper that replaces the image-path CNN head
(``model.head.cnn_head``, a ``CNNHead`` instance).

The hybrid DAGR head is:

    GNNHead.forward:
        image_feat resized via F.interpolate (stays in PyTorch)
        out_cnn = self.cnn_head(image_feat)        <-- we TRT this
        ... SplineConvToDense path for events (stays in PyTorch) ...

``CNNHead`` is a plain YOLOXHead subclass — standard 2D convs + predictors,
no hooks, no dynamic shapes. Two scales in hybrid mode, so:

    Input:  2 tensors (B, 256, H_k, W_k) for k=0,1 (sizes from
            backbone.get_output_sizes()).
    Output: 6 tensors — cls_output[0/1], reg_output[0/1], obj_output[0/1].

The wrapper returns the dict layout that ``GNNHead.forward`` expects
so it is a drop-in replacement for ``model.head.cnn_head``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import torch
from torch import nn

import tensorrt as trt


_LOGGER = trt.Logger(trt.Logger.WARNING)

_INPUT_NAMES = ["feat0", "feat1"]
_OUTPUT_NAMES = [
    "cls0", "cls1",
    "reg0", "reg1",
    "obj0", "obj1",
]


class TrtCnnHead(nn.Module):
    """Drop-in replacement for ``model.head.cnn_head`` when
    DAGR_TRT_HEAD=1. Forward signature matches ``CNNHead.forward``
    (takes a list of 2 feature tensors, returns the outputs dict)."""

    def __init__(self, engine_path: str, device: torch.device | str = "cuda"):
        super().__init__()
        self.device = torch.device(device)

        self._runtime = trt.Runtime(_LOGGER)
        with open(engine_path, "rb") as fh:
            engine_bytes = fh.read()
        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"TRT failed to deserialize {engine_path}")
        self._ctx = self._engine.create_execution_context()

        self._in_shapes = [
            tuple(self._engine.get_tensor_shape(n)) for n in _INPUT_NAMES
        ]
        self._out_shapes = [
            tuple(self._engine.get_tensor_shape(n)) for n in _OUTPUT_NAMES
        ]

        # Preallocate input scratch (in case caller tensor isn't contiguous)
        self._in_scratch = [
            torch.empty(s, dtype=torch.float32, device=self.device)
            for s in self._in_shapes
        ]

        # Preallocate + bind outputs.
        self._out_buffers: List[torch.Tensor] = []
        for name in _OUTPUT_NAMES:
            shape = tuple(self._engine.get_tensor_shape(name))
            dt = self._trt_to_torch_dtype(self._engine.get_tensor_dtype(name))
            buf = torch.empty(shape, dtype=dt, device=self.device)
            self._out_buffers.append(buf)
            self._ctx.set_tensor_address(name, buf.data_ptr())

        self._stream = torch.cuda.current_stream(self.device).cuda_stream

    @staticmethod
    def _trt_to_torch_dtype(dtype) -> torch.dtype:
        if dtype == trt.float32:
            return torch.float32
        if dtype == trt.float16:
            return torch.float16
        raise RuntimeError(f"unsupported TRT dtype {dtype}")

    @torch.no_grad()
    def forward(self, xin) -> dict:
        # Bind the two input feature maps — zero-copy when contiguous.
        for k in range(2):
            x = xin[k]
            if (x.is_contiguous() and x.dtype == torch.float32
                    and x.device.type == "cuda"
                    and tuple(x.shape) == self._in_shapes[k]):
                self._ctx.set_tensor_address(_INPUT_NAMES[k], x.data_ptr())
            else:
                self._in_scratch[k].copy_(x, non_blocking=True)
                self._ctx.set_tensor_address(
                    _INPUT_NAMES[k], self._in_scratch[k].data_ptr()
                )

        ok = self._ctx.execute_async_v3(stream_handle=self._stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")

        # Repackage into CNNHead's dict contract.
        return {
            "cls_output": [self._out_buffers[0], self._out_buffers[1]],
            "reg_output": [self._out_buffers[2], self._out_buffers[3]],
            "obj_output": [self._out_buffers[4], self._out_buffers[5]],
        }


def maybe_install_trt_head(model, engine_path: str | None = None,
                           env_var: str = "DAGR_TRT_HEAD") -> bool:
    if os.environ.get(env_var, "0") != "1":
        return False
    if engine_path is None:
        engine_path = os.environ.get(
            "DAGR_TRT_HEAD_ENGINE",
            "/root/.cache/dagr/cnn_head.engine",
        )
    if not Path(engine_path).is_file():
        print(f"[trt_head] {env_var}=1 but engine not found at {engine_path} "
              f"— skipping, falling back to PyTorch",
              file=sys.stderr, flush=True)
        return False
    trt_head = TrtCnnHead(engine_path).cuda().eval()
    model.head.cnn_head = trt_head
    print(f"[trt_head] installed TRT engine {engine_path} in place of "
          f"model.head.cnn_head", flush=True)
    return True
