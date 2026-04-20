"""TRT runtime wrapper that replaces the ResNet50 HookModule.

Loads a pre-built cnn_branch.engine (produced by ipc/export_cnn_trt.py)
and exposes the same ``(features: list[Tensor], outputs: list[Tensor])``
return tuple as the original HookModule.forward — so ``model.backbone.net``
can be swapped in-place.

Design:
    * Inputs/outputs are static-shape, so buffers are allocated once.
    * Uses TRT 8.6 enqueueV3 binding style (set_tensor_address per name).
    * CUDA graph capture is handled by trtexec/TensorRT internally when the
      engine was built with --useCudaGraph; we just replay the context.
    * We allocate the output tensors on the same CUDA device as the input.
      For hot-path performance we keep output buffers around and zero-copy
      return .clone()-free views; callers should not mutate them.

Only one TrtCnn instance per process — creating multiple contexts against
the same engine costs memory we don't need for our single-client daemon.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn

import tensorrt as trt


_LOGGER = trt.Logger(trt.Logger.WARNING)


# Exactly matches the output_names chosen in ipc/export_cnn_trt.py.
_OUTPUT_NAMES = [
    "feat_conv1", "feat_layer1", "feat_layer2", "feat_layer3", "feat_layer4",
    "out_layer3", "out_layer4",
]
_INPUT_NAME = "image"


class TrtCnn(nn.Module):
    """Drop-in replacement for HookModule when DAGR_TRT_CNN=1."""

    def __init__(self, engine_path: str, device: torch.device | str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.engine_path = engine_path

        self._runtime = trt.Runtime(_LOGGER)
        with open(engine_path, "rb") as fh:
            engine_bytes = fh.read()
        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"TRT failed to deserialize {engine_path}")
        self._ctx = self._engine.create_execution_context()

        # Walk the engine to discover tensor roles + dtypes + shapes.
        self._in_shape = tuple(self._engine.get_tensor_shape(_INPUT_NAME))
        self._out_shapes = [
            tuple(self._engine.get_tensor_shape(n)) for n in _OUTPUT_NAMES
        ]

        dtype = self._trt_to_torch_dtype(
            self._engine.get_tensor_dtype(_INPUT_NAME)
        )
        if dtype != torch.float32:
            raise RuntimeError(
                f"TRT engine input dtype {dtype} — expected float32"
            )

        # Preallocate output tensors matching engine dtypes.
        self._out_buffers: List[torch.Tensor] = []
        for name, shape in zip(_OUTPUT_NAMES, self._out_shapes):
            odt = self._trt_to_torch_dtype(self._engine.get_tensor_dtype(name))
            buf = torch.empty(shape, dtype=odt, device=self.device)
            self._out_buffers.append(buf)
            self._ctx.set_tensor_address(name, buf.data_ptr())

        # Input buffer is set per-call with the live tensor's data pointer.
        # Keep a temp contiguous float32 scratch so we can accept any
        # contiguous FP32 (1,3,215,320) tensor from the caller.
        self._in_scratch = torch.empty(self._in_shape, dtype=torch.float32,
                                       device=self.device)

        self._stream = torch.cuda.current_stream(self.device).cuda_stream

    @staticmethod
    def _trt_to_torch_dtype(dtype) -> torch.dtype:
        if dtype == trt.float32:
            return torch.float32
        if dtype == trt.float16:
            return torch.float16
        if dtype == trt.int8:
            return torch.int8
        if dtype == trt.int32:
            return torch.int32
        raise RuntimeError(f"unsupported TRT dtype {dtype}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[list, list]:
        # Match HookModule's call shape: x is (1, 3, H, W), float32, cuda.
        if tuple(x.shape) != self._in_shape:
            raise RuntimeError(
                f"TrtCnn expected input shape {self._in_shape}, got {tuple(x.shape)}"
            )
        # Zero-copy fast path: if x is already contiguous + fp32 + cuda,
        # point TRT directly at its storage. Otherwise copy into scratch.
        if (x.is_contiguous() and x.dtype == torch.float32
                and x.device.type == "cuda"):
            self._ctx.set_tensor_address(_INPUT_NAME, x.data_ptr())
        else:
            self._in_scratch.copy_(x, non_blocking=True)
            self._ctx.set_tensor_address(_INPUT_NAME, self._in_scratch.data_ptr())
        ok = self._ctx.execute_async_v3(stream_handle=self._stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")

        # Split flat outputs back into the (features, outputs) HookModule
        # contract: 5 features + 2 outputs.
        features = self._out_buffers[:5]
        outputs = self._out_buffers[5:]
        return features, outputs


def maybe_install_trt_cnn(model, engine_path: str | None = None,
                          env_var: str = "DAGR_TRT_CNN") -> bool:
    """If the env var is set and the engine file exists, replace
    ``model.backbone.net`` with a TrtCnn instance. Returns True iff the
    swap happened.
    """
    if os.environ.get(env_var, "0") != "1":
        return False
    if engine_path is None:
        engine_path = os.environ.get(
            "DAGR_TRT_CNN_ENGINE",
            "/root/.cache/dagr/cnn_branch.engine",
        )
    if not Path(engine_path).is_file():
        print(f"[trt_cnn] {env_var}=1 but engine not found at {engine_path} "
              f"— skipping, falling back to PyTorch",
              file=sys.stderr, flush=True)
        return False
    trt_cnn = TrtCnn(engine_path).cuda().eval()
    model.backbone.net = trt_cnn
    print(f"[trt_cnn] installed TRT engine {engine_path} in place of "
          f"model.backbone.net", flush=True)
    return True
