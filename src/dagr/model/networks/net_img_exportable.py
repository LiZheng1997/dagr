"""Trace-friendly CNN branch for TRT deployment.

The production ``HookModule`` uses torch forward hooks to harvest
intermediate feature maps from a torchvision ResNet. Hooks mutate a
side-effect list (``self.features``), which torch.onnx.export cannot
capture — the trace sees the module return ``None``.

``ExportableResNetFeatures`` below mirrors exactly what DAGR's
HookModule configures for ``img_net=resnet50`` with:
    feature_layers = ["conv1", "layer1", "layer2", "layer3", "layer4"]
    output_layers  = ["layer3", "layer4"]
plus the 1x1 ``feature_dconv`` / ``output_dconv`` channel reshapes.

Call forward(image) -> tuple of 7 tensors:
    (f0_conv1, f1_layer1, f2_layer2, f3_layer3, f4_layer4, o0_layer3, o1_layer4)

Weight init: construct the wrapper, then copy weights from a trained
HookModule instance via ``load_from_hookmodule(wrapper, hook_module)``.
"""
from __future__ import annotations

import torch
from torch import nn


class ExportableResNetFeatures(nn.Module):
    def __init__(self, resnet, feature_dconv: nn.ModuleList, output_dconv: nn.ModuleList):
        super().__init__()
        # Reuse the trained submodules directly — no copy, no cast.
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # feature_dconv: 5 1x1 convs matching feature_layers
        # output_dconv:  2 1x1 convs matching output_layers
        self.feature_dconv = feature_dconv
        self.output_dconv = output_dconv

    def forward(self, x: torch.Tensor):
        # Matches torchvision ResNet.forward up through the conv1 hook point
        # (the hook fires on conv1 output, *before* bn/relu/maxpool — but
        # HookModule.register_hooks attaches to layer1's INPUT-side call:
        # register_forward_hook("conv1") fires on conv1's output. torchvision
        # ResNet.forward calls self.conv1(x) then bn1/relu/maxpool. So the
        # captured feature is raw conv1 output.
        f0 = self.conv1(x)
        c = self.bn1(f0)
        c = self.relu(c)
        c = self.maxpool(c)
        f1 = self.layer1(c)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        feats = [f0, f1, f2, f3, f4]
        outs = [f3, f4]
        feats = [dc(f) for f, dc in zip(feats, self.feature_dconv)]
        outs = [dc(o) for o, dc in zip(outs, self.output_dconv)]

        return feats[0], feats[1], feats[2], feats[3], feats[4], outs[0], outs[1]


def build_from_hookmodule(hook_module) -> ExportableResNetFeatures:
    """Wrap an already-trained HookModule so the exportable version
    shares weights with it.

    After construction, hook_module is effectively replaced — call the
    wrapper with the same image input and compare outputs against
    hook_module's (features, outputs) pair to validate parity.
    """
    return ExportableResNetFeatures(
        hook_module.module,
        hook_module.feature_dconv,
        hook_module.output_dconv,
    )
