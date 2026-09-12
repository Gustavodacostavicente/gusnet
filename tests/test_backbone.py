# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for the CSP backbone and the PAFPN neck."""

from __future__ import annotations

import pytest
import torch

from gusnet.nn.backbone import CSPBackbone
from gusnet.nn.neck import PAFPN


def test_backbone_produces_three_levels_at_the_declared_strides():
    backbone = CSPBackbone(width=0.25, depth=0.34).eval()
    size = 256
    with torch.no_grad():
        p3, p4, p5 = backbone(torch.randn(1, 3, size, size))

    for feature, stride in zip((p3, p4, p5), backbone.strides, strict=True):
        assert feature.shape[-2:] == (size // stride, size // stride)


def test_backbone_channels_match_the_declared_ones():
    backbone = CSPBackbone(width=0.5, depth=0.34).eval()
    with torch.no_grad():
        features = backbone(torch.randn(1, 3, 128, 128))
    assert tuple(f.shape[1] for f in features) == backbone.out_channels


@pytest.mark.parametrize("width", [0.25, 0.5, 1.0])
def test_width_scales_the_channels_monotonically(width):
    channels = CSPBackbone(width=width, depth=0.34).out_channels
    assert channels == tuple(sorted(channels)), "deeper levels must not get narrower"
    assert all(c % 8 == 0 for c in channels)


def test_depth_scales_the_number_of_bottlenecks():
    shallow = CSPBackbone(width=0.25, depth=0.34)
    deep = CSPBackbone(width=0.25, depth=1.0)
    assert sum(p.numel() for p in deep.parameters()) > sum(p.numel() for p in shallow.parameters())


def test_depth_never_drops_a_stage_entirely():
    # A tiny multiplier must still leave one bottleneck per stage, otherwise the
    # CSP stages degenerate into plain 1x1 projections.
    backbone = CSPBackbone(width=0.25, depth=0.01)
    assert len(backbone.stage1[1].blocks) >= 1
    assert len(backbone.stage4[1].blocks) >= 1


def test_backbone_rejects_non_positive_multipliers():
    with pytest.raises(ValueError, match="must be positive"):
        CSPBackbone(width=0.0)


def test_neck_preserves_resolution_and_reports_its_channels():
    backbone = CSPBackbone(width=0.25, depth=0.34).eval()
    neck = PAFPN(backbone.out_channels, depth=1).eval()

    with torch.no_grad():
        features = backbone(torch.randn(1, 3, 128, 128))
        outputs = neck(features)

    assert len(outputs) == 3
    for feature, output in zip(features, outputs, strict=True):
        assert output.shape[-2:] == feature.shape[-2:]
    assert tuple(o.shape[1] for o in outputs) == neck.out_channels


def test_neck_can_change_the_output_widths():
    neck = PAFPN((32, 64, 128), depth=1, out_channels=(48, 48, 48)).eval()
    features = (torch.randn(1, 32, 16, 16), torch.randn(1, 64, 8, 8), torch.randn(1, 128, 4, 4))
    with torch.no_grad():
        outputs = neck(features)
    assert tuple(o.shape[1] for o in outputs) == (48, 48, 48)


def test_neck_mixes_every_level_into_every_output():
    """Perturbing the deepest input must reach the shallowest output.

    That round trip is the whole point of the bottom-up path: if a change in P5
    never shows up in N3, the top-down and bottom-up branches are wired wrong.
    """
    neck = PAFPN((16, 32, 64), depth=1).eval()
    features = [torch.randn(1, 16, 16, 16), torch.randn(1, 32, 8, 8), torch.randn(1, 64, 4, 4)]

    with torch.no_grad():
        base = neck(tuple(features))
        features[2] = features[2] + 5.0
        changed = neck(tuple(features))

    assert not torch.allclose(base[0], changed[0]), "P5 does not influence N3"

    with torch.no_grad():
        features[2] = features[2] - 5.0
        features[0] = features[0] + 5.0
        changed_shallow = neck(tuple(features))
    assert not torch.allclose(base[2], changed_shallow[2]), "P3 does not influence N5"
