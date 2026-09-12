# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for the convolutional building blocks."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gusnet.nn.blocks import SPP, Bottleneck, ConvNormAct, CSPLayer, autopad, make_divisible


def test_make_divisible_rounds_to_multiples_of_eight():
    assert make_divisible(64) == 64
    assert make_divisible(63) == 64
    assert make_divisible(16.0) == 16
    assert make_divisible(0.25 * 64) == 16
    assert make_divisible(1) == 8  # never below the divisor


def test_make_divisible_never_loses_more_than_ten_percent():
    # Rounding 100 down to 96 would be fine; rounding 9 down to 8 would lose
    # 11%, so the floor must push it up instead.
    assert make_divisible(100) == 104
    assert make_divisible(9) >= 0.9 * 9


def test_make_divisible_rejects_a_bad_divisor():
    with pytest.raises(ValueError, match="divisor must be positive"):
        make_divisible(32, 0)


@pytest.mark.parametrize(("kernel", "expected"), [(1, 0), (3, 1), (5, 2), (7, 3)])
def test_autopad_preserves_size(kernel, expected):
    assert autopad(kernel) == expected
    conv = nn.Conv2d(1, 1, kernel, padding=autopad(kernel))
    assert conv(torch.zeros(1, 1, 16, 16)).shape[-2:] == (16, 16)


def test_autopad_respects_an_explicit_value():
    assert autopad(3, padding=0) == 0


def test_conv_norm_act_shapes_and_no_redundant_bias():
    block = ConvNormAct(8, 16, kernel_size=3, stride=2)
    out = block(torch.randn(2, 8, 32, 32))
    assert out.shape == (2, 16, 16, 16)
    assert block.conv.bias is None, "the BatchNorm right after already has a bias"


def test_conv_norm_act_can_drop_the_activation():
    block = ConvNormAct(4, 4, activation=False)
    assert isinstance(block.act, nn.Identity)


def test_bottleneck_residual_is_used_only_when_shapes_match():
    same = Bottleneck(16, 16, shortcut=True)
    assert same.add
    changed = Bottleneck(16, 32, shortcut=True)
    assert not changed.add, "a residual across different widths is impossible"
    assert changed(torch.randn(1, 16, 8, 8)).shape == (1, 32, 8, 8)


def test_bottleneck_residual_passes_the_input_through():
    block = Bottleneck(8, 8, shortcut=True).eval()
    # Zero the last convolution so the block computes exactly the identity.
    nn.init.zeros_(block.cv2.conv.weight)
    nn.init.zeros_(block.cv2.norm.weight)
    nn.init.zeros_(block.cv2.norm.bias)
    x = torch.randn(1, 8, 4, 4)
    assert torch.allclose(block(x), x, atol=1e-5)


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_csp_layer_shapes(depth):
    layer = CSPLayer(32, 64, depth=depth)
    assert layer(torch.randn(2, 32, 16, 16)).shape == (2, 64, 16, 16)
    assert len(layer.blocks) == depth


def test_csp_layer_keeps_a_branch_free_of_the_bottlenecks():
    layer = CSPLayer(16, 16, depth=2)
    # cv2 is the bypass: it must see the input directly, not the block stack.
    assert layer.cv2.conv.in_channels == 16
    assert layer.cv3.conv.in_channels == 2 * layer.cv1.conv.out_channels


@pytest.mark.parametrize("fast", [True, False])
def test_spp_shapes(fast):
    block = SPP(64, 32, fast=fast)
    assert block(torch.randn(2, 64, 20, 20)).shape == (2, 32, 20, 20)


def test_fast_spp_is_identical_to_the_parallel_one():
    # Three cascaded stride-1 max-pools of kernel 5 cover exactly the same
    # windows as single pools of kernel 5, 9 and 13. If this ever stops being
    # true, the cheap implementation is quietly changing the architecture.
    fast = SPP(32, 32, fast=True).eval()
    parallel = SPP(32, 32, fast=False).eval()
    parallel.load_state_dict(fast.state_dict())

    x = torch.randn(2, 32, 20, 20)
    with torch.no_grad():
        assert torch.allclose(fast(x), parallel(x), atol=1e-6)


def test_spp_rejects_a_fast_cascade_it_cannot_express():
    with pytest.raises(ValueError, match="fast cascade"):
        SPP(32, 32, kernel_sizes=(3, 5, 7), fast=True)


def test_blocks_are_differentiable():
    model = nn.Sequential(ConvNormAct(3, 8, 3), CSPLayer(8, 8, depth=1), SPP(8, 4))
    out = model(torch.randn(1, 3, 16, 16))
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
