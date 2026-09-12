# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Evaluation: suppression, mean average precision, and the run loop."""

from __future__ import annotations

from gusnet.eval.evaluator import EvalConfig, detections_to_rows, evaluate
from gusnet.eval.metrics import (
    DEFAULT_IOU_THRESHOLDS,
    GroundTruth,
    MeanAveragePrecision,
    MetricResult,
    average_precision,
)
from gusnet.eval.nms import Detections, non_max_suppression

__all__ = [
    "DEFAULT_IOU_THRESHOLDS",
    "Detections",
    "EvalConfig",
    "GroundTruth",
    "MeanAveragePrecision",
    "MetricResult",
    "average_precision",
    "detections_to_rows",
    "evaluate",
    "non_max_suppression",
]
