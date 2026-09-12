# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Label assignment: deciding which grid points are responsible for which object."""

from __future__ import annotations

from gusnet.assign.simota import SimOTAAssigner
from gusnet.assign.tal import TaskAlignedAssigner
from gusnet.assign.utils import (
    Assignment,
    points_in_boxes,
    points_in_centers,
    resolve_conflicts,
    targets_to_batch,
)

__all__ = [
    "Assignment",
    "SimOTAAssigner",
    "TaskAlignedAssigner",
    "points_in_boxes",
    "points_in_centers",
    "resolve_conflicts",
    "targets_to_batch",
]
