# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Tests for the command line, focused on the choices it must not get wrong."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from gusnet.cli import _build_val_dataset, build_parser, main


def _train_args(**overrides) -> argparse.Namespace:
    base = {
        "root": None,
        "split": "train",
        "coco_images": None,
        "coco_annotations": None,
        "imgsz": 64,
        "val_split": None,
        "val_coco_annotations": None,
        "val_coco_images": None,
        "no_augment": False,
        "seed": 0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


# ------------------------------------------------------------------- the parser


def test_parser_exposes_every_implemented_command():
    parser = build_parser()
    actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    commands = set(actions[0].choices)
    assert commands == {
        "check-data",
        "check-assign",
        "model-info",
        "train",
        "val",
        "predict",
        "export",
        "benchmark",
    }


def test_version_flag_exits_cleanly():
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0


def test_model_info_runs(capsys):
    assert main(["model-info", "--model", "n", "--classes", "2", "--imgsz", "64"]) == 0
    printed = capsys.readouterr().out
    assert "parameters" in printed
    assert "grid points" in printed


# --------------------------------------------------- choosing a validation set


def test_no_validation_source_means_no_validation():
    assert _build_val_dataset(_train_args(root=Path("x"))) is None


def test_val_split_with_coco_is_an_error_not_a_guess():
    """It used to be accepted, and silently validated on the training set.

    That reports a memorisation score where a generalisation score belongs and
    picks best.pt by it -- a wrong number that looks entirely right.
    """
    args = _train_args(
        coco_images=Path("images"),
        coco_annotations=Path("instances.json"),
        val_split="val",
    )
    with pytest.raises(SystemExit, match="--val-split applies to folder datasets"):
        _build_val_dataset(args)


def test_val_coco_annotations_with_a_folder_dataset_is_an_error():
    args = _train_args(root=Path("data"), val_coco_annotations=Path("instances.json"))
    with pytest.raises(SystemExit, match="--val-coco-annotations applies to COCO"):
        _build_val_dataset(args)


def test_val_split_builds_from_the_named_split(folder_dataset: Path):
    import shutil

    # Give the dataset a second split to validate on.
    for kind in ("images", "labels"):
        shutil.copytree(folder_dataset / kind / "train", folder_dataset / kind / "val")

    args = _train_args(root=folder_dataset, split="train", val_split="val")
    dataset = _build_val_dataset(args)

    assert dataset is not None
    assert len(dataset) == 6
    assert dataset.augment is False, "validation must never be augmented"


def test_val_coco_images_defaults_to_the_training_images(coco_dataset: tuple[Path, Path]):
    images_dir, annotation_file = coco_dataset
    args = _train_args(
        coco_images=images_dir,
        coco_annotations=annotation_file,
        val_coco_annotations=annotation_file,
    )
    dataset = _build_val_dataset(args)

    assert dataset is not None
    assert len(dataset) == 4
    assert dataset.augment is False
