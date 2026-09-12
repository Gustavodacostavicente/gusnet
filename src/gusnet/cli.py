# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Command line entry point.

Everything except ``predict`` and ``export`` is implemented. Those two are
declared so the interface is fixed, and exit with a message pointing at the
roadmap rather than pretending to work.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gusnet import __version__
from gusnet.models.detector import VARIANTS

__all__ = ["main"]

_NOT_READY = (
    "{command!r} is not implemented yet. GUSNet can train and evaluate "
    "(roadmap phases 1-7); inference and export are phase 8. See docs/ROADMAP.md."
)


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, help="dataset root (images/ + labels/ layout)")
    parser.add_argument("--split", default="train", help="split subdirectory (default: train)")
    parser.add_argument("--coco-images", type=Path, help="image directory for a COCO dataset")
    parser.add_argument("--coco-annotations", type=Path, help="COCO instances_*.json")
    parser.add_argument("--imgsz", type=int, default=640, help="training size (default: 640)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gusnet",
        description="GUSNet - a clean-room, Apache-2.0 one-stage object detector.",
    )
    parser.add_argument("--version", action="version", version=f"gusnet {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check-data",
        help="load a dataset, run the augmentation pipeline and write a preview grid",
    )
    _add_data_arguments(check)
    check.add_argument("--batch-size", type=int, default=8)
    check.add_argument("--no-augment", action="store_true", help="disable augmentation")
    check.add_argument(
        "--out", type=Path, default=Path("runs/check-data/batch.jpg"), help="preview image path"
    )
    check.add_argument("--seed", type=int, default=0)

    assign = sub.add_parser(
        "check-assign",
        help="run an assigner on a real batch and draw which grid points it selected",
    )
    _add_data_arguments(assign)
    assign.add_argument("--model", default="n", choices=sorted(VARIANTS))
    assign.add_argument("--assigner", default="tal", choices=("tal", "simota"))
    assign.add_argument("--topk", type=int, default=13, help="TAL candidates per object")
    assign.add_argument("--batch-size", type=int, default=4)
    assign.add_argument("--seed", type=int, default=0)
    assign.add_argument(
        "--out", type=Path, default=Path("runs/check-assign/batch.jpg"), help="preview image"
    )
    assign.set_defaults(no_augment=True)

    info = sub.add_parser(
        "model-info", help="build a model and report its shapes and parameter count"
    )
    info.add_argument("--model", default="s", choices=sorted(VARIANTS), help="variant")
    info.add_argument("--classes", type=int, default=80, help="number of classes")
    info.add_argument("--imgsz", type=int, default=640, help="input size")

    train = sub.add_parser("train", help="train a detector")
    _add_data_arguments(train)
    train.add_argument("--model", default="s", choices=sorted(VARIANTS), help="variant")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--optimizer", default="sgd", choices=("sgd", "adamw"))
    train.add_argument("--lr0", type=float, default=0.01, help="initial learning rate")
    train.add_argument("--weight-decay", type=float, default=5e-4)
    train.add_argument("--warmup-epochs", type=float, default=3.0)
    train.add_argument(
        "--close-mosaic", type=int, default=15, help="final epochs without mosaic/mixup"
    )
    train.add_argument("--assigner", default="tal", choices=("tal", "simota"))
    train.add_argument("--workers", type=int, default=0, help="dataloader workers")
    train.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ...")
    train.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--save-dir", type=Path, default=Path("runs/train"))
    train.add_argument("--log-interval", type=int, default=10, help="0 to silence steps")
    train.add_argument("--val-split", help="split to validate on, e.g. val")
    train.add_argument("--val-interval", type=int, default=1, help="epochs between validations")
    train.set_defaults(no_augment=False)

    val = sub.add_parser("val", help="evaluate a checkpoint and report mAP")
    _add_data_arguments(val)
    val.add_argument("--weights", type=Path, required=True, help="checkpoint to evaluate")
    val.add_argument("--batch-size", type=int, default=16)
    val.add_argument(
        "--conf", type=float, default=0.001, help="confidence floor (keep it low for mAP)"
    )
    val.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    val.add_argument("--max-det", type=int, default=300)
    val.add_argument("--workers", type=int, default=0)
    val.add_argument("--device", default="auto")
    val.add_argument("--half", action="store_true", help="run in float16 on CUDA")
    val.add_argument(
        "--no-ema", action="store_true", help="evaluate the raw weights, not the average"
    )
    val.set_defaults(no_augment=True, seed=0)

    for name, help_text in (
        ("predict", "run inference on images or video"),
        ("export", "export a checkpoint to ONNX or TorchScript"),
    ):
        sub.add_parser(name, help=f"{help_text} (not implemented yet)")

    return parser


def _build_dataset(args: argparse.Namespace):
    from gusnet.data import DetectionDataset

    common = {
        "img_size": args.imgsz,
        "augment": not args.no_augment,
        "seed": args.seed,
    }
    if args.coco_annotations is not None:
        if args.coco_images is None:
            raise SystemExit("--coco-annotations requires --coco-images")
        return DetectionDataset.from_coco(args.coco_images, args.coco_annotations, **common)
    if args.root is not None:
        return DetectionDataset.from_folder(args.root, args.split, **common)
    raise SystemExit("give either --root or --coco-images/--coco-annotations")


def _check_data(args: argparse.Namespace) -> int:
    from gusnet.data import build_dataloader
    from gusnet.viz import save_batch_preview

    dataset = _build_dataset(args)
    print(f"dataset: {len(dataset)} images, {dataset.num_classes} classes")
    print(f"classes: {', '.join(dataset.class_names[:20])}")

    loader = build_dataloader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    images = batch["images"]
    print(f"batch images: shape={tuple(images.shape)} dtype={images.dtype}")
    print(f"objects in batch: {len(batch['targets'])}")

    out = save_batch_preview(batch, args.out, class_names=dataset.class_names)
    print(f"preview written to {out}")
    return 0


def _check_assign(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np
    import torch

    from gusnet.assign import SimOTAAssigner, TaskAlignedAssigner, targets_to_batch
    from gusnet.data import build_dataloader
    from gusnet.models import GUSNet
    from gusnet.viz import draw_boxes, draw_points, tensor_to_image

    torch.manual_seed(args.seed)
    dataset = _build_dataset(args)
    loader = build_dataloader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    model = GUSNet.from_variant(args.model, num_classes=dataset.num_classes).eval()
    with torch.no_grad():
        out = model(batch["images"])

    gt_labels, gt_boxes, gt_mask = targets_to_batch(batch["targets"], len(batch["images"]))

    if args.assigner == "tal":
        assigner = TaskAlignedAssigner(dataset.num_classes, topk=args.topk)
        result = assigner(out["scores"], out["boxes"], out["points"], gt_labels, gt_boxes, gt_mask)
    else:
        assigner = SimOTAAssigner(dataset.num_classes)
        result = assigner(
            out["scores"],
            out["boxes"],
            out["points"],
            gt_labels,
            gt_boxes,
            gt_mask,
            out["strides"],
        )

    objects = int(gt_mask.sum())
    print(f"assigner:   {args.assigner}")
    print(f"objects:    {objects}")
    print(f"grid points: {out['scores'].shape[1]:,} per image")
    print(f"positives:  {result.num_foreground}")
    if objects:
        print(f"            {result.num_foreground / objects:.1f} per object")

    tiles = []
    points = out["points"].cpu().numpy()
    for index, image in enumerate(batch["images"]):
        tile = tensor_to_image(image)
        tile = draw_boxes(
            tile, batch["boxes"][index], batch["classes"][index], class_names=dataset.class_names
        )
        mask = result.fg_mask[index].cpu().numpy()
        tiles.append(
            draw_points(tile, points[mask], result.target_labels[index].cpu().numpy()[mask])
        )

    grid = np.concatenate(tiles, axis=1) if len(tiles) > 1 else tiles[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"preview written to {args.out}")
    return 0


def _model_info(args: argparse.Namespace) -> int:
    import torch

    from gusnet.models import GUSNet

    model = GUSNet.from_variant(args.model, num_classes=args.classes).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, args.imgsz, args.imgsz))

    print(f"GUSNet-{args.model}  {args.classes} classes  {args.imgsz}x{args.imgsz}")
    print(f"parameters: {model.num_parameters():,}")
    print(f"strides:    {model.strides}")
    print(f"grid points: {out['cls_logits'].shape[1]:,}")
    for key in ("cls_logits", "reg_logits", "boxes"):
        print(f"  {key:<11} {tuple(out[key].shape)}")
    return 0


def _val(args: argparse.Namespace) -> int:
    from gusnet.eval import EvalConfig, evaluate
    from gusnet.train import model_from_checkpoint

    dataset = _build_dataset(args)
    model, checkpoint = model_from_checkpoint(args.weights, prefer_ema=not args.no_ema)

    if model.num_classes != dataset.num_classes:
        raise SystemExit(
            f"checkpoint has {model.num_classes} classes but the dataset has "
            f"{dataset.num_classes}; they must match"
        )

    weights = "raw" if args.no_ema else "EMA"
    print(f"{args.weights} (epoch {checkpoint.get('epoch', '?')}, {weights} weights)")
    print(f"{len(dataset)} images, {dataset.num_classes} classes, {args.imgsz}px")
    print()

    evaluate(
        model,
        dataset,
        EvalConfig(
            batch_size=args.batch_size,
            img_size=args.imgsz,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            max_det=args.max_det,
            workers=args.workers,
            device=args.device,
            half=args.half,
        ),
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    from gusnet.assign import SimOTAAssigner, TaskAlignedAssigner
    from gusnet.losses import DetectionLoss
    from gusnet.models import GUSNet
    from gusnet.train import TrainConfig, Trainer

    dataset = _build_dataset(args)
    model = GUSNet.from_variant(
        args.model, num_classes=dataset.num_classes, class_names=dataset.class_names
    )

    assigner = (
        TaskAlignedAssigner(dataset.num_classes)
        if args.assigner == "tal"
        else SimOTAAssigner(dataset.num_classes)
    )
    criterion = DetectionLoss(dataset.num_classes, reg_max=model.head.reg_max, assigner=assigner)

    val_dataset = None
    if args.val_split:
        val_args = argparse.Namespace(**vars(args))
        val_args.split = args.val_split
        val_args.no_augment = True
        val_dataset = _build_dataset(val_args)
        print(f"validating on {len(val_dataset)} images from split {args.val_split!r}")

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        amp=not args.no_amp,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
    )

    print(f"GUSNet-{args.model}: {model.num_parameters():,} parameters")
    Trainer(model, dataset, config, criterion=criterion, val_dataset=val_dataset).train()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check-data":
        return _check_data(args)
    if args.command == "train":
        return _train(args)
    if args.command == "val":
        return _val(args)
    if args.command == "check-assign":
        return _check_assign(args)
    if args.command == "model-info":
        return _model_info(args)
    raise SystemExit(_NOT_READY.format(command=args.command))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
