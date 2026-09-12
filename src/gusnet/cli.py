# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Command line entry point.

Every subcommand is implemented: ``check-data``, ``check-assign``,
``model-info``, ``train``, ``val``, ``predict``, ``export`` and ``benchmark``.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from gusnet import __version__
from gusnet.models.detector import VARIANTS

__all__ = ["main"]


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
    train.add_argument("--val-split", help="folder datasets: split to validate on, e.g. val")
    train.add_argument(
        "--val-coco-annotations", type=Path, help="COCO datasets: instances json to validate on"
    )
    train.add_argument(
        "--val-coco-images", type=Path, help="COCO validation images (defaults to --coco-images)"
    )
    train.add_argument("--val-interval", type=int, default=1, help="epochs between validations")
    train.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        metavar="CHECKPOINT",
        help="continue a run; bare --resume uses <save-dir>/last.pt",
    )
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

    predict = sub.add_parser("predict", help="run inference on an image, folder or video")
    predict.add_argument("--weights", type=Path, required=True)
    predict.add_argument(
        "--source", type=Path, required=True, help="image, directory or video file"
    )
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    predict.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    predict.add_argument("--max-det", type=int, default=300)
    predict.add_argument("--batch-size", type=int, default=8)
    predict.add_argument("--device", default="auto")
    predict.add_argument("--half", action="store_true")
    predict.add_argument("--no-ema", action="store_true")
    predict.add_argument("--out", type=Path, default=Path("runs/predict"))
    predict.add_argument("--no-save", action="store_true", help="only print the results")
    predict.add_argument("--fps", type=float, default=25.0, help="output video frame rate")

    export = sub.add_parser("export", help="export a checkpoint to ONNX or TorchScript")
    export.add_argument("--weights", type=Path, required=True)
    export.add_argument("--format", default="onnx", choices=("onnx", "torchscript"))
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    export.add_argument("--dynamic", action="store_true", help="dynamic batch dimension")
    export.add_argument("--nms", action="store_true", help="fold NMS into the graph")
    export.add_argument("--conf", type=float, default=0.25, help="baked in, with --nms")
    export.add_argument("--iou", type=float, default=0.45, help="baked in, with --nms")
    export.add_argument("--out", type=Path, help="output file (defaults next to weights)")
    export.add_argument("--no-ema", action="store_true")

    bench = sub.add_parser("benchmark", help="measure forward-pass latency")
    bench.add_argument("--weights", type=Path, help="checkpoint; omit to use --model")
    bench.add_argument("--model", default="s", choices=sorted(VARIANTS))
    bench.add_argument("--classes", type=int, default=80, help="only with --model")
    bench.add_argument("--imgsz", type=int, default=640)
    bench.add_argument("--batch-size", type=int, default=1)
    bench.add_argument("--iterations", type=int, default=50)
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--device", default="auto")
    bench.add_argument("--half", action="store_true")

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


def _build_val_dataset(args: argparse.Namespace):
    """Build the validation set, or return ``None`` if none was asked for.

    The two dataset formats need different arguments, and mixing them up used to
    be silent: passing ``--val-split`` alongside COCO annotations validated on
    the *training* set, reporting a memorisation score as if it were
    generalisation. Each combination is now either explicit or an error.
    """
    using_coco = args.coco_annotations is not None

    if args.val_split and using_coco:
        raise SystemExit(
            "--val-split applies to folder datasets; for COCO use "
            "--val-coco-annotations (with --val-coco-images if the directory differs)"
        )
    if args.val_coco_annotations and not using_coco:
        raise SystemExit("--val-coco-annotations applies to COCO datasets; use --val-split")

    if not args.val_split and not args.val_coco_annotations:
        return None

    val_args = argparse.Namespace(**vars(args))
    val_args.no_augment = True
    if using_coco:
        val_args.coco_annotations = args.val_coco_annotations
        val_args.coco_images = args.val_coco_images or args.coco_images
    else:
        val_args.split = args.val_split
    return _build_dataset(val_args)


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

    val_dataset = _build_val_dataset(args)
    if val_dataset is not None:
        print(f"validating on {len(val_dataset)} images")

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

    resume = args.resume
    if resume is True:
        resume = args.save_dir / "last.pt"
        if not resume.is_file():
            raise SystemExit(f"nothing to resume: {resume} does not exist")

    print(f"GUSNet-{args.model}: {model.num_parameters():,} parameters")
    Trainer(
        model,
        dataset,
        config,
        criterion=criterion,
        val_dataset=val_dataset,
        resume=resume,
    ).train()
    return 0


def _predict(args: argparse.Namespace) -> int:
    import cv2

    from gusnet.predict import VIDEO_SUFFIXES, PredictConfig, Predictor

    config = PredictConfig(
        img_size=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_det=args.max_det,
        batch_size=args.batch_size,
        device=args.device,
        half=args.half,
    )
    predictor = Predictor.from_checkpoint(args.weights, config=config, prefer_ema=not args.no_ema)
    names = predictor.class_names

    is_video = args.source.suffix.lower() in VIDEO_SUFFIXES
    writer = None
    saved = 0
    total = 0

    if not args.no_save:
        args.out.mkdir(parents=True, exist_ok=True)

    for name, image, detections in predictor.run(args.source):
        total += len(detections)
        labels = [
            names[int(label)] if names and int(label) < len(names) else str(int(label))
            for label in detections.labels.tolist()
        ]
        summary = ", ".join(sorted(set(labels))) if labels else "nothing"
        print(f"{name}: {len(detections)} detections ({summary})")

        if args.no_save:
            continue

        annotated = cv2.cvtColor(predictor.annotate(image, detections), cv2.COLOR_RGB2BGR)
        if is_video:
            if writer is None:
                height, width = annotated.shape[:2]
                target = args.out / f"{args.source.stem}.mp4"
                writer = cv2.VideoWriter(
                    str(target), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
                )
            writer.write(annotated)
        else:
            cv2.imwrite(str(args.out / f"{name}.jpg"), annotated)
        saved += 1

    if writer is not None:
        writer.release()

    print()
    print(f"{total} detections total")
    if not args.no_save and saved:
        print(f"results written to {args.out}")
    return 0


def _export(args: argparse.Namespace) -> int:
    from gusnet.export import export_onnx, export_torchscript
    from gusnet.train import model_from_checkpoint

    model, checkpoint = model_from_checkpoint(args.weights, prefer_ema=not args.no_ema)
    suffix = ".onnx" if args.format == "onnx" else ".torchscript"
    out = args.out or args.weights.with_suffix(suffix)
    where = "in" if args.nms else "out of"

    print(f"GUSNet: {model.num_parameters():,} parameters, {model.num_classes} classes")
    print(f"exporting {args.format} at {args.imgsz}px, NMS {where} the graph")

    options = {"conf_threshold": args.conf, "iou_threshold": args.iou}
    if args.format == "onnx":
        path = export_onnx(
            model,
            out,
            img_size=args.imgsz,
            opset=args.opset,
            dynamic_batch=args.dynamic,
            nms=args.nms,
            **options,
        )
    else:
        path = export_torchscript(model, out, img_size=args.imgsz, nms=args.nms, **options)

    size_mb = path.stat().st_size / 1e6
    print(f"wrote {path} ({size_mb:.1f} MB)")
    if checkpoint.get("model_args", {}).get("class_names"):
        print("class names live in the checkpoint, not in the exported graph")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from gusnet.export import benchmark
    from gusnet.models import GUSNet

    if args.weights:
        from gusnet.train import model_from_checkpoint

        model, _ = model_from_checkpoint(args.weights)
        label = str(args.weights)
    else:
        model = GUSNet.from_variant(args.model, num_classes=args.classes)
        label = f"GUSNet-{args.model} ({args.classes} classes, untrained)"

    print(f"{label}: {model.num_parameters():,} parameters")
    result = benchmark(
        model,
        img_size=args.imgsz,
        batch_size=args.batch_size,
        iterations=args.iterations,
        warmup=args.warmup,
        device=args.device,
        half=args.half,
    )
    print(
        f"{args.imgsz}px  batch {args.batch_size}  "
        f"{result['ms_per_image']:.2f} ms/image  {result['fps']:.1f} FPS"
    )
    return 0


def _force_utf8_output() -> None:
    """Stop a non-UTF-8 console from killing the process over a character.

    Windows consoles default to a legacy code page, and several libraries in
    this stack print status lines containing symbols outside it -- torch's ONNX
    exporter uses a check mark. Writing one raises UnicodeEncodeError and takes
    the whole command down, after the real work has already succeeded. Replacing
    unencodable characters is strictly better than crashing over decoration.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    args = build_parser().parse_args(argv)
    if args.command == "check-data":
        return _check_data(args)
    if args.command == "train":
        return _train(args)
    if args.command == "val":
        return _val(args)
    if args.command == "predict":
        return _predict(args)
    if args.command == "export":
        return _export(args)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "check-assign":
        return _check_assign(args)
    if args.command == "model-info":
        return _model_info(args)
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
