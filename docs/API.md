# API reference

Every public symbol in GUSNet, with its signature and what it is for. For *why*
things are built this way, read [`ARCHITECTURE.md`](ARCHITECTURE.md).

Conventions used throughout:

* **Shapes** — `B` batch, `A` grid points (8400 at 640 px), `C` classes,
  `M` objects per image, `N` a flat count.
* **Box formats** are always named: `xyxy` corners, `cxcywh` centre/size,
  `ltrb` distances from a point to the four edges.
* **Coordinates** are absolute pixels in the letterboxed image, unless a
  docstring says "normalised" (`[0, 1]`) or "stride units".

---

## `gusnet`

```python
__version__  # "0.1.0.dev0"
__author__  # "Gustavo da Costa Vicente"
__license__  # "Apache-2.0"
```

---

## `gusnet.ops.boxes`

Box conversions and overlap metrics. All functions take a tensor whose last
dimension is 4 and preserve the leading dimensions.

| Function | Purpose |
|---|---|
| `xyxy_to_cxcywh(boxes)` | corners → centre/size |
| `cxcywh_to_xyxy(boxes)` | centre/size → corners |
| `xyxy_to_ltrb(boxes, points)` | distances from `points` to the box edges; negative components mean the point is outside |
| `ltrb_to_xyxy(dist, points)` | decode predicted distances back into boxes |
| `box_area(boxes)` | area; degenerate boxes clamp to 0 |
| `box_iou(boxes1, boxes2)` | pairwise `(N, 4) × (M, 4) → (N, M)` |
| `bbox_iou(boxes1, boxes2, *, kind="iou")` | element-wise with broadcasting; `kind` ∈ `iou`, `giou`, `diou`, `ciou` |
| `clip_boxes(boxes, shape)` | clamp to an image of `(height, width)` |
| `scale_boxes(boxes, ratio, pad, orig_shape)` | undo a letterbox, back to original-image coordinates |

```python
from gusnet.ops import boxes as B

B.bbox_iou(pred, target, kind="ciou")  # (...) overlap, ≤ 1, can go negative
B.scale_boxes(pred, result.ratio, result.pad, result.orig_shape)
```

---

## `gusnet.data`

### `letterbox`

```python
letterbox(image, new_shape=640, *, color=114, scaleup=True,
          center=True, stride=None) -> LetterboxResult
letterbox_boxes(boxes, result) -> np.ndarray
```

`LetterboxResult` is a frozen dataclass with `image`, `ratio`, `pad` and
`orig_shape`. Pass `scaleup=False` at inference, `center=False` for mosaic
tiles, `stride=32` for rectangular inference.

### `transforms`

```python
mosaic4(tiles, img_size, *, fill=114, rng=None)
random_affine(image, boxes, classes, *, degrees=0.0, translate=0.1,
              scale=0.5, shear=0.0, border=(0, 0), fill=114, rng=None)
mixup(a, b, *, alpha=32.0, rng=None)
hsv_augment(image, *, hgain=0.015, sgain=0.7, vgain=0.4, rng=None)
random_hflip(image, boxes, *, p=0.5, rng=None)
filter_boxes(boxes, classes, *, min_size=2.0, min_area_ratio=0.1,
             original=None, max_aspect=20.0)
```

All take `(H, W, 3)` uint8 RGB and `(N, 4)` float32 xyxy in absolute pixels.
`mosaic4` needs exactly four `(image, boxes, classes)` tuples and returns a
canvas of side `2 × img_size`; pair it with
`random_affine(..., border=(-img_size // 2, -img_size // 2))` to crop back.

### `DetectionDataset`

```python
DetectionDataset(annotations, class_names, *, img_size=640,
                 augment=False, config=None, seed=None)

DetectionDataset.from_folder(root, split="train", *, class_names=None, **kwargs)
DetectionDataset.from_coco(images_dir, annotation_file, *, drop_crowd=True, **kwargs)
```

Properties: `num_classes`, `class_names`, `annotations`, `config`.
Method: `load_raw(index) -> _RawSample` (decoded image, boxes in pixels).

Each item is a dict:

| Key | Shape / type | Meaning |
|---|---|---|
| `image` | `(3, H, W)` float32 | RGB in `[0, 1]` |
| `boxes` | `(N, 4)` float32 | xyxy in this image's coordinates |
| `classes` | `(N,)` int64 | class indices |
| `image_id` | int | for evaluation |
| `ratio`, `pad`, `orig_shape` | float, tuple, tuple | letterbox parameters |

**Folder layout** expected by `from_folder`:

```
root/
  images/train/frame001.jpg
  labels/train/frame001.txt     # "<class> <cx> <cy> <w> <h>", normalised
  classes.txt                   # one name per line
```

### `Annotation`

```python
Annotation(image_path, boxes, classes, image_id=0)
```

`boxes` is `(N, 4)` **normalised** xyxy — the only representation that survives
not knowing the image size before decoding.

### `AugmentConfig`

```python
AugmentConfig(
    mosaic=1.0,
    mixup=0.1,
    hflip=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    mixup_alpha=32.0,
)
```

### `build_dataloader` / `collate_detection`

```python
build_dataloader(dataset, *, batch_size=16, shuffle=True, num_workers=0,
                 pin_memory=None, drop_last=False, persistent_workers=None)
```

The collated batch:

| Key | Shape | Meaning |
|---|---|---|
| `images` | `(B, 3, H, W)` | stacked |
| `boxes` / `classes` | lists of `B` tensors | per image |
| `targets` | `(M, 6)` | `[batch_index, class, x1, y1, x2, y2]` |
| `image_ids`, `ratios`, `pads`, `orig_shapes` | lists of `B` | metadata |

---

## `gusnet.nn`

### `blocks`

```python
ConvNormAct(in_channels, out_channels, kernel_size=1, stride=1, *,
            padding=None, groups=1, dilation=1, activation=True)
Bottleneck(in_channels, out_channels, *, shortcut=True, expansion=0.5,
           kernel_sizes=(1, 3), groups=1)
CSPLayer(in_channels, out_channels, *, depth=1, shortcut=True, expansion=0.5)
SPP(in_channels, out_channels, *, kernel_sizes=(5, 9, 13), fast=True)

autopad(kernel_size, padding=None, dilation=1) -> int
make_divisible(value, divisor=8) -> int
```

`ConvNormAct.fuse_forward(x)` skips the norm, for use after conv/bn fusion.
`SPP(fast=True)` is the cascade of three 5×5 pools — bit-identical to the
parallel 5/9/13 pools, and cheaper.

### `CSPBackbone`

```python
CSPBackbone(*, width=1.0, depth=1.0, in_channels=3, divisor=8)
```

`forward(x) -> (P3, P4, P5)` at strides 8, 16, 32.
Attributes: `out_channels`, `strides`, `width`, `depth`, `in_channels`.

### `PAFPN`

```python
PAFPN(in_channels, *, depth=3, out_channels=None)
```

`forward((P3, P4, P5)) -> (N3, N4, N5)` at the same resolutions.
Attribute: `out_channels`.

### `DetectHead`

```python
DetectHead(in_channels, num_classes, *, strides=(8, 16, 32),
           reg_max=16, hidden_channels=None, stem_depth=2)
```

`forward(features) -> DetectionOutput`. `reset_parameters(prior_probability=0.01)`
re-applies the classification bias prior.

```python
make_anchor_points(feature_shapes, strides, *, offset=0.5,
                   device="cpu", dtype=torch.float32) -> (points, strides)
decode_distances(reg_logits, project) -> Tensor
```

### `DetectionOutput`

A plain `dict` (so it survives `torch.compile`) with:

| Key | Shape | Meaning |
|---|---|---|
| `cls_logits` | `(B, A, C)` | raw, pre-sigmoid |
| `reg_logits` | `(B, A, 4, reg_max + 1)` | raw distance distributions |
| `points` | `(A, 2)` | grid points in image pixels |
| `strides` | `(A, 1)` | stride of each point |
| `boxes` | `(B, A, 4)` | decoded xyxy in image pixels |
| `scores` | `(B, A, C)` | per-class confidence in `[0, 1]` |

---

## `gusnet.models`

### `GUSNet`

```python
GUSNet(num_classes=80, *, width=1.0, depth=1.0, reg_max=16,
       in_channels=3, class_names=None)
GUSNet.from_variant(variant, num_classes=80, **kwargs)   # "n" | "s" | "m" | "l" | "x"
```

`forward(images) -> DetectionOutput`. Input must be `(B, 3, H, W)` with `H` and
`W` divisible by 32.

Properties: `strides`, `stride_max`, `num_classes`, `class_names`.
Method: `num_parameters(trainable_only=True)`.

```python
VARIANTS = {
    "n": (0.25, 0.34),
    "s": (0.50, 0.34),
    "m": (0.75, 0.67),
    "l": (1.00, 1.00),
    "x": (1.25, 1.34),
}  # name -> (width, depth)
```

---

## `gusnet.assign`

### `Assignment`

```python
Assignment(fg_mask, target_labels, target_boxes, target_scores, target_gt_index)
```

| Attribute | Shape | Meaning |
|---|---|---|
| `fg_mask` | `(B, A)` bool | points assigned to an object |
| `target_labels` | `(B, A)` int64 | class of the assigned object |
| `target_boxes` | `(B, A, 4)` | xyxy of the assigned object |
| `target_scores` | `(B, A, C)` | soft classification target — **not** one-hot |
| `target_gt_index` | `(B, A)` int64 | which object, in the padded tensors |

Property `num_foreground`; method `to(device)`.

### `TaskAlignedAssigner` *(default)*

```python
TaskAlignedAssigner(num_classes, *, topk=13, alpha=0.5, beta=6.0)
assigner(pred_scores, pred_boxes, points, gt_labels, gt_boxes, gt_mask) -> Assignment
```

Ranks candidates by `s**alpha * u**beta` and keeps the top `topk` per object.
Runs under `no_grad`.

### `SimOTAAssigner`

```python
SimOTAAssigner(num_classes, *, center_radius=2.5, iou_weight=3.0, candidate_topk=10)
assigner(pred_scores, pred_boxes, points, gt_labels, gt_boxes, gt_mask, strides) -> Assignment
```

Note the extra `strides` argument, needed to size the centre region.

### Helpers

```python
targets_to_batch(targets, batch_size) -> (gt_labels, gt_boxes, gt_mask)
points_in_boxes(points, boxes, *, eps=0.01) -> (B, M, A) bool
points_in_centers(points, boxes, strides, *, radius=2.5) -> (B, M, A) bool
resolve_conflicts(mask_pos, overlaps) -> (mask_pos, fg_mask, target_gt_index)
```

---

## `gusnet.losses`

### Components

```python
VarifocalLoss(*, alpha=0.75, gamma=2.0, reduction="sum")
    forward(logits, targets)            # (..., C) each

IoULoss(*, kind="ciou", reduction="none")
    forward(pred, target)               # (..., 4) xyxy each

DistributionFocalLoss(reg_max=16, *, reduction="none")
    forward(logits, target)             # (N, 4, reg_max+1) and (N, 4) in bin units
```

`reduction` ∈ `"sum"`, `"mean"`, `"none"` for all three.

### `DetectionLoss`

```python
DetectionLoss(num_classes, *, reg_max=16, assigner=None,
              cls_weight=0.5, box_weight=7.5, dfl_weight=1.5, iou_kind="ciou")

total, breakdown = criterion(output, targets)
assignment = criterion.assign_only(output, targets)   # diagnostics
```

`output` is a `DetectionOutput`; `targets` is the `(N, 6)` table from the
collate. Returns the scalar to backpropagate plus a `LossBreakdown`.

### `LossBreakdown`

```python
LossBreakdown(total, cls, box, dfl, num_foreground)
breakdown.items()  # {"total": ..., "cls": ..., "box": ..., "dfl": ..., "fg": ...}
```

All tensors are detached; the terms are already multiplied by their weights and
sum to `total`.

---

## `gusnet.eval`

### `non_max_suppression`

```python
non_max_suppression(output, *, conf_threshold=0.25, iou_threshold=0.45,
                    max_det=300, max_nms=30000,
                    class_agnostic=False, multi_label=False) -> list[Detections]
```

`output` is a `DetectionOutput` (or any mapping with `boxes` `(B, A, 4)` and
`scores` `(B, A, C)`). Returns one `Detections` per image, sorted by descending
score. Use `conf_threshold=0.001, iou_threshold=0.7` when measuring mAP and the
defaults when showing results to a person.

### `Detections`

```python
Detections(boxes, scores, labels)  # (N,4) xyxy, (N,), (N,) int64
len(detections)
detections.to(device)
detections.filter(min_score)
detections.scale_to_original(ratio, pad, orig_shape)
```

### `GroundTruth`

```python
GroundTruth(boxes, labels)  # (N,4) xyxy, (N,) int64
```

Must be in the **same coordinate space** as the detections it is matched
against.

### `MeanAveragePrecision`

```python
MeanAveragePrecision(num_classes, *, iou_thresholds=DEFAULT_IOU_THRESHOLDS)
metric.update(detections, truth)  # one image at a time
metric.compute()  # -> MetricResult
metric.reset()
```

`DEFAULT_IOU_THRESHOLDS` is COCO's `(0.5, 0.55, ..., 0.95)`.

### `MetricResult`

| Attribute | Meaning |
|---|---|
| `map50_95` | mAP averaged over IoU 0.50:0.95 — the headline number |
| `map50` | mAP at IoU 0.50 |
| `map75` | mAP at IoU 0.75 |
| `precision`, `recall` | at the best-F1 point of the IoU 0.50 curve |
| `ap_per_class` | `{class index: AP averaged over thresholds}` |
| `num_images`, `num_objects`, `num_detections` | counts |

```python
result.items()  # the scalars, ready to log
result.format(class_names)  # a printable summary table
```

### `average_precision`

```python
average_precision(recalls, precisions) -> float
```

Area under one precision/recall curve: made monotonically non-increasing, then
sampled at 101 evenly spaced recall levels.

### `evaluate` and `EvalConfig`

```python
EvalConfig(
    batch_size=16,
    img_size=640,
    conf_threshold=0.001,
    iou_threshold=0.7,
    max_det=300,
    workers=0,
    device="auto",
    half=False,
    verbose=True,
)

evaluate(model, dataset, config=None) -> MetricResult
```

Forces `dataset.augment = False`, and maps both predictions and ground truth
back to original-image coordinates before matching.

### `detections_to_rows`

```python
detections_to_rows(detections, image_id) -> list[dict]
```

COCO-style records with `bbox` as `[x, y, width, height]`.

---

## `gusnet.train`

### `TrainConfig`

```python
TrainConfig(
    epochs=100,
    batch_size=16,
    img_size=640,
    optimizer="sgd",
    lr0=0.01,
    lr_final_factor=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    close_mosaic=15,
    grad_clip=10.0,
    amp=True,
    ema_decay=0.9999,
    ema_tau=2000.0,
    val_interval=1,
    eval_conf_threshold=0.001,
    eval_iou_threshold=0.7,
    workers=0,
    device="auto",
    seed=0,
    save_dir=Path("runs/train"),
    log_interval=10,
)
```

Methods: `resolved_device()`, `as_dict()`.

### `Trainer`

```python
Trainer(model, dataset, config=None, *, criterion=None, val_dataset=None)
history = trainer.train()     # list of per-epoch metric dicts
```

Attributes: `model`, `ema`, `optimizer`, `loader`, `history`, `device`,
`amp_enabled`, `val_dataset`. Each history entry holds `total`, `cls`, `box`,
`dfl`, `fg`, `lr`, `seconds`, plus `mAP50-95`, `mAP50`, `mAP75`, `precision`
and `recall` on epochs that were validated.

Writes `save_dir/last.pt` (with optimiser state, resumable) every epoch and
`save_dir/best.pt` (EMA weights, deployable) when the model improves — by
**mAP** when `val_dataset` is given, and by training loss otherwise. Epochs
that were not validated never compete for `best.pt`.

### `ModelEMA`

```python
ModelEMA(model, *, decay=0.9999, tau=2000.0, updates=0)
ema.update(model)
ema.current_decay() -> float
ema.state_dict() / ema.load_state_dict(state)
ema.ema                       # the averaged nn.Module
```

### Optimiser and schedule

```python
build_optimizer(model, *, name="sgd", lr=0.01, momentum=0.937, weight_decay=5e-4)
cosine_schedule(epochs, final_factor=0.01) -> Callable[[int], float]
warmup_factor(iteration, total_warmup) -> float
seed_everything(seed)
```

`build_optimizer` returns two parameter groups: index 0 decayed (conv/linear
weights), index 1 undecayed (norm weights and all biases).

### Checkpoints

```python
save_checkpoint(path, model, *, ema=None, optimizer=None,
                epoch=0, metrics=None, config=None) -> Path
load_checkpoint(path, map_location="cpu") -> dict
model_from_checkpoint(path, *, prefer_ema=True, map_location="cpu") -> (model, checkpoint)
```

A checkpoint contains `format`, `gusnet_version`, `epoch`, `model`,
`model_args`, `metrics`, `config`, and optionally `ema` and `optimizer`.
`model_args` is what lets the architecture be rebuilt without guessing.

---

## `gusnet.predict`

### `PredictConfig`

```python
PredictConfig(
    img_size=640,
    conf_threshold=0.25,
    iou_threshold=0.45,
    max_det=300,
    batch_size=8,
    device="auto",
    half=False,
)
```

### `Predictor`

```python
Predictor(model, *, class_names=None, config=None)
Predictor.from_checkpoint(path, *, config=None, prefer_ema=True)

predictor.predict(images)  # list[np.ndarray] -> list[Detections]
predictor.predict_one(image)  # np.ndarray -> Detections
predictor.run(source)  # -> iterator of (name, image, detections)
predictor.annotate(image, detections)  # -> np.ndarray
```

Images are `(H, W, 3)` uint8 **RGB**. Detections come back in each image's own
pixel coordinates, never letterboxed ones.

### `iter_source`

```python
iter_source(source)  # -> iterator of (name, image)
```

Accepts an image file, a directory of images, or a video. `VIDEO_SUFFIXES`
lists the recognised video extensions.

---

## `gusnet.export`

### `ExportWrapper`

```python
ExportWrapper(model)  # forward(images) -> (boxes, scores)
```

A GUSNet that returns tensors instead of a dict.

### `SuppressedModel`

```python
SuppressedModel(core, conf_threshold=0.25, iou_threshold=0.45, max_det=300)
# forward(images) -> (N, 6) of [x1, y1, x2, y2, score, class]
```

Scriptable by construction, which is what keeps `N` dynamic. Batch size 1 only.

### `export_torchscript`

```python
export_torchscript(model, path, *, img_size=640, nms=False,
                   conf_threshold=0.25, iou_threshold=0.45, max_det=300) -> Path
```

Bit-exact with PyTorch. With `nms=True` the network is traced and the
suppression scripted around it.

### `export_onnx`

```python
export_onnx(model, path, *, img_size=640, opset=17, dynamic_batch=False,
            nms=False, conf_threshold=0.25, iou_threshold=0.45,
            max_det=300) -> Path
```

Requires the `export` extra. `dynamic_batch` and `nms` cannot be combined.
Agrees with PyTorch to about `1e-5`.

### `benchmark`

```python
benchmark(model, *, img_size=640, batch_size=1, iterations=50,
          warmup=10, device="auto", half=False) -> dict
```

Returns `ms_per_image`, `ms_per_batch`, `fps`, `batch_size`, `img_size`.

---

## `gusnet.viz`

```python
draw_boxes(image, boxes, classes=None, *, class_names=None, scores=None,
           thickness=2, font_scale=0.45) -> np.ndarray
draw_points(image, points, classes=None, *, radius=3) -> np.ndarray
save_batch_preview(batch, path, *, class_names=None, max_images=16) -> Path
tensor_to_image(image) -> np.ndarray          # (3,H,W) float -> (H,W,3) uint8
class_color(index) -> (r, g, b)
```

All drawing functions copy their input; none modify it.

---

## `gusnet.cli`

```
gusnet --version
gusnet check-data   [--root DIR | --coco-images DIR --coco-annotations JSON]
                    [--split S] [--imgsz N] [--batch-size N] [--no-augment]
                    [--out PATH] [--seed N]
gusnet check-assign [data options] [--model {n,s,m,l,x}] [--assigner {tal,simota}]
                    [--topk N] [--batch-size N] [--out PATH] [--seed N]
gusnet model-info   [--model {n,s,m,l,x}] [--classes N] [--imgsz N]
gusnet train        [data options] [--model {n,s,m,l,x}] [--epochs N]
                    [--batch-size N] [--optimizer {sgd,adamw}] [--lr0 F]
                    [--weight-decay F] [--warmup-epochs F] [--close-mosaic N]
                    [--assigner {tal,simota}] [--val-split S] [--val-interval N]
                    [--workers N] [--device D] [--no-amp] [--seed N]
                    [--save-dir DIR] [--log-interval N]
gusnet val          [data options] --weights PATH [--batch-size N] [--conf F]
                    [--iou F] [--max-det N] [--workers N] [--device D]
                    [--half] [--no-ema]
gusnet predict      --weights PATH --source PATH [--imgsz N] [--conf F]
                    [--iou F] [--max-det N] [--batch-size N] [--device D]
                    [--half] [--no-ema] [--out DIR] [--no-save] [--fps F]
gusnet export       --weights PATH [--format {onnx,torchscript}] [--imgsz N]
                    [--opset N] [--dynamic] [--nms] [--conf F] [--iou F]
                    [--out PATH] [--no-ema]
gusnet benchmark    [--weights PATH | --model {n,s,m,l,x} --classes N]
                    [--imgsz N] [--batch-size N] [--iterations N]
                    [--warmup N] [--device D] [--half]
```

`main(argv=None) -> int` is the entry point; `build_parser()` returns the
`argparse.ArgumentParser` if you want to extend it.
