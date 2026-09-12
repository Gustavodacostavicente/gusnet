# GUSNet

**A clean-room, permissively licensed one-stage object detector in PyTorch.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

GUSNet is an anchor-free, single-stage object detector written from scratch.
It is licensed under **Apache-2.0**: use it in a commercial product, keep your
own source closed, no paid license, no obligations beyond attribution.

> **Status: complete and untrained.** Data pipeline, network, label assignment,
> losses, training, COCO-style mAP evaluation, inference and export are all
> implemented and tested — the eight-phase roadmap is done, and the whole chain
> has been run end to end on real COCO data. What is missing is a *long* run:
> a full schedule is about six days of GPU on one RTX 3060, so **there are no
> released weights yet**. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
> for how it works and [`docs/TRAINING.md`](docs/TRAINING.md) for what it
> costs.

## Why this exists

The best-known YOLO implementations are copyleft: Ultralytics ships under
AGPL-3.0, YOLOv6/v7/v9 under GPL-3.0. Using them in a product means either
open-sourcing your whole application or buying a commercial license. GUSNet is
an independent implementation built from published papers so that it can be
Apache-2.0 all the way down — including the weights.

Every algorithm's origin is documented in [`PROVENANCE.md`](PROVENANCE.md). No
line of code comes from a GPL or AGPL project.

## Architecture

```
image 640x640
   │
   ├─ CSP backbone ──►  P3 (1/8)  ─┐
   │                    P4 (1/16) ─┤
   │                    P5 (1/32) ─┤
   │                               │
   ├─ PAN-FPN neck ────────────────┤
   │                               │
   └─ Decoupled head per scale ────┘
          ├─ classification ──► (B, A, num_classes)
          └─ regression     ──► (B, A, 4×(reg_max+1))  →  DFL → ltrb → xyxy
```

Anchor-free: every grid point predicts the distances to the four box edges as a
discrete distribution (Distribution Focal Loss) rather than a direct regression.
At 640 px that is 8400 predictions per image, with no anchor boxes anywhere.

| Variant | Params (80 classes) |
|---|---|
| GUSNet-n | 2.5 M |
| GUSNet-s | 9.8 M |
| GUSNet-m | 26.8 M |
| GUSNet-l | 56.0 M |
| GUSNet-x | 100.7 M |

## Install

```bash
git clone https://github.com/Gustavodacostavicente/gusnet
cd gusnet
uv sync --extra dev
```

For CUDA training, install the matching PyTorch build:

```bash
uv pip install --reinstall-package torch --reinstall-package torchvision     torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> **Then stop using `uv run`.** It re-syncs the environment against the lock
> file on every invocation, which silently reinstalls the CPU build of PyTorch
> and leaves you wondering why `--device cuda` says *"Torch not compiled with
> CUDA enabled"*. Call the entry point in the virtualenv directly instead —
> `.venv\Scripts\gusnet` on Windows, `.venv/bin/gusnet` elsewhere — or pass
> `uv run --no-sync`.

## Getting COCO

```bash
python scripts/download_coco.py --split val2017                  # 1 GB, 5k images
python scripts/download_coco.py --split train2017                # 19 GB, 118k images
python scripts/download_coco.py --split val2017 --subset 512     # a smoke-test slice
```

Fetches from `images.cocodataset.org` directly. GUSNet does not redistribute
COCO: the annotations are CC BY 4.0 and the images are Flickr-hosted under
individual licences, so using them means accepting the
[COCO terms of use](https://cocodataset.org/#termsofuse).

## Usage

Train on a dataset laid out as `images/<split>/` + `labels/<split>/` + `classes.txt`:

```bash
gusnet train --root datasets/mydata --model s --imgsz 640 --epochs 300              --batch-size 16 --device cuda
```

or on COCO-style annotations:

```bash
gusnet train --coco-images datasets/coco/train2017              --coco-annotations datasets/coco/annotations/instances_train2017.json
```

Add a validation split and `best.pt` is selected by mAP instead of by training
loss:

```bash
gusnet train --root datasets/mydata --val-split val --val-interval 5 ...
```

Checkpoints land in `runs/train/` as `last.pt` (resumable) and `best.pt`
(deployable, EMA weights, no optimiser state). Score a checkpoint at any time:

```bash
gusnet val --root datasets/mydata --weights runs/train/best.pt --imgsz 640
```

```
images 24  objects 87  detections 5092
mAP50-95 0.2790   mAP50 0.7471   mAP75 0.1518
precision 1.0000   recall 0.7826
```

Run it on an image, a folder or a video:

```bash
gusnet predict --weights best.pt --source photo.jpg --conf 0.25 --out runs/predict
```

Export it, and measure it:

```bash
gusnet export    --weights best.pt --format onnx --imgsz 640      # or torchscript
gusnet export    --weights best.pt --format onnx --nms            # NMS in the graph
gusnet benchmark --weights best.pt --imgsz 640 --batch-size 1
```

TorchScript exports are bit-exact with PyTorch; ONNX agrees to about `1e-5`
and needs the `export` extra (`uv sync --extra export`, Python 3.11+).
With `--nms` the detection count stays genuinely dynamic in both formats — which
is harder than it sounds, and is
[explained here](docs/ARCHITECTURE.md#10-inference-and-export).

### Diagnostics

```bash
# load a dataset, run the full augmentation pipeline, write a preview grid
gusnet check-data --root datasets/mydata --imgsz 640 --out runs/check.jpg

# build a model and report its shapes and parameter count
gusnet model-info --model s --classes 80 --imgsz 640

# run an assigner on a real batch and draw the grid points it selected
gusnet check-assign --root datasets/mydata --assigner tal --out runs/assign.jpg
```

```python
from gusnet.data import DetectionDataset, build_dataloader

# images/<split>/*.jpg + labels/<split>/*.txt + classes.txt
ds = DetectionDataset.from_folder("datasets/mydata", "train", img_size=640, augment=True)

# ...or a COCO instances_*.json
ds = DetectionDataset.from_coco("datasets/coco/train2017", "annotations/instances_train2017.json")

loader = build_dataloader(ds, batch_size=16)
batch = next(iter(loader))
batch["images"]  # (B, 3, 640, 640) float32 in [0, 1]
batch["targets"]  # (M, 6) -> [batch_index, class, x1, y1, x2, y2]
```

Mosaic, mixup, random affine, HSV jitter, horizontal flip and letterboxing are
all implemented, with the box bookkeeping tested in both directions.

```python
import torch

from gusnet.models import GUSNet

model = GUSNet.from_variant("s", num_classes=80).eval()
with torch.no_grad():
    out = model(torch.rand(1, 3, 640, 640))

out["boxes"]  # (1, 8400, 4) xyxy in image pixels
out["scores"]  # (1, 8400, 80) per-class confidence
out["cls_logits"], out["reg_logits"]  # raw outputs, for the loss
out["points"], out["strides"]  # (8400, 2) and (8400, 1), for the assigner
```

Two label assigners are implemented and interchangeable:

```python
from gusnet.assign import TaskAlignedAssigner, targets_to_batch

gt_labels, gt_boxes, gt_mask = targets_to_batch(batch["targets"], batch_size=16)
assignment = TaskAlignedAssigner(num_classes=80, topk=13)(
    out["scores"], out["boxes"], out["points"], gt_labels, gt_boxes, gt_mask
)
assignment.fg_mask  # (B, A) which grid points are responsible for an object
assignment.target_boxes  # (B, A, 4) the box each one must predict
assignment.target_scores  # (B, A, C) soft target: confidence tied to IoU
```

And the full objective, which runs assignment and scores all three terms:

```python
from gusnet.losses import DetectionLoss

criterion = DetectionLoss(num_classes=80)
total, breakdown = criterion(model(images), batch["targets"])
total.backward()
print(breakdown.items())  # {'total': ..., 'cls': ..., 'box': ..., 'dfl': ..., 'fg': ...}
```

And the training loop, which drives all of it:

```python
from gusnet.train import TrainConfig, Trainer

config = TrainConfig(epochs=300, batch_size=16, img_size=640, device="cuda")
history = Trainer(model, dataset, config).train()
```

It runs warmup, cosine decay, an EMA of the weights, mixed precision, gradient
clipping, and switches mosaic off for the final epochs.

And evaluation, implemented from the definition rather than imported:

```python
from gusnet.eval import EvalConfig, evaluate

result = evaluate(model, val_dataset, EvalConfig(img_size=640, device="cuda"))
result.map50_95, result.map50, result.ap_per_class
```

Verified end to end. The test suite overfits a single image from scratch and
checks the most confident box lands on the object; an oracle model that reports
the ground truth exactly must score mAP 1.0 through the real evaluation loop;
and every export is checked both for numerical agreement with PyTorch and for
keeping its detection count dynamic.

## Documentation

| Document | What it covers |
|---|---|
| [`docs/STATUS.md`](docs/STATUS.md) | Where the project stands and what is pending, by priority (Portuguese) |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How everything works and why it is built that way — the full walkthrough, feature by feature |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Training on COCO for real: CUDA setup, measured cost on an RTX 3060, and the platform problems that showed up |
| [`docs/API.md`](docs/API.md) | Every public class and function, with signatures and shapes |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | What is done, what is next (Portuguese) |
| [`docs/LICENCIAMENTO.md`](docs/LICENCIAMENTO.md) | The licensing reasoning behind the project (Portuguese) |
| [`PROVENANCE.md`](PROVENANCE.md) | Where every algorithm came from, and which projects were deliberately not consulted |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Weight provenance and limitations |

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Released weights (when available) are Apache-2.0 as well, trained only by this
repository's own code. See [MODEL_CARD.md](MODEL_CARD.md) for training data
provenance and for the limitations that come with it — GUSNet is **not**
validated for safety-critical use.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Commits must be signed off (`git commit -s`,
Developer Certificate of Origin). Do not submit code adapted from AGPL/GPL
projects.

## Trademark notice

"YOLO" and "Ultralytics" are trademarks of Ultralytics Inc. GUSNet is an
independent project, not affiliated with or endorsed by Ultralytics. References
to YOLO in this documentation are descriptive only.
