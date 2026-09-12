# How GUSNet works

A complete walkthrough of everything implemented, in the order the data moves
through it. Every section says *what* the component does, *why* it is built that
way, and *what breaks* if it is not.

This is the design document. For the legal reasoning behind the project see
[`LICENCIAMENTO.md`](LICENCIAMENTO.md) (Portuguese) and
[`../PROVENANCE.md`](../PROVENANCE.md); for what is done and what is next, see
[`ROADMAP.md`](ROADMAP.md).

---

## Table of contents

1. [The problem, in one page](#1-the-problem-in-one-page)
2. [The pipeline end to end](#2-the-pipeline-end-to-end)
3. [Box representations (`gusnet.ops.boxes`)](#3-box-representations)
4. [Data (`gusnet.data`)](#4-data)
5. [Network (`gusnet.nn`, `gusnet.models`)](#5-network)
6. [Label assignment (`gusnet.assign`)](#6-label-assignment)
7. [Losses (`gusnet.losses`)](#7-losses)
8. [Training (`gusnet.train`)](#8-training)
9. [Evaluation (`gusnet.eval`)](#9-evaluation)
10. [Inference and export (`gusnet.predict`, `gusnet.export`)](#10-inference-and-export)
11. [Visualisation (`gusnet.viz`)](#11-visualisation)
12. [Command line (`gusnet.cli`)](#12-command-line)
13. [Testing strategy](#13-testing-strategy)
14. [Things learned the hard way](#14-things-learned-the-hard-way)
15. [What comes next](#15-what-comes-next)

---

## 1. The problem, in one page

Object detection asks a model to answer two questions at once for an unknown
number of objects: *what* is in the image and *where*. That "unknown number" is
what makes it harder than classification — the output has no fixed shape.

One-stage detectors solve it by turning the question inside out. Instead of
proposing objects, they cover the image in a dense grid of points and ask every
point the same two fixed-shape questions:

* is there an object here, and of which class?
* if so, where are its four edges relative to me?

At 640×640 with strides 8, 16 and 32 that is 6400 + 1600 + 400 = **8400 points**,
each producing one class vector and one box. Detection then becomes a
fixed-shape regression plus classification problem, and the real difficulty
moves somewhere unexpected: deciding *which* of those 8400 points is responsible
for which object. That decision is called label assignment, it is section 6, and
it is where most of the accuracy in a modern detector actually comes from.

GUSNet is **anchor-free**: a grid point is just a point, and the box is
described by four distances from it. There are no anchor boxes, no aspect-ratio
priors, and nothing to tune per dataset.

---

## 2. The pipeline end to end

```
   dataset on disk
        │
        │  gusnet.data.DetectionDataset
        ▼
   decode → letterbox / mosaic → affine → HSV → flip
        │
        │  gusnet.data.build_dataloader
        ▼
   batch: images (B,3,640,640)   targets (M,6) = [img, class, x1,y1,x2,y2]
        │
        │  gusnet.models.GUSNet
        ▼
   CSPBackbone → PAFPN → DetectHead
        │
        ▼
   cls_logits (B,8400,C)   reg_logits (B,8400,4,17)   points (8400,2)
        │
        │  gusnet.assign.TaskAlignedAssigner          ◄── no_grad
        ▼
   assignment: fg_mask, target_boxes, target_scores
        │
        │  gusnet.losses.DetectionLoss
        ▼
   VarifocalLoss + CIoU + DFL  →  one scalar
        │
        │  gusnet.train.Trainer
        ▼
   AMP backward → clip → step → EMA → checkpoint
        │
        │  gusnet.eval  (validation, and gusnet val)
        ▼
   NMS → detections → match against ground truth → mAP
```

Every arrow in that diagram is covered by tests, and the whole chain is verified
by an overfit test that trains a model from random initialisation on a single
image and checks the resulting box lands on the object.

---

## 3. Box representations

**Module:** `gusnet/ops/boxes.py`

Three formats are used, and every function name says which one it takes. Silent
format confusion is the single most common source of detection bugs, so the
convention is enforced by naming rather than by comments.

| Format | Meaning | Used by |
|---|---|---|
| `xyxy` | `(x1, y1, x2, y2)` absolute corners | datasets, NMS, evaluation, losses |
| `cxcywh` | `(cx, cy, w, h)` centre and size | label files, some augmentations |
| `ltrb` | distances from a point to the four edges | the anchor-free regression target |

`ltrb` is the one worth dwelling on. It is what makes the head anchor-free: the
grid point at `(px, py)` describes a box by saying "the left edge is `l` pixels
to my left, the top edge is `t` pixels above me", and so on. Converting back is
just `(px - l, py - t, px + r, py + b)`.

### Overlap metrics

`box_iou(a, b)` builds the full `N × M` matrix — used by the assigner, which
needs every object against every prediction.

`bbox_iou(a, b, kind=...)` is element-wise with broadcasting, used by the loss
once predictions are already matched to targets. Four variants:

* **IoU** — the metric itself. Its flaw: for two boxes that do not overlap it is
  zero regardless of how far apart they are, so it provides no gradient
  direction at exactly the moment a gradient would be most useful.
* **GIoU** — adds a penalty from the smallest box enclosing both, restoring a
  direction for non-overlapping boxes.
* **DIoU** — penalises the distance between centres instead, which converges
  faster than GIoU's area term.
* **CIoU** *(default)* — DIoU plus an aspect-ratio consistency term. A box with
  the right centre and area but the wrong shape is still penalised.

`scale_boxes(boxes, ratio, pad, orig_shape)` undoes a letterbox, mapping
predictions back to the original image's coordinates. Its correctness is tested
as a round trip: original box → letterbox → scale back → the same box.

---

## 4. Data

**Module:** `gusnet/data/`

### 4.1 `letterbox.py` — resize without distortion

Detectors take a fixed-size square input; images come in arbitrary shapes.
Squashing a 16:9 photo into a square distorts every object in it, and a model
trained on distorted objects has to learn each aspect ratio twice.

`letterbox` scales by a single factor and fills the leftover area with neutral
grey (114), returning a `LetterboxResult` carrying the ratio, the padding and
the original shape — everything needed to map predictions back.

Details that matter:

* **`INTER_AREA` when shrinking**, `INTER_LINEAR` when growing. Bilinear
  downsampling aliases: it samples a sparse set of pixels and drops the rest, so
  thin structures flicker or vanish. `INTER_AREA` averages over the source
  region.
* **`scaleup=False` at inference.** Enlarging a small image adds no information
  and costs accuracy.
* **`center=False`** places the image top-left instead of padding both sides,
  which is what mosaic tiles need.
* **`stride=32`** pads only to the next multiple of the stride instead of
  filling a full square — "rectangular inference", noticeably cheaper on
  non-square images.

### 4.2 `transforms.py` — augmentation

All functions take `(image, boxes, classes)` as plain numpy, with boxes in
absolute `xyxy` pixels, and every geometric operation transforms the boxes with
the image. `filter_boxes` runs afterwards to drop what the transform destroyed:
boxes under 2 px, aspect ratios over 20:1, or boxes that kept less than 10% of
their original area after a crop.

**`mosaic4`** stitches four images into a canvas of side `2 × img_size` around a
random centre. It is the single most effective augmentation here, for two
reasons: it multiplies the number of objects per batch (more positives per
gradient step) and it puts objects at scales and positions they never occupy in
their own images, which is most of why it helps small-object accuracy. The
canvas is deliberately twice the training size — a following `random_affine`
with `border=(-img_size//2, -img_size//2)` crops it back down, and that crop is
part of the augmentation.

**`random_affine`** applies rotation, scale, shear and translation as a single
3×3 matrix, then transforms each box by mapping its four corners and taking the
enclosing box. (That is why a rotated box grows: an axis-aligned box cannot
represent a rotated one.)

**`hsv_augment`** jitters hue, saturation and value through a lookup table —
building three 256-entry LUTs and applying them is far cheaper than arithmetic
over every pixel. Hue wraps modulo 180 because OpenCV packs it into a byte.

**`mixup`** blends two samples with a weight drawn from `Beta(32, 32)`, which
concentrates near 0.5 and so produces a genuine double exposure rather than a
barely visible ghost. Labels from both images are concatenated.

### 4.3 `dataset.py` — two layers

`Annotation` is a pure description of one labelled image: a path, boxes and
classes. `DetectionDataset` turns those into tensors. The split matters because
reading a dataset format is cheap and format-specific, while the augmentation
pipeline is expensive and shared.

Boxes are stored **normalised** `xyxy` in `[0, 1]`, because the image size is
not known until the file is decoded. They become absolute pixels in `load_raw`.

Two readers:

* **`from_folder`** — `images/<split>/` + `labels/<split>/` + `classes.txt`,
  with one `<class> <cx> <cy> <w> <h>` line per object, normalised. A missing
  label file is a legitimate negative sample, not an error.
* **`from_coco`** — a COCO `instances_*.json`. COCO stores `[x, y, w, h]` in
  absolute pixels and numbers its categories with gaps, so category ids are
  remapped to a contiguous zero-based range.

`__getitem__` returns a dict with the image as `(3, H, W)` float in `[0, 1]`,
boxes in the coordinates of that image, and the letterbox parameters needed to
map predictions back.

`AugmentConfig` holds the augmentation strengths. The two probabilities at the
top — `mosaic` and `mixup` — are the ones worth tuning first, and `mosaic` is
the one the trainer switches off near the end of the schedule.

### 4.4 `loader.py` — batching

Detection batches are awkward: every image has a different number of objects.
Two representations are produced and both are kept:

* `boxes` / `classes` — lists of per-image tensors, natural for evaluation and
  drawing;
* `targets` — one `(M, 6)` tensor of `[batch_index, class, x1, y1, x2, y2]`
  covering the whole batch, which is what the assigner and the loss want.

`num_workers=0` is the default because on Windows each worker re-imports the
package, and for small datasets that costs more than it saves.

---

## 5. Network

**Modules:** `gusnet/nn/`, `gusnet/models/detector.py`

### 5.1 `blocks.py` — the four building pieces

**`ConvNormAct`** — Conv2d → BatchNorm2d → SiLU. The convolution carries no
bias because the batch norm right after it has one; keeping both is redundant
parameters and makes conv/bn fusion at export messier.

**`Bottleneck`** — a 1×1 convolution then a 3×3, optionally residual. The 1×1
mixes channels cheaply and the 3×3 does the spatial work. Two 3×3s would cost
about 80% more for the same widths, and since the backbone is almost entirely
bottlenecks that choice alone moved GUSNet-l from 76 M to 56 M parameters.

**`CSPLayer`** — the cross-stage partial stage. The input is projected twice:
one branch runs through `depth` bottlenecks, the other skips them entirely, and
the concatenation is fused by a final 1×1. Half the gradient path therefore
never touches the bottleneck stack, which is the duplicated gradient flow
CSPNet was designed to remove — and it costs less compute than putting every
channel through the stack.

**`SPP`** — spatial pyramid pooling, so the deepest stage can see context far
wider than its kernels without another stride. Two equivalent implementations:

* parallel max-pools with kernels 5, 9 and 13;
* *(default)* a cascade of three 5×5 pools.

These are **exactly** equivalent, because the maximum of maxima over
overlapping windows is the maximum over their union: two stacked stride-1 5×5
pools cover a 9×9 window, three cover 13×13. The cascade is cheaper because each
stage pools an already pooled map. A test asserts both forms agree bit for bit —
if that identity ever stops holding, the cheap path would be silently changing
the architecture.

**`make_divisible`** rounds scaled channel counts to a multiple of 8 (fast paths
on tensor cores, friendlier to quantisation) with a 90% floor so rounding never
halves a small layer.

### 5.2 `backbone.py` — CSPBackbone

Five stages, each halving the spatial size:

| Stage | Output | Stride | Role |
|---|---|---|---|
| stem | 320×320 | 2 | |
| stage1 | 160×160 | 4 | |
| stage2 | 80×80 | 8 | **P3** — small objects |
| stage3 | 40×40 | 16 | **P4** — medium objects |
| stage4 | 20×20 | 32 | **P5** — large objects (+ SPP) |

Two multipliers scale the family: `width` changes channels per stage, `depth`
changes bottlenecks per CSP stage. Base channels `(64, 128, 256, 512, 1024)`,
base depths `(3, 6, 6, 3)`.

### 5.3 `neck.py` — PAFPN

The backbone's three maps are good at different things: P3 has fine spatial
detail but weak semantics, P5 the opposite. The neck gives every level both, in
two passes:

* **top-down** (FPN) — semantics flow from P5 down into P4 and P3 through
  upsampling;
* **bottom-up** (PANet) — localisation detail flows from P3 back up through
  strided convolutions, shortening the path a small object's features travel to
  reach the deep levels.

The test that matters here is not a shape check: perturbing P5 must change N3,
and perturbing P3 must change N5. A neck wired backwards still produces
perfectly shaped tensors.

### 5.4 `head.py` — DetectHead

Two decisions define it.

**Decoupled branches.** Classification and localisation want different features
— one needs to know *what* is there, the other *where its edges are* — so each
gets its own stack of convolutions rather than sharing one output tensor.

**Distances as distributions.** Instead of regressing four numbers, every point
predicts four discrete distributions over `0..reg_max` (in stride units), and
the reported distance is their expectation:

```
reg_logits (B, A, 4, 17) ──softmax──► p ──Σ k·p(k)──► ltrb in strides
                                                      × stride → pixels
                                                      + point   → xyxy
```

A blurry or occluded edge then shows up as a *flat* distribution instead of a
confident wrong number, and the shape of that distribution is itself a usable
quality signal. With `reg_max=16` and stride 32, one point can describe an edge
up to 512 px away — enough for any box at 640 px input.

**`make_anchor_points`** generates the grid points in image pixels, at cell
centres (`offset=0.5`), ordered x-fastest to match the `(B, H*W, C)` flattening,
and concatenated level by level. It returns the per-point stride too, which both
the decode and the assigner need.

**Initialisation** sets the classification bias so every logit starts at a
confidence of 0.01. Without it, 8400 points all claim an object on step one and
the resulting loss spike can stall training outright.

### 5.5 `detector.py` — GUSNet

Backbone + neck + head, in five sizes:

| Variant | width | depth | Params (80 classes) |
|---|---|---|---|
| n | 0.25 | 0.34 | 2.5 M |
| s | 0.50 | 0.34 | 9.8 M |
| m | 0.75 | 0.67 | 26.8 M |
| l | 1.00 | 1.00 | 56.0 M |
| x | 1.25 | 1.34 | 100.7 M |

The forward pass rejects inputs not divisible by 32 with a message saying so,
rather than failing deep inside a concatenation with a shape mismatch.

Weight init is Kaiming for convolutions, and BatchNorm gets `eps=1e-3`,
`momentum=0.03` instead of the defaults: those are tuned for classification with
large batches, while detection trains with far fewer images per batch and so has
noisier running statistics. (This choice has a consequence — see section 12.)

---

## 6. Label assignment

**Module:** `gusnet/assign/`

This is the part that decides what the model is even being asked to learn, and
it is where most of the accuracy lives.

### 6.1 The shared machinery (`utils.py`)

* **`targets_to_batch`** pads the flat `(N, 6)` table into `(B, M)` tensors plus
  a mask marking the real entries.
* **`points_in_boxes`** — a point can only be a candidate for an object it is
  inside, because the head describes a box by *positive* distances to its edges.
* **`points_in_centers`** — a square of side `2 × radius × stride` around the
  object's centre. Points near the centre see the whole object; and for a very
  large object the in-box mask alone would nominate thousands of candidates.
* **`resolve_conflicts`** — one point, one object. Two overlapping objects can
  nominate the same point, and training it towards both would average them into
  a box matching neither, so the tie goes to the object it already localises
  best.
* **`Assignment`** — the output: `fg_mask`, `target_labels`, `target_boxes`,
  `target_scores` and `target_gt_index`.

### 6.2 TaskAlignedAssigner *(default)*

The problem it solves is that classification and localisation drift apart. A
point can be confident about the class and put the box in the wrong place, or
localise perfectly and score low. NMS then keeps confident boxes that are badly
placed.

The task-aligned metric scores each candidate with both at once:

```
t = s**alpha * u**beta          alpha = 0.5, beta = 6.0
```

`s` is the predicted confidence for the object's class, `u` the IoU of the
point's predicted box with the object. The top `k` (13) candidates per object
are selected from the points inside it.

The part that does the real work is the **target**: the classification target is
not 1, it is that same metric rescaled so the object's best candidate is trained
towards the best IoU it achieved. A point that localises poorly is therefore
trained towards a *low* confidence — so the score the model outputs comes to
mean "this box is good", which is exactly what NMS needs it to mean.

Measured on a controlled scene with one object:

| Predictions | Positives | Classification target |
|---|---|---|
| perfect (IoU 1.0) | 13 | 1.000 |
| poor (IoU 0.06) | 13 | 0.061 |

### 6.3 SimOTAAssigner

Assignment read as a transport problem: each object has a supply of labels, each
point a demand, every pairing a cost. Solving it exactly (Sinkhorn, as in OTA)
is expensive; SimOTA keeps the two ideas that carry the benefit:

* **dynamic k** — an object's supply is estimated from the IoUs of its top-10
  candidates, so a large clearly-found object gets many positives and a small or
  occluded one gets few. A fixed k has to be wrong for one of those cases.
* **a cost combining both tasks** — `BCE(√score) + 3·(−log IoU)`, with
  candidates outside the centre region priced out by a constant large enough to
  never be selected. The square root softens the classification term, which is
  otherwise pure noise from a head that has not learned anything yet;
  `−log(IoU)` grows steeply near zero, which is what makes a badly localised
  candidate genuinely expensive rather than merely slightly worse.

Same controlled scene:

| Predictions | Positives |
|---|---|
| perfect (IoU 1.0) | 10 |
| poor (IoU 0.06) | 1 |

**Why TAL is the default.** That second row is SimOTA's weakness here. On an
untrained model every IoU is near zero, dynamic k collapses to one positive per
object, and the model starves for gradient exactly when it needs it most. YOLOX
compensates with a separate objectness branch; GUSNet has none, so TAL — which
delivers a fixed 13 candidates from step one — is what the trainer uses. SimOTA
stays available for comparison.

---

## 7. Losses

**Module:** `gusnet/losses/`

### 7.1 The three terms

**`VarifocalLoss`** — classification, over the assigner's soft targets. The two
sides are not symmetric, so they are not weighted the same:

* positives carry a target `q ∈ (0, 1]` and are weighted by `q` itself;
* negatives have `q = 0` and are weighted by `alpha · p**gamma`
  (`alpha = 0.75`, `gamma = 2`). Once the model is confident a point is
  background, `p` is near zero and the weight vanishes, so the loss concentrates
  on the few background points that still look like objects.

Plain cross-entropy would drown in the thousands of easy negatives and would
train confidence towards a constant 1, disconnecting the score from box quality.

**`IoULoss`** — `1 − CIoU`. An L1 loss on the four coordinates optimises
something that is not the metric: two boxes with the same coordinate error can
have very different overlap.

**`DistributionFocalLoss`** — cross-entropy against the two bins bracketing the
target, linearly interpolated. A target of 4.3 pushes mass onto bins 4 and 5 in
proportion 0.7 / 0.3. Supervising only the expectation would leave the shape
unconstrained: the model could place mass on bins 0 and 9 and still report 4.5,
and the distribution would no longer be readable as a confidence. A test asserts
exactly that case is penalised.

### 7.2 `DetectionLoss` — putting it together

```
output + targets
   │
   ├─ assigner (no_grad, on detached predictions)
   │      └─► fg_mask, target_boxes, target_scores
   │
   ├─ cls  = VFL(cls_logits, target_scores).sum() / target_scores.sum()
   ├─ box  = Σ (1−CIoU)·w / Σ w        over assigned points
   └─ dfl  = Σ DFL(...)·w  / Σ w       over assigned points

total = 0.5·cls + 7.5·box + 1.5·dfl
```

Predictions are **detached** before assignment. That is not an optimisation:
assignment decides what is being asked of the model, and the model must not be
able to lower its loss by changing the question instead of improving the answer.

The classification term is normalised by the sum of target scores rather than by
the count of positives, so the denominator reflects how much quality was
assigned and the gradient does not lurch when the positive count jumps between
batches.

The box terms are weighted **averages** normalised by their own weights. That
detail is not cosmetic — see section 12.

`box_weight = 7.5` is much larger than the others because `1 − CIoU` is bounded
by about 2, while the classification term sums over every class and every point.

---

## 8. Training

**Module:** `gusnet/train/`

### 8.1 `optim.py` — optimiser and schedule

`build_optimizer` splits parameters into two groups:

* convolution and linear **weights** get weight decay — shrinking them is a real
  capacity constraint;
* **normalisation weights and all biases** do not. Decaying a BatchNorm scale
  towards zero does not regularise anything, it squashes the activations the
  scale exists to rescale; and a bias is an offset, so pulling every offset
  towards zero is a bias, not a regulariser.

Lumping them together is the most common silent mistake in a training script: it
costs accuracy and nothing reports it.

`cosine_schedule` holds the rate high for most of the run then anneals smoothly
to 1% of it. The long tail at a small rate is where the model settles into a
minimum instead of bouncing around it.

`warmup_factor` is a linear ramp. A randomly initialised detector handed
thousands of targets at the full learning rate takes a step it does not recover
from.

### 8.2 `ema.py` — ModelEMA

SGD does not converge to a point, it orbits one. The weights at the end of any
given step are one sample from that orbit; a running average lands nearer its
centre. So the EMA copy, not the live model, is what gets saved and deployed.

Two details:

* the decay **ramps in** (`decay · (1 − e^(−updates/tau))`). A fixed 0.9999 at
  step one would cling to the random initialisation for thousands of steps.
* the average covers **buffers**, not just parameters. BatchNorm running
  statistics lag the weights by hundreds of steps early on, and evaluating raw
  weights against stale statistics is a reliable way to conclude a healthy model
  is broken. Integer buffers such as `num_batches_tracked` are counters, not
  quantities, and are copied rather than averaged.

### 8.3 `checkpoint.py`

A checkpoint records the **constructor arguments**, not only the weights. A file
of tensors whose shapes imply an architecture you have to reverse-engineer is a
file you will eventually be unable to load. `model_from_checkpoint` rebuilds the
model from nothing and prefers the EMA weights.

Loading uses `weights_only=True`: a checkpoint is a pickle, and an unrestricted
pickle from an untrusted source executes arbitrary code on load.

`last.pt` carries the optimiser state so a run can resume exactly; `best.pt`
omits it and is about a third of the size.

### 8.4 `trainer.py` — the loop

```
for epoch:
    maybe close mosaic
    for batch:
        set lr and momentum (warmup ramp, then cosine)
        forward under autocast
        loss
        scaled backward
        unscale → clip → step → update scaler
        update EMA
    save last.pt, and best.pt if improved
```

* **AMP** — `torch.amp.autocast` plus a `GradScaler`. The gradient is
  **unscaled before clipping**: clipping a scaled gradient clips the wrong
  thing, since the scale factor changes between steps.
* **Closing mosaic** — mosaic is worth a lot of accuracy, but every image it
  produces is a collage that cannot occur at inference. Training to the last
  step on collages leaves a gap between what the model has seen and what it will
  see, so the final epochs run on plain images. Worker processes hold their own
  copy of the dataset, so the dataloader is rebuilt when this happens.
* **Warmup on three things at once** — learning rate, SGD momentum, and the bias
  group's rate. The bias group starts *high* (0.1) and comes down to `lr0`:
  biases are the fastest thing a detector can usefully learn early, especially
  the classification prior, so they move while the weights are still ramping.

---

## 9. Evaluation

**Module:** `gusnet/eval/`

Until this existed there was no number to compare two runs by. The trainer had
to pick `best.pt` by training loss, which keeps falling long after the model has
stopped generalising.

### 9.1 `nms.py` — dense predictions to a list of objects

The head answers the same question at every grid point, so one object arrives
described by a dozen overlapping boxes. Non-maximum suppression sorts by
confidence, keeps the best box, discards everything overlapping it too much, and
repeats.

Three thresholds, and they are not interchangeable:

| | Measuring mAP | Showing a person |
|---|---|---|
| `conf_threshold` | **0.001** | 0.25 |
| `iou_threshold` | **0.7** | 0.45 |

The confidence one matters most. mAP integrates the whole precision/recall
curve, so cutting predictions at 0.25 truncates the curve and under-reports the
metric, often by several points. And suppressing aggressively removes duplicates
but also removes recall — which is half of what is being measured.

NMS is **per class** by default: a car overlapping a person is two objects, not
a duplicate. `class_agnostic=True` is available for datasets where the classes
are competing descriptions of the same thing. `multi_label` lets one grid point
emit several classes; off by default, since one box per point at its best class
is right unless the dataset genuinely has overlapping labels.

Suppression itself is `torchvision.ops.batched_nms`, which offsets each class
into its own coordinate band so classes cannot suppress one another.

### 9.2 `metrics.py` — mean average precision

Implemented from the definition rather than imported, because almost every
surprising number a detector produces is explained by one of its details.

**The curve.** For one class at one IoU threshold: sort every prediction in the
dataset by confidence, walk down the list, and mark each as a true positive if
it matches a ground-truth box nothing better has already claimed. That traces a
precision/recall curve — accepting lower-confidence predictions raises recall
and lowers precision. AP is the area under it.

**Why the curve is flattened first.** The raw curve is jagged; one lucky
detection can push precision back up. AP uses the maximum precision at or beyond
each recall level, which answers the question that actually matters — *if I need
this much recall, what is the best precision available?* — and makes the number
stable.

**Why ten thresholds.** AP at IoU 0.50 barely rewards good localisation: a box
can be visibly wrong and still count. COCO-style mAP averages AP over IoU 0.50
to 0.95 in steps of 0.05, so a model that puts boxes *almost* in the right place
scores well below one that puts them exactly right. That gap is large and
informative — on the synthetic benchmark here, mAP50 0.75 against mAP50-95 0.28
says plainly that the boxes are being found but not tightly.

**Matching rules**, following COCO:

* a detection can only claim an object of its own class;
* each object can be claimed once, by the highest-confidence detection that
  overlaps it enough — so a duplicate is a false positive, not a second hit;
* a class with no ground truth anywhere contributes **no AP at all**, rather
  than a zero that would drag the mean down with a class the data never tested.

`average_precision` interpolates at 101 evenly spaced recall levels rather than
integrating exactly, which is what makes the number comparable across datasets
of different sizes.

### 9.3 `evaluator.py` — the run loop

`evaluate(model, dataset, config)` runs the model over a dataset and returns a
`MetricResult`. Two things it does on your behalf:

* **forces augmentation off.** Measuring a model on mosaics measures something
  that will never be asked of it. This is a silent and serious mistake, so it is
  not left to the caller to remember.
* **maps everything back to original-image coordinates** before matching. IoU
  happens to be invariant under the shared letterbox scale-and-pad, so this
  changes no number — but every box it reports is then meaningful, and a whole
  class of silent mismatches disappears.

`detections_to_rows` serialises to COCO's `[x, y, width, height]` records, in
one place rather than at every call site that writes results out.

### 9.4 Validation inside training

Pass `val_dataset` to `Trainer` and `best.pt` is selected by mAP instead of by
training loss. The **EMA copy** is what gets evaluated — it is the one that will
be deployed, and early on its BatchNorm statistics are the more settled of the
two.

A 30-epoch run on the synthetic set, validating every 10 epochs:

```
val  mAP50-95 0.0000
val  mAP50-95 0.0441   ← best.pt
val  mAP50-95 0.0139
```

Note that the last epoch is not the best one, while the training loss fell
monotonically throughout. That is the entire argument for selecting by a
held-out metric.

---

## 10. Inference and export

**Modules:** `gusnet/predict.py`, `gusnet/export.py`

### 10.1 Inference

Inference is the training pipeline with almost everything removed, and the few
things that stay differ in ways that matter:

* **no upscaling.** The letterbox runs with `scaleup=False` — enlarging a small
  image adds no information and costs accuracy.
* **a high confidence threshold.** Evaluation uses 0.001 to trace the whole
  precision/recall curve; a person wants 0.25, not 300 boxes sorted by hope.
* **boxes mapped back to the source image.** Everything the model says is in
  letterboxed coordinates, and nothing outside `Predictor` should ever see
  those.

A folder is processed in batches even though the images have different shapes,
because the letterbox makes them the same size before they are stacked — each
one keeps its own ratio and padding for the trip back.

`iter_source` accepts a file, a directory or a video and yields RGB frames.
OpenCV's BGR is converted once, at the boundary, rather than being allowed to
leak inward.

### 10.2 Export, and the one hard part

A deployment runtime wants a graph of tensor operations, not a Python object.
Two things stand in the way.

**The output is a dict** — convenient in Python, inexpressible in ONNX.
`ExportWrapper` flattens it to `(boxes, scores)` and drops what inference does
not need: the raw logits and the anchor points exist for the loss.

**Suppression has a data-dependent output size**, and this is where export
gets genuinely difficult. How many detections there are depends on the pixels,
so a plain trace records whatever count the *example* input produced and freezes
it. The result runs, returns tensors of the right rank, and is wrong for every
other image — silently. Measured here:

```
torch.jit.trace(model_with_nms)  →  detections per input: [0, 0, 0]
```

Zero, forever, because the example was noise and nothing scored above the
threshold. Nothing in that failure looks like a failure.

The two paths solve it differently:

| | Approach | Result |
|---|---|---|
| TorchScript | trace the network, then **script** the suppression around it | counts `[125, 134, 117, 121]` |
| ONNX | the **legacy** tracing exporter, which lowers NMS to the ONNX `NonMaxSuppression` op | counts `[123, 123, 130, 122]` |

Scripted code keeps real control flow and real dynamic shapes; traced code does
not. And the current dynamo-based ONNX exporter cannot represent a
data-dependent output size at all — it fails outright rather than producing
something wrong, which is the better of the two behaviours but still needs the
older path to get a working graph.

Folding NMS in is **off by default** regardless: it freezes the thresholds into
the file, and a threshold that can only be changed by re-exporting is a
threshold nobody tunes.

Fidelity against PyTorch, measured on the same input:

| Format | Max difference in boxes | In scores |
|---|---|---|
| TorchScript | `0.0` (bit-exact) | `0.0` |
| ONNX | `6.1e-05` | `4.0e-07` |

The ONNX difference is the graph optimiser reassociating float arithmetic, not
an error. Both are asserted by tests.

### 10.3 Benchmarking

`benchmark()` measures forward-pass latency. The warmup passes are not optional
noise reduction: the first CUDA call initialises the context and cuDNN picks its
convolution algorithms over the first few passes, so timing those measures the
setup rather than the model. `torch.cuda.synchronize()` before stopping the
clock, for the same reason — CUDA is asynchronous, and without it the timer
measures how fast work can be queued.

---

## 11. Visualisation

**Module:** `gusnet/viz.py`

Visual inspection is the cheapest bug detector in a detection pipeline. A box
off by a factor of two, a flip that did not move its labels, a mosaic tile
pasted at the wrong offset — all invisible in a loss curve, all obvious in a
rendered image.

* `draw_boxes` — boxes with class labels and optional scores.
* `draw_points` — the grid points an assigner selected. A wrong assigner is
  obvious here (positives scattered outside objects, clustered on one edge, far
  too few) and invisible in any scalar.
* `class_color` — a stable colour per class, stepped by the golden ratio in hue
  so neighbouring ids never come out nearly identical.
* `save_batch_preview` — a whole collated batch as one grid.

---

## 12. Command line

```bash
gusnet check-data   --root DATA --imgsz 640 --out runs/check.jpg
gusnet check-assign --root DATA --assigner tal --out runs/assign.jpg
gusnet model-info   --model s --classes 80 --imgsz 640
gusnet train        --root DATA --model s --epochs 300 --device cuda                     --val-split val --val-interval 5
gusnet val          --root DATA --weights runs/train/best.pt
gusnet predict      --weights best.pt --source photo.jpg --conf 0.25
gusnet export       --weights best.pt --format onnx --imgsz 640 [--nms]
gusnet benchmark    --model s --imgsz 640 --batch-size 1
```

Every subcommand is implemented.

The two `check-*` commands exist because every bug found in this project so far
was found by looking at an image, not at a number.

---

## 13. Testing strategy

267 tests. The ones worth knowing about are not the shape checks.

**Round trips.** A box converted to another format and back must be identical;
a box through letterbox and `scale_boxes` must return to where it started.

**Identities.** The fast SPP cascade must equal the parallel pools bit for bit.

**Properties, not values.** CIoU must rank a near miss above a far miss where
plain IoU cannot tell them apart. DFL must prefer a distribution with mass in
the right place over one with merely the right mean. The neck must propagate P5
into N3 and P3 into N5.

**Behaviour under control.** The assigners are driven with predictions
constructed by hand — perfect, then deliberately poor — and checked for the
properties they exist to have: TAL's target tracking the achieved IoU, SimOTA's
k shrinking.

**The end-to-end proof.** A model is trained from random initialisation on a
single image and must (a) collapse its loss and (b) place its most confident box
on the object, in eval mode, with confidence above 0.5. If the chain from data
to gradient is broken anywhere, this fails.

**An oracle for the evaluator.** A stub model that reports the ground truth
exactly is run through the real evaluation loop and must score mAP 1.0. Any
mistake in the coordinate bookkeeping between letterboxed predictions and
original-image ground truth shows up immediately — a perfect detector that
scores 0.8 means the evaluator is wrong, not the model.

**Exports must stay dynamic.** Both NMS export paths are run on four different
inputs and the detection counts must differ. This is the only way that class of
bug is visible: an export with a frozen count produces perfectly shaped, wrong
answers.

**Exports must agree with PyTorch.** TorchScript bit-exactly; ONNX within 1e-3.
An export that runs but computes something else is worse than no export.

---

## 14. Things learned the hard way

Two real bugs, both found by running the thing rather than by reading it.

### The BatchNorm lag

The overfit test failed at IoU 0.07 after 120 steps. The model was **perfect** —
in train mode, from step 50. The gap was entirely BatchNorm: with
`momentum=0.03` the running statistics need roughly 200 updates to catch up with
the batch statistics the network is actually training on, so a model that
already predicts perfectly can look broken in eval mode.

Invisible in real training, where there are thousands of steps. Very visible on
a single repeated image — and it explains bad-looking evaluations in the first
epochs of any run. It is also part of why EMA covers buffers.

### The cold-start collapse

The first real training run produced a loss of **0.0004 on epoch 1** and barely
moved. Not a healthy loss — a false zero.

The task-aligned metric contains `IoU**6`. For an object that nothing yet
overlaps — a small object in the first epochs — every target score comes out
around `1e-8`:

```
large object   fg=13   Σ target_scores = 1.8e+00   max weight = 1.4e-01
small object   fg= 2   Σ target_scores = 3.2e-08   max weight = 3.2e-08
```

The box terms were normalised by the global sum of those scores, so they
inherited that magnitude. The model had **no usable box gradient at all**, and
training stalled exactly where it most needed to move.

The fix: normalise the box terms by their own weights — a weighted average is
scale-free and cannot collapse — with a uniform fallback if the weights
degenerate entirely. The initial loss went from 0.0004 to 11.1, and on the
synthetic benchmark class accuracy rose from 78% to 90%.

The lesson generalises: a loss term that is *small* is not the same as a loss
term that is *satisfied*, and a training curve that looks flat and low deserves
suspicion, not celebration.

### The best-checkpoint mix-up

Wiring validation into the trainer, the first run finished with
`best mAP50-95 11.1202` — a mAP above 1, which is impossible.

With `val_interval=5`, epochs in between produce no mAP, and the selection code
fell back to the training loss for those. Selecting the *maximum* then compared
two quantities on different scales, and epoch 1's loss of 11.12 beat every real
mAP. `best.pt` was the worst checkpoint of the run.

The fix is that epochs without a validation result simply do not compete. The
general lesson is narrower than the previous two but no less useful: a fallback
default that silently changes the *units* of a comparison is worse than no
fallback at all. Better to have nothing to compare than to compare the wrong
thing.

### The export that always returned nothing

Tracing the model with NMS folded in produced a file that ran, loaded, and
returned a correctly shaped tensor — of zero detections, for every input
forever. The trace had been taken on random noise, nothing scored above the
threshold, and `torch.jit.trace` recorded "the answer has zero rows" as a
constant.

There is no error, no warning that survives being read, and no way to notice
from the shapes. It only became visible because the test asked a question a
shape check never does: *run this on four different inputs and confirm the
counts are not all the same.*

Fixed by scripting the suppression instead of tracing it, and by using the
legacy exporter for the ONNX equivalent. The lesson: when a function's output
*shape* depends on its input *values*, tracing is not a safe way to capture it —
and the failure mode is silence, so the test has to go looking.

---

## 15. What comes next

The eight-phase roadmap is complete: GUSNet reads data, trains, measures itself,
runs on real inputs and exports to two runtimes.

**The obvious next thing is weights.** Everything here has been verified on
synthetic data and on overfitting a single image. A real COCO run is what turns
this from a correct implementation into a usable detector, and nothing in the
code can substitute for it.

**Sensible after that:** multi-scale training; DDP for multi-GPU; resuming from
`last.pt` (the state is saved, the flag is not wired); rectangular inference
batching; an ATSS-style static warmup assigner so SimOTA becomes usable from
step one; a cross-check of the mAP implementation against `pycocotools` as an
optional test; and TensorRT or OpenVINO export on top of the ONNX graph.
