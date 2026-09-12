# CLAUDE.md

Instructions for Claude Code working on GUSNet. Read this first, every session.

## Start here, always

1. **Read [`docs/STATUS.md`](docs/STATUS.md)** before doing anything else. It is
   the living record of what is done, what is in progress, and what is pending,
   with priorities. Everything else in this file is stable background; that file
   is the current state.
2. **Update `docs/STATUS.md` at the end of the session**, as part of the same
   commit as the work. A status file that lags the code is worse than none.
3. Answer Gustavo in **Portuguese**. The code, the docstrings and the public
   docs stay in English.

## What this is

GUSNet is a one-stage anchor-free object detector in PyTorch, written from
scratch so it can be Apache-2.0 all the way down — including the weights. The
entire point of the project is that a company can use it commercially without
buying a licence, which is exactly what Ultralytics' AGPL prevents.

The eight-phase roadmap is complete. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for how every part works and why, [`docs/API.md`](docs/API.md) for the reference,
and [`docs/TRAINING.md`](docs/TRAINING.md) for what training actually costs.

## Non-negotiable rules

**Never copy, adapt or consult code from a GPL/AGPL project.** Specifically:
Ultralytics (YOLOv5/v8/v11/v12, AGPL-3.0), YOLOv6, YOLOv7, YOLOv9 (GPL-3.0),
YOLOv10 (AGPL by inheritance), YOLO-NAS (non-commercial). One copied function
would force the whole project to AGPL and destroy its reason to exist. This
covers weights too: a weight file is a derivative of the code that produced it.

Permitted references, all permissive: YOLOX, PP-YOLOE, RT-DETR, D-FINE,
DAMO-YOLO, MMDetection (Apache-2.0), torchvision (BSD), timm (Apache-2.0),
pycocotools (BSD). When adapting from one, record it in
[`PROVENANCE.md`](PROVENANCE.md). When in doubt, reimplement from the paper.

**"YOLO" is a trademark of Ultralytics Inc.** Never use it as a product name.
Descriptive use ("a YOLO-style detector") is fine.

Every `.py` file starts with `# SPDX-License-Identifier: Apache-2.0` — CI fails
without it. Every commit is signed off (`git commit -s`) — CI fails without it
on pull requests.

## Working on this machine

Gustavo's setup: Windows 10, RTX 3060 12 GB, 12 CPU cores, `uv` for Python.

* **`C:` is nearly full (~6 GB free).** Keep datasets, checkpoints and the uv
  cache on `G:`. Set `UV_CACHE_DIR=G:/uv-cache` for anything that downloads.
* **CUDA PyTorch must be installed explicitly**, and then `uv run` must be
  avoided: it re-syncs against the lock file and silently reinstalls the CPU
  build, even on an unrelated `uv run ruff check`. Use `.venv/Scripts/python`
  and `.venv/Scripts/gusnet` directly, or `uv run --no-sync`. Full instructions
  in [`docs/TRAINING.md`](docs/TRAINING.md#1-setting-up-cuda).
* **`--workers 4` is the ceiling here.** Eight exhausts Windows shared memory
  (error 1455) because worker tensors go through the paging file on the full
  `C:` drive. Four holds the GPU at 88%.

## Commands

```bash
.venv/Scripts/python -m pytest            # 275 tests, ~55 s
.venv/Scripts/python -m ruff check .      # lint
.venv/Scripts/python -m ruff format .     # format (also formats code in .md)

.venv/Scripts/gusnet check-data   --root DATA --out runs/check.jpg
.venv/Scripts/gusnet check-assign --root DATA --assigner tal
.venv/Scripts/gusnet train        --coco-images ... --coco-annotations ... --device cuda
.venv/Scripts/gusnet val          --weights best.pt --root DATA
.venv/Scripts/gusnet predict      --weights best.pt --source photo.jpg
.venv/Scripts/gusnet export       --weights best.pt --format onnx
.venv/Scripts/gusnet benchmark    --model s --device cuda
```

## How to work here

**Look at images, not only at numbers.** Every bug found in this project so far
was found by rendering something or by asking a question a shape check does not
ask. `check-data` and `check-assign` exist for that. A loss that is *small* is
not the same as a loss that is *satisfied*.

**Tests assert properties, not values.** "CIoU ranks a near miss above a far
miss", "the detection count differs across inputs", "a perfect detector scores
mAP 1.0". Numbers chosen arbitrarily become flaky across platforms — that
already happened once with a confidence threshold of 0.5.

**Comments explain why, not what.** The existing code says what breaks if a
choice is reversed. Match that. Do not add comments that restate the line above.

**Write up what went wrong.** `docs/ARCHITECTURE.md` has a "things learned the
hard way" section and `docs/TRAINING.md` has one for platform problems. Real
failures with real numbers belong there; they are the most useful part of the
documentation.

## Repository

`https://github.com/Gustavodacostavicente/gusnet` — public, Apache-2.0.

The `gh` CLI has two accounts on this machine. Run
`gh auth switch --user Gustavodacostavicente` before any GitHub operation; the
other account (`gdcvicente`) does not own this repository.
