# Contributing to GUSNet

Thanks for your interest. Two rules matter more than everything else in this
file, so they come first.

## 1. Never bring in copyleft code

GUSNet is Apache-2.0 so that anyone — including companies — can use it without
paying for a license and without being forced to open their own source. A single
copied function from an AGPL or GPL project would destroy that.

**Do not copy, adapt, translate, or transcribe code from:**

- Ultralytics YOLOv5 / v8 / v11 / v12 (AGPL-3.0)
- YOLOv6, YOLOv7, YOLOv9 (GPL-3.0)
- YOLOv10 (AGPL-3.0 by inheritance)
- YOLO-NAS / super-gradients (non-commercial license)
- any project whose license you have not checked

This includes weights: do not submit a model fine-tuned from Ultralytics
checkpoints, and do not submit a model distilled from one.

Reimplement from the paper, or from a permissively licensed reference listed in
[PROVENANCE.md](PROVENANCE.md). If you adapt from a permissive project, say so
in your pull request and add the entry to `PROVENANCE.md`.

## 2. Sign your commits (DCO)

Every commit must carry a `Signed-off-by` line:

```bash
git commit -s -m "your message"
```

By signing off you certify the Developer Certificate of Origin 1.1
(https://developercertificate.org/) — in short, that you wrote the code or
otherwise have the right to submit it under Apache-2.0.

There is no CLA. You keep the copyright on your contribution.

## Development setup

```bash
uv sync --extra dev        # install deps (CPU wheels by default)
uv run pytest              # run the test suite
uv run ruff check .        # lint
uv run ruff format .       # format
```

For CUDA, install the matching PyTorch wheel:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Code conventions

- Every source file starts with `# SPDX-License-Identifier: Apache-2.0`.
- Type hints on public functions; `from __future__ import annotations` at the top.
- Tensor shapes documented in docstrings, e.g. `(B, N, 4)`.
- Box format is stated in the name: `xyxy`, `cxcywh`, `ltrb`. Never guess.
- No hidden global state, no monkey-patching, no `sys.path` manipulation.
- New numerical code comes with a test, however small.

## Pull requests

1. One logical change per PR.
2. Tests pass (`uv run pytest`) and lint is clean.
3. If the change affects accuracy, report the metric you measured and on which
   dataset and split.
4. Describe the source of any algorithm you implemented (paper or permissive
   reference).
