# Training GUSNet for real

Everything in [`ARCHITECTURE.md`](ARCHITECTURE.md) describes how the code works.
This describes what actually happens when you point it at COCO on one consumer
GPU — the setup, the measured numbers, and the four things that went wrong the
first time.

All figures here were measured on an **NVIDIA RTX 3060 12 GB**, Windows 10,
12 CPU cores, PyTorch 2.11 + CUDA 12.8.

---

## 1. Setting up CUDA

The default install gives you CPU PyTorch. Replace it:

```bash
uv pip install --reinstall-package torch --reinstall-package torchvision \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

`--reinstall-package` matters: without it uv sees a package called `torch`
already present, decides the requirement is satisfied, and changes nothing.

> ### Then stop using `uv run`
>
> `uv run` re-syncs the environment against the lock file on every invocation.
> The lock file names the CPU build, so any `uv run` — including an unrelated
> `uv run ruff check` — silently reinstalls CPU PyTorch underneath you. The next
> training run then fails with *"Torch not compiled with CUDA enabled"*, with no
> hint that a linting command caused it.
>
> Call the entry point directly instead: `.venv\Scripts\gusnet` on Windows,
> `.venv/bin/gusnet` elsewhere. Or pass `uv run --no-sync`.

Verify:

```bash
.venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"
```

## 2. Getting the data

```bash
python scripts/download_coco.py --split val2017                # 1 GB, 5k images
python scripts/download_coco.py --split train2017              # 19 GB, 118k images
python scripts/download_coco.py --split val2017 --subset 512   # a smoke-test slice
```

Both splits share one annotations archive, so fetching `val2017` also leaves you
`instances_train2017.json` — the 19 GB you would add later is images only.

Check the drive before the big download. And check **which** drive: on Windows
the paging file usually lives on `C:`, and dataloader workers pass tensors
through shared memory backed by it. A nearly full system drive breaks training
in a way that looks nothing like a disk problem — see §5.

## 3. What it costs on one RTX 3060

Measured training throughput, 640 px, AMP, SGD, ~7 objects per image:

| Model | Batch | ms/iter | img/s | Peak VRAM |
|---|---|---|---|---|
| GUSNet-n | 16 | 225 | 71 | 2.0 GB |
| GUSNet-n | 32 | 386 | 83 | 4.0 GB |
| GUSNet-s | 16 | 218 | 74 | 4.3 GB |
| GUSNet-m | 8 | 502 | 16 | 3.2 GB |

VRAM is not the constraint at these sizes — a 12 GB card has room for batch 32
at `s`, which is worth taking.

Projected onto COCO's 118 287 training images:

| Model | Batch | 1 epoch | 100 epochs | 300 epochs |
|---|---|---|---|---|
| GUSNet-n | 32 | 24 min | 1.7 days | 5.0 days |
| GUSNet-s | 16 | 27 min | 1.9 days | 5.6 days |
| GUSNet-m | 8 | 124 min | 8.6 days | 25.8 days |

So a full 300-epoch GUSNet-s schedule is about **six days of uninterrupted
GPU**. That is a real commitment but not an unreasonable one; 100 epochs in
under two days is the sensible first attempt.

Inference, for comparison — 640 px, batch 1: **GUSNet-n 100 FPS, GUSNet-s 64
FPS, GUSNet-m 35 FPS**.

## 4. A smoke test on real COCO

Before committing days of GPU, run the pipeline on a slice. This is step 2 of
the validation order in [`ROADMAP.md`](ROADMAP.md) — memorise a small set and
confirm the whole chain closes on real data rather than on synthetic shapes:

```bash
.venv/Scripts/gusnet train \
  --coco-images        datasets/coco/val2017 \
  --coco-annotations   datasets/coco/annotations/instances_val2017_subset512.json \
  --val-coco-annotations datasets/coco/annotations/instances_val2017_subset512.json \
  --model s --imgsz 640 --batch-size 16 --epochs 100 \
  --optimizer adamw --lr0 0.001 --warmup-epochs 5 --close-mosaic 20 \
  --val-interval 25 --workers 4 --device cuda
```

What that run produced here, in about 12 minutes:

| Epoch | Loss | mAP50-95 | mAP50 |
|---|---|---|---|
| 1 | ~11.1 | — | — |
| 50 | 5.70 | 0.0007 | 0.0019 |
| 100 | 5.14 | 0.0124 | 0.0262 |

**Read that honestly.** The loss falls, mAP rises monotonically, checkpoints and
validation work — the pipeline closes on real COCO. But mAP 0.026 is *barely
started*, not *good*. Eighty classes over 512 photographs in 100 epochs is a
much harder memorisation problem than the synthetic three-shape set, which
reached mAP50 0.75 in 40 epochs. Nothing here substitutes for the real run.

## 5. Four things that went wrong

Each of these cost time, and none of them announced itself.

### Validation inside training hung forever

The first COCO run reached epoch 20, entered its first validation, and stopped.
No error, no output, GPU at 9% — for eight minutes, until it was killed.

Running the same validation standalone took **15 seconds**. The difference:
inside training, the loader's four persistent workers are alive when
`evaluate()` spawns four more, and on Windows that combination deadlocks.

Fixed by giving in-training validation `workers=0` unconditionally rather than
inheriting the training config. It needs no workers anyway: there is no mosaic
during evaluation, so each sample decodes one image instead of four.

### Eight workers exhausted Windows shared memory

Raising `--workers` to 8 produced:

```
RuntimeError: Couldn't open shared file mapping: <torch_14092_...>, error code: <1455>
```

Error 1455 is `ERROR_NO_SYSTEM_RESOURCES`. Dataloader workers hand tensors back
through shared memory backed by the paging file, and the system drive here had
6 GB free. Four workers is what this machine supports; the number is a property
of your pagefile, not of your CPU.

With four workers the GPU sits at **88% utilisation** and an epoch over 512
images takes 7.2 s — 71 img/s, matching the synthetic benchmark exactly. The
dataloader is not the bottleneck at this scale, which is worth knowing before
anyone starts optimising it.

### `--val-split` silently validated on the training set

With a COCO dataset, `--val-split anything` was accepted and quietly built the
validation set from the *training* annotations. The run reports a memorisation
score in the place where a generalisation score belongs, and picks `best.pt` by
it — the single most expensive kind of wrong number, because it looks right.

Fixed: folder datasets take `--val-split`, COCO datasets take
`--val-coco-annotations`, and mixing them is now an error rather than a guess.

### The ONNX exporter killed the process over an emoji

Covered in [`ARCHITECTURE.md`](ARCHITECTURE.md#14-things-learned-the-hard-way):
torch's exporter prints a check mark, a legacy Windows console cannot encode it,
and the `UnicodeEncodeError` takes down the command *after* the export has
already succeeded. The CLI now forces UTF-8 with replacement on its streams.

## 6. Running the real thing

```bash
python scripts/download_coco.py --split train2017

.venv/Scripts/gusnet train \
  --coco-images          datasets/coco/train2017 \
  --coco-annotations     datasets/coco/annotations/instances_train2017.json \
  --val-coco-images      datasets/coco/val2017 \
  --val-coco-annotations datasets/coco/annotations/instances_val2017.json \
  --model s --imgsz 640 --batch-size 16 --epochs 100 \
  --warmup-epochs 3 --close-mosaic 15 \
  --val-interval 5 --workers 4 --device cuda \
  --save-dir runs/coco-s
```

Notes for a run of that length:

* **`--val-interval 5`**, not 1. Validating on all 5000 val2017 images costs
  about two and a half minutes; every epoch would add four hours over 100
  epochs.
* **`best.pt` is chosen by mAP**, and only epochs that were validated compete
  for it. `last.pt` carries optimiser state.
* **SGD is the default** and is the right choice at this length; AdamW converges
  faster on short schedules and small data, which is why the smoke test above
  uses it.
* **Resuming is not wired up yet.** The state is in `last.pt`, the flag is not
  there. On a six-day run that is a gap worth closing before you start.
