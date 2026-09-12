# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Gustavo da Costa Vicente
"""Fetch COCO 2017 from the official source.

Downloads come from ``images.cocodataset.org`` and nowhere else. Redistributed
copies of COCO are common and convenient, but many of them are packaged inside
copyleft repositories, and a dataset script that quietly pulls from one of those
would undo the provenance discipline the rest of this project keeps.

What this does **not** do is vendor the data. COCO's annotations are CC BY 4.0
from the COCO Consortium; the images are hosted by Flickr under individual
licences and are not GUSNet's to redistribute. This script fetches them on your
behalf, and using them means accepting the COCO terms of use:
https://cocodataset.org/#termsofuse

Usage::

    # 1 GB: the validation split, enough to exercise the whole pipeline
    python scripts/download_coco.py --split val2017

    # 19 GB: the real training split
    python scripts/download_coco.py --split train2017

    # a small training set carved out of val2017, for a smoke test
    python scripts/download_coco.py --split val2017 --subset 256

The result is plain COCO layout, which GUSNet reads directly::

    gusnet train --coco-images datasets/coco/val2017 \\
                 --coco-annotations datasets/coco/annotations/instances_val2017.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE_IMAGES = "http://images.cocodataset.org/zips"
BASE_ANNOTATIONS = "http://images.cocodataset.org/annotations"

#: Approximate download sizes, so the script can warn before a long wait.
SIZES_GB = {"val2017": 1.0, "train2017": 19.3, "annotations": 0.25}


def _report(done: int, total: int, started: float, label: str) -> None:
    """A single rewritten progress line — no dependency on tqdm."""
    elapsed = max(time.perf_counter() - started, 1e-6)
    speed = done / elapsed / 1e6
    if total > 0:
        percent = 100 * done / total
        eta = (total - done) / max(done / elapsed, 1)
        tail = f"{percent:5.1f}%  ETA {eta / 60:5.1f} min"
    else:
        tail = "size unknown"
    sys.stdout.write(f"\r  {label}: {done / 1e6:8.1f} MB  {speed:5.1f} MB/s  {tail}   ")
    sys.stdout.flush()


def download(url: str, destination: Path) -> Path:
    """Download ``url`` to ``destination``, skipping it if already complete.

    The file lands at a ``.part`` path first and is renamed only once the
    transfer finishes, so an interrupted download can never be mistaken for a
    complete one on the next run.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"  {destination.name}: already downloaded")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    started = time.perf_counter()
    print(f"  {destination.name}: fetching {url}")

    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:  # noqa: S310
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            _report(done, total, started, destination.name)

    print()
    partial.rename(destination)
    return destination


def unzip(archive: Path, target: Path, marker: Path) -> None:
    """Extract ``archive`` into ``target`` unless ``marker`` already exists."""
    if marker.exists():
        print(f"  {archive.name}: already extracted")
        return
    print(f"  {archive.name}: extracting")
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(target)


def make_subset(annotation_file: Path, count: int, out_file: Path) -> int:
    """Write a smaller annotations file holding only the first ``count`` images.

    Useful for proving the pipeline works before committing to a multi-day run:
    the classes, the annotation format and the image statistics are all real,
    only the volume is not.
    """
    with annotation_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    kept = data["images"][:count]
    keep_ids = {image["id"] for image in kept}
    data["images"] = kept
    data["annotations"] = [a for a in data["annotations"] if a["image_id"] in keep_ids]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)

    print(f"  subset: {len(kept)} images, {len(data['annotations'])} objects -> {out_file.name}")
    return len(kept)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download COCO 2017 from the official source.")
    parser.add_argument(
        "--split",
        default="val2017",
        choices=("val2017", "train2017"),
        help="which image split to fetch (val2017 is 1 GB, train2017 is 19 GB)",
    )
    parser.add_argument(
        "--root", type=Path, default=Path("datasets/coco"), help="where to put everything"
    )
    parser.add_argument(
        "--subset", type=int, help="also write a reduced annotations file with N images"
    )
    parser.add_argument(
        "--keep-zips", action="store_true", help="do not delete the archives after extracting"
    )
    args = parser.parse_args(argv)

    root: Path = args.root
    needed = SIZES_GB[args.split] + SIZES_GB["annotations"]
    free = shutil.disk_usage(root.parent if root.parent.exists() else Path(".")).free / 1e9
    print(f"COCO 2017 -> {root.resolve()}")
    print(f"about {needed:.1f} GB to download, {free:.0f} GB free on that drive\n")
    if free < needed * 1.6:  # the zips and the extracted copy coexist for a while
        print("not enough free space for the download plus the extracted copy")
        return 1

    print("annotations:")
    annotations_zip = download(
        f"{BASE_ANNOTATIONS}/annotations_trainval2017.zip", root / "annotations_trainval2017.zip"
    )
    unzip(annotations_zip, root, root / "annotations" / f"instances_{args.split}.json")

    print(f"\n{args.split}:")
    images_zip = download(f"{BASE_IMAGES}/{args.split}.zip", root / f"{args.split}.zip")
    unzip(images_zip, root, root / args.split)

    if args.subset:
        print("\nsubset:")
        make_subset(
            root / "annotations" / f"instances_{args.split}.json",
            args.subset,
            root / "annotations" / f"instances_{args.split}_subset{args.subset}.json",
        )

    if not args.keep_zips:
        for archive in (annotations_zip, images_zip):
            archive.unlink(missing_ok=True)
        print("\narchives removed (pass --keep-zips to keep them)")

    images = root / args.split
    annotations = root / "annotations" / f"instances_{args.split}.json"
    print("\nready. train with:")
    print(f"  gusnet train --coco-images {images} \\")
    print(f"               --coco-annotations {annotations} \\")
    print("               --model s --imgsz 640 --batch-size 16 --device cuda")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
