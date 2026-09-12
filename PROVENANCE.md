# Code Provenance

GUSNet is a **clean-room** implementation. This file records where every
non-trivial idea in the codebase comes from, so that anyone auditing the project
can verify that it is free of copyleft contamination and safe for commercial use
under the Apache-2.0 license.

## The rule

> No code in this repository may be copied, adapted, or transcribed from a
> project licensed under GPL, AGPL, or a non-commercial license.

Algorithms are reimplemented from the published papers. Where an implementation
detail was taken from another codebase, that codebase is permissively licensed
and is credited below.

## Forbidden sources (never consulted for code)

| Project | License | Why excluded |
|---|---|---|
| Ultralytics YOLOv5 / v8 / v11 / v12 | AGPL-3.0 | Network copyleft; would force GUSNet to AGPL |
| YOLOv6 (Meituan) | GPL-3.0 | Strong copyleft |
| YOLOv7 | GPL-3.0 | Strong copyleft |
| YOLOv9 | GPL-3.0 | Strong copyleft |
| YOLOv10 (THU) | AGPL-3.0 (inherited) | Built on the Ultralytics package |
| YOLO-NAS (Deci) | Non-commercial | Prohibits the intended use |
| Darknet forks with unclear terms | unclear | Provenance cannot be established |

Pretrained weights produced by any of the above are equally excluded: a weight
file is a derivative of the code that produced it. GUSNet ships only weights
trained by this repository's own training code.

## Permitted references (permissive licenses)

| Project | License | Used as reference for |
|---|---|---|
| YOLOX (Megvii) | Apache-2.0 | SimOTA assignment, decoupled head |
| PP-YOLOE (PaddleDetection) | Apache-2.0 | Task-aligned assignment, VFL/DFL formulation |
| RT-DETR / RT-DETRv2 | Apache-2.0 | End-to-end detection, denoising queries |
| D-FINE / DEIM | Apache-2.0 | Fine-grained distribution refinement |
| DAMO-YOLO | Apache-2.0 | Neck design |
| MMDetection / MMYOLO | Apache-2.0 | Module structure, evaluation conventions |
| torchvision | BSD-3-Clause | `batched_nms`, `box_iou`, backbone conventions |
| timm | Apache-2.0 | Optional pretrained backbones |
| pycocotools | BSD-2-Clause | COCO-style mAP evaluation |

## Papers reimplemented from scratch

| Component | Paper |
|---|---|
| One-stage dense detection | Redmon et al., *You Only Look Once* (2016), arXiv:1506.02640 |
| Feature pyramid | Lin et al., *Feature Pyramid Networks* (2017), arXiv:1612.03144 |
| Path aggregation | Liu et al., *Path Aggregation Network* (2018), arXiv:1803.01534 |
| CSP stage design | Wang et al., *CSPNet* (2019), arXiv:1911.11929 |
| SPP / fast SPP | He et al., *SPPNet* (2014), arXiv:1406.4729 |
| Anchor-free decoupled head | Ge et al., *YOLOX* (2021), arXiv:2107.08430 |
| SimOTA assignment | Ge et al., *OTA* (2021), arXiv:2103.14259 |
| Task-aligned assignment | Feng et al., *TOOD* (2021), arXiv:2108.07755 |
| Distribution Focal Loss | Li et al., *Generalized Focal Loss* (2020), arXiv:2006.04388 |
| VariFocal Loss | Zhang et al., *VarifocalNet* (2020), arXiv:2008.13367 |
| CIoU / DIoU loss | Zheng et al., *Distance-IoU Loss* (2019), arXiv:1911.08287 |
| Mosaic / MixUp augmentation | Bochkovskiy et al., *YOLOv4* (2020), arXiv:2004.10934 (paper only) |
| Classification bias prior | Lin et al., *Focal Loss for Dense Object Detection* (2017), arXiv:1708.02002 |

## Derivations made in this repository

Some implementation choices are derived here rather than taken from anywhere:

* **Cascaded spatial pyramid pooling.** Three stride-1 max-pools of kernel 5,
  applied in sequence, cover exactly the windows of single pools of kernel 5, 9
  and 13, because the maximum of maxima over overlapping windows is the maximum
  over their union. `gusnet.nn.blocks.SPP` implements both forms and
  `tests/test_blocks.py::test_fast_spp_is_identical_to_the_parallel_one` asserts
  they agree bit for bit.

## Trademarks

"YOLO" and "Ultralytics" are trademarks of Ultralytics Inc. GUSNet is an
independent project with no affiliation to Ultralytics. References to YOLO in
documentation are descriptive only and are not used as a product name.

## Auditing

Every source file carries an `SPDX-License-Identifier: Apache-2.0` header.
Contributions are accepted under the Developer Certificate of Origin (see
CONTRIBUTING.md), which requires each contributor to certify that they have the
right to submit the code under this project's license.
