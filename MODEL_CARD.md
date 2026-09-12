# Model Card — GUSNet

> Status: **no released weights yet.** This card is the template every released
> checkpoint must fill in. It exists from day one because weight provenance is
> as legally significant as code provenance.

## Model details

- **Name:** GUSNet-<size> (n / s / m / l)
- **Architecture:** one-stage anchor-free detector — CSP-style backbone,
  PAN-FPN neck, decoupled head with Distribution Focal Loss regression.
- **Framework:** PyTorch
- **License of the weights:** Apache-2.0, same as the code.
- **Author:** Gustavo da Costa Vicente

## Training provenance

This is the section that matters for license compliance. Every released
checkpoint must state:

- **Trained by:** GUSNet's own `gusnet.train` code at commit `<sha>`.
- **Initialized from:** random init, or a named permissively licensed backbone
  (e.g. a `timm` ImageNet checkpoint, Apache-2.0). **Never** from an Ultralytics
  or other AGPL/GPL checkpoint.
- **Distilled from:** nothing. No teacher model under a copyleft or
  non-commercial license was used.
- **Dataset:** name, version, split, and the license of its annotations.

## Data

- **COCO 2017** — annotations are CC BY 4.0 (COCO Consortium); the images are
  hosted by Flickr under individual licenses. GUSNet does **not** redistribute
  COCO images; the download script fetches them from the official source and the
  user accepts the COCO terms of use.
- Alternatives with clearer terms: **Objects365**, **Open Images V7**
  (annotations CC BY 4.0).

## Intended use

General-purpose object detection: research, prototyping, and commercial
integration.

## Out of scope / limitations

GUSNet is **not** validated for safety-critical use. Do not deploy it as the
sole decision-making component in:

- autonomous vehicles or any system controlling physical motion,
- medical diagnosis or triage,
- biometric identification, surveillance, or law-enforcement targeting,
- any application where a false negative or false positive causes physical,
  legal, or financial harm to a person.

Detection accuracy degrades on domains far from the training distribution
(thermal, aerial, microscopy, heavy occlusion, small objects at long range).
The training data carries the geographic and demographic biases of its source;
performance is not uniform across contexts.

## Warranty

None. The software and weights are provided "AS IS", without warranties or
conditions of any kind, as stated in sections 7 and 8 of the Apache License 2.0.
