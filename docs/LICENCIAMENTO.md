# Anotações — Licenciamento e procedência (GUSNet)

> Notas de trabalho em português. A versão pública e formal dessas decisões está
> em `PROVENANCE.md`, `NOTICE` e `CONTRIBUTING.md`.
> Nada aqui é aconselhamento jurídico.

## 1. O problema central: AGPL é contagiosa

A Ultralytics licencia YOLOv5/v8/v11 sob **AGPL-3.0**. Copiar qualquer trecho
relevante — mesmo adaptado ou traduzido — obriga o GUSNet a virar AGPL, o que
significa liberar o código-fonte inclusive para quem só usa o modelo via rede/API.
É exatamente disso que a licença comercial paga deles vive.

Solução: **clean-room**. Implementar a partir dos *papers* e de referências com
licença permissiva. Nunca a partir do código deles.

## 2. Mapa de licenças

**Proibido** (contamina o projeto):

| Projeto | Licença |
|---|---|
| Ultralytics YOLOv5/v8/v11/v12 | AGPL-3.0 |
| YOLOv6 (Meituan) | GPL-3.0 |
| YOLOv7 | GPL-3.0 |
| YOLOv9 | GPL-3.0 |
| YOLOv10 | AGPL-3.0 (herdada do pacote ultralytics) |
| YOLO-NAS / Deci | não comercial |

**Permitido** (permissivo, basta atribuir):

| Projeto | Licença | Serve para |
|---|---|---|
| YOLOX (Megvii) | Apache-2.0 | SimOTA, head desacoplada, anchor-free |
| PP-YOLOE (PaddleDetection) | Apache-2.0 | TAL assigner, VFL/DFL |
| RT-DETR / RT-DETRv2 | Apache-2.0 | detecção end-to-end sem NMS |
| D-FINE / DEIM | Apache-2.0 | estado da arte em real-time detection |
| DAMO-YOLO | Apache-2.0 | neck, backbone |
| MMDetection / MMYOLO | Apache-2.0 | estrutura de módulos |
| torchvision | BSD-3 | `nms`, `box_iou`, backbones |
| timm | Apache-2.0 | backbones pré-treinados |
| pycocotools | BSD-2 | avaliação mAP |

Regra prática: não escrever código logo depois de ler o repo da Ultralytics.
Usar o paper e o YOLOX/PP-YOLOE como referência visual.

## 3. Marca registrada — por isso o nome não é "YOLO"

"YOLO" e "Ultralytics" são marcas registradas da Ultralytics Inc. Licença de
código e marca são coisas separadas: mesmo com código 100% próprio, usar "YOLO"
no nome do pacote/produto expõe o projeto. Daí **GUSNet** (de Gustavo).
No README pode-se dizer "detector no estilo YOLO" — uso nominativo/descritivo é
aceitável; nome de produto não é.

## 4. Pesos pré-treinados também têm licença

Erro que quase todo mundo comete. Um arquivo de pesos é derivado do código que o
gerou:

- pesos `.pt` da Ultralytics são **AGPL** → não redistribuir, não fazer
  fine-tune e publicar, não destilar;
- treinar do zero, ou partir de backbones `timm`/`torchvision` (Apache/BSD);
- **COCO**: anotações são CC BY 4.0, mas as imagens são do Flickr com licenças
  individuais → distribuir só os pesos e apontar para o download oficial;
- alternativas com termos mais claros: Objects365, Open Images V7;
- publicar `MODEL_CARD.md` dizendo com que código e que dados os pesos foram
  treinados. É isso que protege.

## 5. Por que Apache-2.0 e não MIT

Ambas permitem uso comercial fechado, mas a Apache tem:

- **concessão expressa de patentes** (art. 3) + cláusula de retaliação — relevante
  em visão computacional, área minada de patentes;
- **disclaimer de garantia e limitação de responsabilidade** mais robusto
  (arts. 7 e 8): `AS IS, WITHOUT WARRANTIES ... IN NO EVENT SHALL ANY
  CONTRIBUTOR BE LIABLE`. É literalmente o "sem maiores responsabilidades";
- exige `NOTICE`, o que força documentar atribuições — boa higiene.

Não é preciso nada além disso para se eximir de responsabilidade. Opcional e
recomendável: aviso de que o modelo não é validado para uso em segurança crítica
(já está no `MODEL_CARD.md`).

## 6. Higiene do repositório

```
LICENSE            Apache-2.0 completo, com linha de copyright preenchida
NOTICE             atribuições de terceiros
PROVENANCE.md      de onde veio cada ideia, com licença
CONTRIBUTING.md    + DCO
MODEL_CARD.md      origem dos pesos e dos dados
```

- Header SPDX em todo arquivo: `# SPDX-License-Identifier: Apache-2.0`
- **DCO** (`git commit -s`) em vez de CLA — é o padrão do Linux e do PyTorch.
  Cada contribuidor certifica que tem direito sobre o código enviado; protege
  contra alguém colar código AGPL num PR.
- Regra explícita no `CONTRIBUTING.md` listando os repos proibidos.

## 7. Checklist antes de publicar no GitHub

- [ ] `LICENSE` com o nome correto do titular
- [ ] SPDX em todos os `.py`
- [ ] `PROVENANCE.md` atualizado com tudo que foi adaptado
- [ ] nenhum peso de terceiros commitado
- [ ] nenhuma imagem de dataset commitada (só scripts de download)
- [ ] nome do pacote no PyPI não contém "yolo"
- [ ] `MODEL_CARD.md` preenchido para cada checkpoint publicado
- [ ] para uso comercial sério: revisão com advogado de PI
