# Roteiro de desenvolvimento — GUSNet

Notas de trabalho. Cada fase precisa rodar e ser testada antes da seguinte.
Regra de procedência em `docs/LICENCIAMENTO.md`.

## Arquitetura alvo

Detector one-stage **anchor-free**, no espírito do YOLOX/PP-YOLOE:

```
imagem 640x640
   │
   ├─ Backbone CSP  ──► P3 (80x80, 1/8)   ──┐
   │                    P4 (40x40, 1/16)  ──┤
   │                    P5 (20x20, 1/32)  ──┤
   │                                        │
   ├─ Neck PAN-FPN  (top-down + bottom-up) ─┤
   │                                        │
   └─ Head desacoplada por escala ──────────┘
          ├─ branch cls  ──► (B, A, num_classes)
          └─ branch reg  ──► (B, A, 4*(reg_max+1))  distribuição DFL → ltrb
```

Sem âncoras: cada ponto da grade prevê distâncias `l,t,r,b` até as bordas da
caixa, como distribuição discreta (DFL) em vez de regressão direta.

## Fases

### Fase 1 — Fundação de dados ✅
- [x] `gusnet.ops.boxes`: conversões xyxy/cxcywh/ltrb, IoU, CIoU, clipping
- [x] `gusnet.data.letterbox`: resize com padding preservando aspect ratio
- [x] `gusnet.data.dataset`: dataset de detecção + formato de anotação
- [x] `gusnet.data.transforms`: flip, HSV, mosaic, mixup
- [x] `gusnet.data.loader`: collate de batches com nº variável de caixas
- [x] `gusnet.viz`: desenhar caixas para inspeção visual
- [x] testes unitários

### Fase 2 — Backbone + Neck ✅
- [x] `ConvNormAct`, `Bottleneck`, `CSPLayer`, `SPP` (cascata rápida, com teste
      provando equivalência com os pools paralelos 5/9/13)
- [x] `CSPBackbone` com larguras/profundidades escaláveis (n/s/m/l/x)
- [x] `PAFPN` (top-down + bottom-up)
- [x] teste de shapes: entrada 640 → P3/P4/P5 com canais corretos
- [x] teste de que P5 influencia N3 e P3 influencia N5 (fiação do neck)

### Fase 3 — Head anchor-free ✅
- [x] branches cls e reg separadas, com stem próprio por escala
- [x] projeção DFL (`reg_max=16`), decode para xyxy em coordenadas de imagem
- [x] geração de pontos de grade + strides (`make_anchor_points`)
- [x] init do bias de classificação com prior 0.01 (evita pico de loss inicial)
- [x] teste: forward completo produz (B, A, C) e (B, A, 4); 640 → 8400 pontos

### Fase 4 — Label assignment ✅
- [x] `TaskAlignedAssigner` (TOOD/PP-YOLOE) — métrica s^α · u^β, topk por objeto,
      alvo de classificação = métrica normalizada (não one-hot)
- [x] `SimOTA` (OTA/YOLOX) — dynamic-k a partir da soma dos top-10 IoUs,
      custo = BCE(√score) + 3·(−log IoU) + penalidade fora da região central
- [x] `targets_to_batch`: tabela plana (N,6) → tensores (B,M) com máscara
- [x] `resolve_conflicts`: um ponto, um objeto (desempate por IoU)
- [x] teste: cena sintética com 1 objeto atribui pontos dentro da caixa
- [x] teste: alvo de classificação cai junto com o IoU alcançado
- [x] teste: dynamic-k dá mais positivos a objeto fácil que a difícil
- [x] `gusnet check-assign` desenha os pontos escolhidos sobre a imagem

> Nota prática observada: com modelo não treinado, o TAL entrega ~13 positivos
> por objeto e o SimOTA colapsa para ~1 (todos os IoUs perto de zero). É o
> comportamento esperado de cada um — o SimOTA depende de predições minimamente
> razoáveis para o dynamic-k abrir. **Default do treino: TAL.**

### Fase 5 — Losses ✅
- [x] `VarifocalLoss` — alvo soft do assigner, negativos ponderados por α·p^γ
- [x] `IoULoss` com CIoU/DIoU/GIoU/IoU
- [x] `DistributionFocalLoss` — CE contra os dois bins vizinhos, interpolado
- [x] `DetectionLoss` — junta assigner + 3 termos, normaliza pela soma dos
      target_scores, pondera os termos de caixa pelo score de cada positivo
- [x] pesos default: cls 0.5, box 7.5, dfl 1.5
- [x] teste: loss cai >65% em overfit de 1 imagem
- [x] teste: caixa mais confiante chega a IoU > 0.7 em modo eval
- [x] teste: DFL prefere massa nos dois bins vizinhos, não só a média certa

> Nota operacional descoberta aqui: com `momentum=0.03` no BatchNorm, as
> running stats levam ~200 passos para alcançar as estatísticas de batch. Num
> overfit de 1 imagem o modelo acerta em `train()` no passo 50 mas só acerta em
> `eval()` lá pelo passo 200. Em treino real isso é invisível, mas explica
> avaliações ruins nas primeiras épocas — considerar na fase 6 (EMA ajuda).

### Fase 6 — Treino ✅
- [x] loop com AMP (`torch.amp`) e grad clipping (unscale antes do clip)
- [x] `ModelEMA` com ramp de decay; média cobre buffers do BN
- [x] cosine LR + warmup de lr/momentum/bias-lr; SGD nesterov ou AdamW
- [x] `build_optimizer` sem weight decay em BN e biases
- [x] mosaic desligando nas últimas N épocas (rebuild do dataloader)
- [x] checkpoint com `model_args` — dá para reconstruir a arquitetura do zero
- [x] `gusnet train`
- [x] treino real validado: 24 imagens sintéticas, GUSNet-n, 40 épocas →
      best-IoU médio 0.71, 93% dos objetos com IoU > 0.5, classe certa em 90%

> **Bug encontrado e corrigido aqui (importante).** A métrica do TAL contém
> `IoU**6`. Para um objeto que nada ainda cobre — objeto pequeno nas primeiras
> épocas — todos os `target_scores` ficam na casa de `1e-8`. Como os termos de
> caixa eram normalizados pela soma global desses scores, eles ficavam em `1e-8`
> também: **o modelo não tinha gradiente de caixa nenhum e o treino travava**.
> Correção: os termos de caixa agora são médias ponderadas normalizadas pelos
> próprios pesos (à prova de escala), com fallback uniforme se os pesos
> degenerarem. Loss inicial passou de 0.0004 (falso zero) para 11.1.

### Fase 7 — Avaliação ✅
- [x] NMS via `torchvision.ops.batched_nms` — por classe, com modo
      class-agnostic e multi-label
- [x] **mAP implementada do zero** (não via pycocotools): curva P/R, precisão
      interpolada à direita, 101 pontos de recall, 10 thresholds de IoU 0.50:0.95
- [x] `MeanAveragePrecision` com regras COCO: uma detecção por objeto, classe
      tem que bater, classe sem ground truth não entra na média
- [x] `evaluate()` força `augment=False` e mapeia predições **e** ground truth
      de volta às coordenadas originais antes de casar
- [x] validação dentro do treino: `best.pt` escolhido por mAP, não por loss
- [x] CLI `gusnet val`
- [x] teste com modelo-oráculo (reporta o ground truth exato) → mAP 1.0,
      o que prova a contabilidade de coordenadas do avaliador

> **Bug encontrado aqui.** Ao ligar a validação no trainer, a primeira execução
> terminou com `best mAP50-95 11.1202` — mAP acima de 1, impossível. Com
> `val_interval=5`, as épocas sem validação caíam num fallback para a loss de
> treino, e maximizar misturava duas escalas: a loss 11.12 da época 1 ganhava de
> qualquer mAP real. Corrigido: época sem validação não concorre a `best.pt`.

> Resultado no dataset sintético (GUSNet-n, 40 épocas, 160px, 24 imagens):
> **mAP50 0.747, mAP50-95 0.279, precision 1.00, recall 0.78**. O vão entre
> mAP50 e mAP50-95 diz exatamente o que se espera: acha os objetos, mas as
> caixas não são justas.

### Fase 8 — Inferência e export
- [ ] `gusnet predict` (imagem, pasta, vídeo)
- [ ] export ONNX (com e sem NMS no grafo) e TorchScript
- [ ] benchmark de latência

## Tamanhos da família (80 classes, 640px)

| Variante | width | depth | Parâmetros |
|---|---|---|---|
| n | 0.25 | 0.34 | 2,5 M |
| s | 0.50 | 0.34 | 9,8 M |
| m | 0.75 | 0.67 | 26,8 M |
| l | 1.00 | 1.00 | 56,0 M |
| x | 1.25 | 1.34 | 100,7 M |

## Hardware disponível

RTX 3060 12 GB. Suficiente para treinar tamanhos n/s em 640px com batch 16-32 e
AMP. Para m/l, usar accumulate de gradiente ou resolução menor.

## Ordem de validação

1. overfit de 1 imagem (loss → ~0)
2. COCO128, 100 épocas → mAP alto (memorização, prova que o pipeline fecha)
3. VOC ou COCO subset
4. COCO 2017 completo
