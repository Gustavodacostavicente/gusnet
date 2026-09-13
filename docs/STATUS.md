# Estado do projeto

> **Este é o arquivo que diz onde paramos.** Leia antes de começar qualquer
> coisa; atualize antes de commitar. Um status desatualizado é pior que nenhum.
>
> Última atualização: **2026-09-13** (treino COCO **pausado na época 9/100**)

---

## Onde estamos

O roteiro de 8 fases está **completo**, e o treino agora é **retomável**. O
código lê dados, treina, se mede, roda em entradas reais e exporta para dois
runtimes. 283 testes, CI verde em Ubuntu e Windows × Python 3.10 e 3.12.

**A corrida COCO começou e está pausada na época 9 de 100.** Os checkpoints
estão em `H:/datasets/coco-runs/s-100` e a retomada está pronta — o comando
exato está logo abaixo. Ainda não há pesos publicáveis: a corrida precisa
terminar primeiro.

| Fase | Estado |
|---|---|
| 1 — dados | ✅ |
| 2 — backbone + neck | ✅ |
| 3 — head anchor-free | ✅ |
| 4 — label assignment | ✅ |
| 5 — losses | ✅ |
| 6 — treino | ✅ |
| 7 — avaliação (NMS + mAP) | ✅ |
| 8 — inferência + export | ✅ |
| `--resume` do `last.pt` | ✅ |
| **treino de verdade** | ⏸️ **pausado na época 9/100** |

---

## Pendências, por prioridade

### 🔴 1. Retomar a corrida COCO — **pausada na época 9/100**

Pausada em 2026-09-13 a pedido, com o checkpoint da época 9 íntegro. Perdeu-se
apenas a época 10 em andamento (~24 min). **Não recomeçar do zero.**

**Retomar com exatamente isto** (o `--resume` no fim é o que importa):

```powershell
cd G:/Personal/TesteYolo
.venv/Scripts/gusnet train `
  --coco-images H:/datasets/coco/train2017 `
  --coco-annotations H:/datasets/coco/annotations/instances_train2017.json `
  --val-coco-images H:/datasets/coco/val2017 `
  --val-coco-annotations H:/datasets/coco/annotations/instances_val2017.json `
  --model s --imgsz 640 --batch-size 16 --epochs 100 `
  --optimizer sgd --lr0 0.01 --warmup-epochs 3 --close-mosaic 15 `
  --val-interval 5 --workers 4 --device cuda --log-interval 1000 `
  --save-dir H:/datasets/coco-runs/s-100 --resume
```

Para rodar destacado, de modo que sobreviva ao fechamento do terminal, use
`Start-Process` com `-RedirectStandardOutput` (foi assim que a primeira parte
rodou; ver `train-part1.log` no mesmo diretório).

**Antes de retomar, confirme o CUDA:**
`.venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"`.
Qualquer `uv run` desde então pode ter reinstalado o torch CPU por baixo —
ver [`TRAINING.md`](TRAINING.md#1-setting-up-cuda).

#### Estado no momento da pausa

| Arquivo | Conteúdo |
|---|---|
| `last.pt` | época 9, loss 4.3867, `step` 66528, `ema_updates` 66528, `best` 0.0117 |
| `best.pt` | época 5, mAP50-95 **0.0117**, mAP50 **0.0214** |
| `train-part1.log` | o log completo das 9 épocas |

#### O que já foi medido

| Época | Loss | cls | box | dfl | LR |
|---|---|---|---|---|---|
| 1 | 6.9997 | 0.8081 | 3.1723 | 3.0193 | 0.00333 |
| 2 | 5.2804 | 0.8471 | 2.2241 | 2.2092 | 0.00666 |
| 3 | 4.8477 | 0.8310 | 2.0058 | 2.0109 | 0.00999 |
| 4 | 4.6801 | 0.8169 | 1.9245 | 1.9386 | 0.00998 |
| 5 | 4.5523 | 0.8052 | 1.8630 | 1.8841 | 0.00996 |
| 6 | 4.4905 | 0.8000 | 1.8366 | 1.8539 | 0.00994 |
| 7 | 4.4381 | 0.7945 | 1.8077 | 1.8359 | 0.00991 |
| 8 | 4.4042 | 0.7908 | 1.7932 | 1.8201 | 0.00988 |
| 9 | 4.3867 | 0.7887 | 1.7845 | 1.8135 | 0.00984 |

Validação na época 5 (val2017 completo, 5.000 imagens):
**mAP50-95 0.0117, mAP50 0.0214, P 0.321, R 0.316**.

Leitura honesta: o número é baixo porque são 9 de 100 épocas. O que importa
nele é que **precisão e recall estão equilibrados** (0.321 contra 0.316) — um
detector quebrado costuma ter um dos dois perto de zero. E a validação aninhada
rodou sobre 5.000 imagens sem travar, que é o ponto exato onde a corrida
anterior travava antes do conserto de `workers=0`.

**Ritmo medido:** 24,6 min por época, 79 img/s. As 91 épocas restantes levam
**~38 h ≈ 1,6 dias**.

#### Quando terminar

1. `gusnet val --weights H:/datasets/coco-runs/s-100/best.pt` contra o val2017
   completo, e anotar o mAP aqui.
2. Preencher o `MODEL_CARD.md` (item 3) — sem ele os pesos não devem ser
   publicados, que é justamente o que o projeto existe para evitar.
3. Considerar continuar até 300 épocas (`--epochs 300 --resume`): a curva de
   cosine se estende e o schedule longo é onde os detectores desta família
   chegam ao mAP de referência.

### 🟠 2. Liberar espaço no `C:`

Está com ~6 GB livres (99% de uso). Isso limita os workers do dataloader a 4
(o pagefile é onde os workers passam tensores) e é um risco para qualquer coisa
que o Windows queira escrever. Com mais espaço, dá para subir os workers e
ganhar throughput.

### 🟡 3. Preencher o `MODEL_CARD.md`

Só faz sentido quando houver pesos. Todos os campos de procedência já estão
esboçados: commit que treinou, inicialização, dataset, licença. Sem isso os
pesos não devem ser publicados.

### 🟡 4. Conferir a mAP própria contra o `pycocotools`

A implementação é do zero e está testada por propriedades (detector perfeito
= 1.0, caixa frouxa passa em IoU 50 e falha em 75, duplicata custa precisão).
Falta uma conferência numérica contra a referência, como teste opcional que
pula se o `pycocotools` não estiver instalado. Aumenta bastante a confiança
antes de publicar qualquer número.

### 🟡 5. Assigner estático no aquecimento (tipo ATSS)

Hoje o SimOTA colapsa para ~1 positivo por objeto com modelo não treinado, e o
default é o TAL por causa disso. Um assigner estático nas primeiras épocas
tornaria o SimOTA usável desde o passo 1 — é o que o PP-YOLOE faz.

### 🟢 6. Melhorias que não bloqueiam nada

- **treino multi-escala** — varia a resolução por batch, ganho conhecido em mAP
- **DDP multi-GPU** — não ajuda nesta máquina, mas ajuda quem clonar o projeto
- **batching retangular na inferência** — agrupa imagens de aspecto parecido e
  evita padding desperdiçado; `letterbox(stride=32)` já existe
- **export TensorRT / OpenVINO** em cima do grafo ONNX
- **publicar no PyPI** — o nome `gusnet` precisa ser verificado

---

## Coisas que já sabemos e não vale redescobrir

Estão documentadas por extenso, mas vale a lista curta:

- `uv run` reinstala o torch CPU por baixo. Usar `.venv/Scripts/...` direto.
- `--workers 8` estoura a memória compartilhada do Windows (erro 1455). 4 é o
  teto nesta máquina.
- Validação aninhada no treino com workers > 0 **deadlocka** no Windows — o
  trainer já força `workers=0` ali.
- BatchNorm com `momentum=0.03` leva ~200 passos para as running stats
  alcançarem as de batch. Modelo pode parecer quebrado em `eval()` cedo.
- A métrica do TAL tem `IoU**6`: para objeto que nada cobre, os target scores
  vão a ~1e-8. Por isso os termos de caixa são médias ponderadas normalizadas
  pelos próprios pesos.
- Tracing com NMS embutido **grava a contagem de detecções como constante**.
  Por isso o TorchScript scripta a supressão e o ONNX usa o exportador legado.
- Retomar exige mais que pesos e optimizer: sem o contador de iterações o
  warmup roda de novo sobre um modelo já treinado. `last.pt` guarda `step`,
  `ema_updates`, o scaler do AMP e o melhor score.
- O checkpoint é escrito **ao fim de cada época**. Parar no meio de uma época
  descarta o que ela já andou — no COCO isso são até ~25 min. Se der para
  escolher a hora, pare logo depois de uma linha `epoch N/100` aparecer no log.
- A corrida COCO roda em ~24,6 min/época com 79 img/s. Se estiver muito mais
  lenta que isso, algo mudou — checar `nvidia-smi` (deve ficar em ~85-88%).

---

## Como atualizar este arquivo

Ao terminar uma sessão de trabalho: mover o que foi feito para "Onde estamos",
ajustar prioridades, e acrescentar qualquer armadilha nova na lista final.
Commitar junto com o código, nunca depois.
