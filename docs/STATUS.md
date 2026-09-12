# Estado do projeto

> **Este é o arquivo que diz onde paramos.** Leia antes de começar qualquer
> coisa; atualize antes de commitar. Um status desatualizado é pior que nenhum.
>
> Última atualização: **2026-09-12**

---

## Onde estamos

O roteiro de 8 fases está **completo**. O código lê dados, treina, se mede,
roda em entradas reais e exporta para dois runtimes. 275 testes, CI verde em
Ubuntu e Windows × Python 3.10 e 3.12.

O que **não** existe: pesos treinados. Tudo foi validado em dados sintéticos,
em overfit de 1 imagem, e numa corrida de 12 minutos em 512 imagens COCO reais.
A corrida longa nunca foi feita.

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
| **treino de verdade** | ❌ **é o que falta** |

Última medição real (GUSNet-s, 512 imagens de COCO val2017, 100 épocas, 12 min
na RTX 3060): loss 11.1 → 5.14, mAP50-95 0 → 0.0124, mAP50 0.0262.
O pipeline fecha; o número é baixo porque a corrida foi curta.

---

## Pendências, por prioridade

### 🔴 1. `--resume` a partir de `last.pt`

**Por que é o primeiro:** uma corrida completa no COCO leva ~6 dias de GPU
ininterruptos. Sem retomada, uma queda de energia no quarto dia joga tudo fora.

O estado já está salvo — `last.pt` carrega `model`, `ema`, `optimizer`, `epoch`
e `config`. Falta apenas ligar:

- flag `--resume PATH` (ou `--resume` apontando para `save_dir/last.pt`)
- no `Trainer.__init__`: carregar pesos, EMA (incluindo `updates`), estado do
  optimizer, e começar de `epoch + 1`
- o contador `_step` precisa ser restaurado também, senão o warmup recomeça
- cuidado: `_maybe_close_mosaic` tem que reconhecer que o mosaic já foi fechado
  se a retomada cair depois desse ponto
- teste: treinar 4 épocas, retomar de `last.pt`, confirmar que o histórico
  continua e que a LR segue a curva certa em vez de reiniciar

### 🟠 2. Baixar `train2017` e rodar o treino completo

19 GB de imagens (as anotações de treino **já estão** em
`G:/datasets/coco/annotations/instances_train2017.json`, vieram no mesmo zip).

```bash
python scripts/download_coco.py --split train2017 --root G:/datasets/coco
```

Comando do treino em [`TRAINING.md`](TRAINING.md#6-running-the-real-thing).
Sugestão de primeira tentativa: **GUSNet-s, 100 épocas, batch 16** ≈ 1.9 dias,
em vez das 300 épocas ≈ 5.6 dias. Fazer só depois do item 1.

### 🟠 3. Liberar espaço no `C:`

Está com ~6 GB livres (99% de uso). Isso limita os workers do dataloader a 4
(o pagefile é onde os workers passam tensores) e é um risco para qualquer coisa
que o Windows queira escrever. Com mais espaço, dá para subir os workers e
ganhar throughput.

### 🟡 4. Preencher o `MODEL_CARD.md`

Só faz sentido quando houver pesos. Todos os campos de procedência já estão
esboçados: commit que treinou, inicialização, dataset, licença. Sem isso os
pesos não devem ser publicados.

### 🟡 5. Conferir a mAP própria contra o `pycocotools`

A implementação é do zero e está testada por propriedades (detector perfeito
= 1.0, caixa frouxa passa em IoU 50 e falha em 75, duplicata custa precisão).
Falta uma conferência numérica contra a referência, como teste opcional que
pula se o `pycocotools` não estiver instalado. Aumenta bastante a confiança
antes de publicar qualquer número.

### 🟡 6. Assigner estático no aquecimento (tipo ATSS)

Hoje o SimOTA colapsa para ~1 positivo por objeto com modelo não treinado, e o
default é o TAL por causa disso. Um assigner estático nas primeiras épocas
tornaria o SimOTA usável desde o passo 1 — é o que o PP-YOLOE faz.

### 🟢 7. Melhorias que não bloqueiam nada

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

---

## Como atualizar este arquivo

Ao terminar uma sessão de trabalho: mover o que foi feito para "Onde estamos",
ajustar prioridades, e acrescentar qualquer armadilha nova na lista final.
Commitar junto com o código, nunca depois.
