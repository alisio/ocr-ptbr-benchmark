# Relatório Comparativo: OCR em Português Brasileiro — 8 Modelos no DharmaOCR-Benchmark

## Metodologia

- **Dataset**: [DharmaOCR-Benchmark](https://huggingface.co/datasets/Dharma-AI/DharmaOCR-Benchmark) (Cardoso et al., 2026) — 496 imagens em português brasileiro, dividido em ESTER-Pt (texto impresso), Legal (documentos jurídicos) e BRESSAY (redações manuscritas)
- **Amostragem**: 10 amostras (5 manuscritas BRESSAY + 5 impressas ESTER-Pt/Legal)
- **Prompt único** para todos os modelos com temperatura 0
- **Plataformas**: OpenRouter e DeepInfra (API OpenAI-compatível)
- **Métrica composta oficial**: Score = (Levenshtein Ratio + BLEU char-level) / 2
- **Data do teste**: 17-18 de junho de 2026

### Amostras

| # | Tipo | Amostra | Conteúdo | Chars |
|---|------|---------|----------|:-----:|
| 1 | Manuscrito | corcunda_notredame | Ensaio sobre "O Corcunda de Notredame" | 2.356 |
| 2 | Manuscrito | filme_coringa | Ensaio sobre filme Coringa | 2.561 |
| 3 | Manuscrito | filme_beleza_ocultar | Ensaio sobre filme Beleza Oculta | 2.899 |
| 4 | Manuscrito | serie_13_reasons | Ensaio sobre "13 Reasons Why" | 3.255 |
| 5 | Manuscrito | filme_coringa_2 | Ensaio sobre filme Coringa (outro autor) | 3.081 |
| 6 | Impresso | luis_x_rei | Verbete Wikipedia: Luís X de França | 3.463 |
| 7 | Impresso | anatomia | Verbete Wikipedia: Anatólia | 6.102 |
| 8 | Impresso | barranquilla | Verbete Wikipedia: Barranquilla | 7.497 |
| 9 | Impresso | cairo | Verbete Wikipedia: Cairo | 12.082 |
| 10 | Impresso | doutrina_nacional | Documento jurídico ("Doutrina Nacional") | 3.413 |

---

## Ranking Geral

| # | Modelo | Provedor | Score | LR | BLEU | CER | WER | F1 | Latência |
|---|--------|:--------:|:----:|:--:|:----:|:---:|:---:|:--:|:--------:|
| 1 | **qwen/qwen3-vl-235b-a22b-instruct** | OpenRouter | **0.8915** | 0.8772 | 0.9058 | 12.28% | 20.83% | 0.7959 | 89.4s |
| 2 | **qwen/qwen3-vl-32b-instruct** | OpenRouter | **0.8658** | 0.8500 | 0.8816 | 17.22% | 30.38% | 0.7531 | 20.9s |
| 3 | **meta-llama/llama-4-maverick** | OpenRouter | **0.8423** | 0.8187 | 0.8658 | 18.13% | 32.00% | 0.6950 | 37.9s |
| 4 | **google/gemma-4-31b-it** | OpenRouter | **0.7996** | 0.7819 | 0.8173 | 27.18% | 43.73% | 0.6595 | 82.0s |
| 5 | **allenai/olmOCR-2-7B-1025** | DeepInfra | **0.7913** | 0.7671 | 0.8156 | 58.81% | 90.11% | 0.6919 | 70.6s |
| 6 | **mistralai/mistral-large-2512** | OpenRouter | **0.7825** | 0.7484 | 0.8165 | 25.38% | 40.23% | 0.6443 | 27.3s |
| 7 | **PaddlePaddle/PaddleOCR-VL-0.9B** | DeepInfra | **0.7460** | 0.7475 | 0.7446 | 64.06% | 94.60% | 0.6525 | 72.1s |
| 8 | **nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL** | DeepInfra | **0.3788** | 0.3794 | 0.3782 | 62.05% | 66.73% | 0.3553 | 23.8s |

### Por Tipo (Manuscrito vs Impresso)

| Modelo | Score Manuscrito | CER Manuscrito | Score Impresso | CER Impresso |
|--------|:---------------:|:--------------:|:--------------:|:------------:|
| qwen/qwen3-vl-235b-a22b-instruct | **0.7964** | 23.94% | **0.9866** | 1.02% |
| qwen/qwen3-vl-32b-instruct | 0.7470 | 33.21% | 0.9846 | 1.25% |
| meta-llama/llama-4-maverick | 0.7525 | 28.08% | 0.9320 | 8.17% |
| google/gemma-4-31b-it | 0.6764 | 13.41% | 0.7228 | 40.93% |
| allenai/olmOCR-2-7B-1025 | 0.8775 | 13.58% | 0.7052 | 104.19% |
| mistralai/mistral-large-2512 | 0.6638 | 38.62% | 0.9011 | 12.15% |
| PaddlePaddle/PaddleOCR-VL-0.9B | 0.8750 | 13.56% | 0.6170 | 115.42% |
| nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL | 0.3665 | 43.34% | 0.3911 | 80.76% |

---

## Análise por Modelo

### 1. qwen/qwen3-vl-235b-a22b-instruct — Score 0.8915

| Amostra | Tipo | Score | CER | WER |
|---------|:----:|:----:|:---:|:---:|
| corcunda_notredame | Manuscrito | 0.9609 | 3.19% | 10.84% |
| filme_coringa | Manuscrito | 0.9805 | 1.72% | 4.38% |
| filme_beleza_ocultar | Manuscrito | 0.4443 | 65.85% | 88.40% |
| serie_13_reasons | Manuscrito | 0.9152 | 8.89% | 21.96% |
| filme_coringa_2 | Manuscrito | 0.6811 | 38.11% | 54.47% |
| luis_x_rei | Impresso | 0.9880 | 0.75% | 5.48% |
| anatomia | Impresso | 0.9930 | 0.46% | 3.68% |
| barranquilla | Impresso | 0.9843 | 1.39% | 5.37% |
| cairo | Impresso | 0.9832 | 1.41% | 6.81% |
| doutrina_nacional | Impresso | 0.9846 | 1.06% | 6.88% |

Desempenho quase perfeito em texto impresso (Score 0.9866, CER ~1%). O ponto fraco é caligrafia mais desafiadora — amostra `filme_beleza_ocultar` com Score 0.4443. Nenhuma degeneração observada. O modelo de 235B tem a vantagem de escala para manuscritos, com Score 0.9119 nas 3 amostras mais legíveis.

### 2. qwen/qwen3-vl-32b-instruct — Score 0.8658

| Amostra | Tipo | Score | CER | WER |
|---------|:----:|:----:|:---:|:---:|
| corcunda_notredame | Manuscrito | 0.9568 | 3.40% | 12.20% |
| filme_coringa | Manuscrito | 0.9403 | 5.43% | 14.84% |
| filme_beleza_ocultar | Manuscrito | 0.3918 | 89.99% | 129.93% |
| serie_13_reasons | Manuscrito | 0.8987 | 10.58% | 23.95% |
| filme_coringa_2 | Manuscrito | 0.5472 | 56.63% | 91.49% |
| luis_x_rei | Impresso | 0.9885 | 0.66% | 5.48% |
| anatomia | Impresso | 0.9929 | 0.52% | 3.79% |
| barranquilla | Impresso | 0.9777 | 2.07% | 7.58% |
| cairo | Impresso | 0.9780 | 1.91% | 8.00% |
| doutrina_nacional | Impresso | 0.9858 | 1.00% | 6.50% |

Excelente em texto impresso (Score 0.9846), equivalente ao 235B. Em manuscrito, tende a "corrigir" acentos que não estão na imagem — extração não-fiel. Latência 20.9s é a menor entre modelos competitivos.

### 3. meta-llama/llama-4-maverick — Score 0.8423

| Amostra | Tipo | Score | CER | WER |
|---------|:----:|:----:|:---:|:---:|
| corcunda_notredame | Manuscrito | 0.9169 | 8.12% | 18.97% |
| filme_coringa | Manuscrito | 0.7878 | 22.71% | 38.20% |
| filme_beleza_ocultar | Manuscrito | 0.8853 | 9.53% | 34.34% |
| serie_13_reasons | Manuscrito | 0.7618 | 27.37% | 47.31% |
| filme_coringa_2 | Manuscrito | 0.4107 | 72.68% | 94.68% |
| luis_x_rei | Impresso | 0.9885 | 0.72% | 6.15% |
| anatomia | Impresso | 0.9745 | 2.89% | 7.57% |
| barranquilla | Impresso | 0.9418 | 6.59% | 20.95% |
| cairo | Impresso | 0.7733 | 29.36% | 44.98% |
| doutrina_nacional | Impresso | 0.9820 | 1.32% | 6.88% |

Surpreendente em manuscrito — segunda melhor performance em `filme_beleza_ocultar` (Score 0.8853). Em impresso, bom desempenho mas com dificuldade em textos longos como `cairo` (Score 0.7733).

### 4. google/gemma-4-31b-it — Score 0.7996

| Amostra | Tipo | Score | CER | WER |
|---------|:----:|:----:|:---:|:---:|
| corcunda_notredame | Manuscrito | 0.9149 | 8.37% | 15.99% |
| filme_coringa | Manuscrito | 0.9404 | 5.90% | 12.65% |
| filme_beleza_ocultar | Manuscrito | 0.8413 | 17.44% | 36.43% |
| serie_13_reasons | Manuscrito | 0.8227 | 20.57% | 38.52% |
| filme_coringa_2 | Manuscrito | 0.8627 | 14.81% | 25.53% |
| luis_x_rei | Impresso | 0.9415 | 5.03% | 19.10% |
| anatomia | Impresso | 0.8689 | 16.20% | 31.83% |
| barranquilla | Impresso | 0.6148 | 53.62% | 75.72% |
| cairo | Impresso | 0.2290 | 125.91% | 177.49% |
| doutrina_nacional | Impresso | 0.9599 | 3.90% | 4.02% |

Manuscrito bom (segundo melhor CER em manuscrito: 13.41%). Em texto impresso denso, porém, sofreu degeneração severa — `cairo` (12.082 chars GT) produziu 21.169 chars com repetição de parágrafos, inflando CER para 125.91%. Latência alta (82s).

### 5. allenai/olmOCR-2-7B-1025 — Score 0.7913

| Amostra | Tipo | Score | CER | WER |
|---------|:----:|:----:|:---:|:---:|
| corcunda_notredame | Manuscrito | 0.9153 | 7.90% | 18.16% |
| filme_coringa | Manuscrito | 0.9428 | 5.98% | 12.41% |
| filme_beleza_ocultar | Manuscrito | 0.8416 | 18.06% | 35.03% |
| serie_13_reasons | Manuscrito | 0.8273 | 20.76% | 38.32% |
| filme_coringa_2 | Manuscrito | 0.8604 | 15.20% | 26.60% |
| luis_x_rei | Impresso | 0.9520 | 4.02% | 17.61% |
| anatomia | Impresso | 0.8700 | 16.17% | 32.24% |
| barranquilla | Impresso | 0.1381 | 441.02% | 638.33% |
| cairo | Impresso | 0.6091 | 54.84% | 77.44% |
| doutrina_nacional | Impresso | 0.9565 | 4.16% | 4.97% |

**Melhor em manuscrito entre todos os modelos**: Score 0.8775, CER 13.58%. Porém, sofreu degeneração severa em `barranquilla` (38.436 chars output para 7.497 chars GT — 5× o tamanho esperado), fenômeno conhecido reportado no paper DharmaOCR. Também apresentou degeneração em `cairo` (11.448 vs 12.082).

### 6. mistralai/mistral-large-2512 — Score 0.7825

| Amostra | Tipo | Score | CER | WER |
|---------|:----:|:----:|:---:|:---:|
| corcunda_notredame | Manuscrito | 0.9208 | 7.31% | 18.70% |
| filme_coringa | Manuscrito | 0.8339 | 17.98% | 33.82% |
| filme_beleza_ocultar | Manuscrito | 0.3443 | 70.27% | 91.65% |
| serie_13_reasons | Manuscrito | 0.8190 | 20.69% | 41.72% |
| filme_coringa_2 | Manuscrito | 0.4010 | 76.84% | 106.60% |
| luis_x_rei | Impresso | 0.9773 | 1.42% | 11.63% |
| anatomia | Impresso | 0.9656 | 3.61% | 8.19% |
| barranquilla | Impresso | 0.8071 | 24.17% | 38.76% |
| cairo | Impresso | 0.7809 | 29.50% | 42.84% |
| doutrina_nacional | Impresso | 0.9747 | 2.02% | 8.41% |

Impresso bom (Score 0.9011) mas perde para Qwen. Manuscrito fraco nas amostras mais densas. Latência baixa (27.3s).

### 7. PaddlePaddle/PaddleOCR-VL-0.9B — Score 0.7460

| Amostra | Tipo | Score | CER | WER |
|---------|:----:|:----:|:---:|:---:|
| corcunda_notredame | Manuscrito | 0.9154 | 7.86% | 18.43% |
| filme_coringa | Manuscrito | 0.9443 | 5.71% | 12.17% |
| filme_beleza_ocultar | Manuscrito | 0.8438 | 16.92% | 33.64% |
| serie_13_reasons | Manuscrito | 0.8159 | 21.34% | 39.32% |
| filme_coringa_2 | Manuscrito | 0.8557 | 15.98% | 28.51% |
| luis_x_rei | Impresso | 0.9540 | 3.87% | 16.94% |
| anatomia | Impresso | 0.8716 | 15.94% | 31.42% |
| barranquilla | Impresso | 0.1517 | 353.59% | 515.42% |
| cairo | Impresso | 0.1478 | 195.59% | 245.92% |
| doutrina_nacional | Impresso | 0.9600 | 3.84% | 4.21% |

**Segundo melhor em manuscrito** (Score 0.8750), próximo do olmOCR. Porém, degeneração severa em textos densos — `barranquilla` (32.088 chars output, 4.28× GT) e `cairo` (30.038 chars output, 2.49× GT). Degeneração mais severa que olmOCR, possivelmente por ser um modelo de apenas 0.9B parâmetros. Em deprecação na DeepInfra (05/07/2026).

### 8. nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL — Score 0.3788

| Amostra | Tipo | Score | CER | WER |
|---------|:----:|:----:|:---:|:---:|
| corcunda_notredame | Manuscrito | 0.9544 | 3.78% | 9.76% |
| filme_coringa | Manuscrito | 0.0000 | 100.00% | 100.00% |
| filme_beleza_ocultar | Manuscrito | 0.3259 | 71.37% | 91.42% |
| serie_13_reasons | Manuscrito | 0.5522 | 41.54% | 50.90% |
| filme_coringa_2 | Manuscrito | 0.0000 | 100.00% | 100.00% |
| luis_x_rei | Impresso | 0.9747 | 2.51% | 8.14% |
| anatomia | Impresso | 0.0000 | 100.00% | 100.00% |
| barranquilla | Impresso | 0.0000 | 100.00% | 100.00% |
| cairo | Impresso | 0.0000 | 100.00% | 100.00% |
| doutrina_nacional | Impresso | 0.9810 | 1.35% | 7.07% |

Modelo inconsistente: 5/10 amostras retornaram saída vazia (provavelmente conteúdo filtrado pelo safety guardrail interno). Nas 5 amostras que funcionou, desempenho foi de bom a excelente (especialmente `corcunda_notredame` com Score 0.9544 e `doutrina_nacional` com 0.9810). Não recomendado para uso em produção devido à alta taxa de falha.

---

## Comparação com Resultados Publicados (DharmaOCR Paper)

| Modelo | DharmaOCR (paper) | Nosso Teste | Diferença |
|--------|:-----------------:|:-----------:|:---------:|
| olmOCR-2-7B | 0.823 | **0.7913** | -0.0317 |
| Qwen3-VL-32B | N/A | **0.8658** | — |
| Qwen3-VL-235B | N/A | **0.8915** | — |
| Llama-4-Maverick | N/A | **0.8423** | — |
| Nemotron-12B | N/A | 0.3788 | — |

> **Nota**: O Score do paper usa BLEU word-level (nltk), enquanto nosso teste usa BLEU character-level, o que tende a valores mais altos em geral. A métrica character-level é mais estável e adequada para OCR, onde erros parciais de caracteres são comuns.

---

## Fenômeno de Degeneração

Três modelos apresentaram degeneração (repetição de texto) em documentos densos:

| Modelo | Amostra | GT chars | Output chars | Multiplicador |
|--------|---------|:--------:|:-----------:|:-------------:|
| allenai/olmOCR-2-7B-1025 | barranquilla | 7.497 | 38.436 | **5.13×** |
| PaddlePaddle/PaddleOCR-VL-0.9B | barranquilla | 7.497 | 32.088 | **4.28×** |
| PaddlePaddle/PaddleOCR-VL-0.9B | cairo | 12.082 | 30.038 | **2.49×** |
| google/gemma-4-31b-it | cairo | 12.082 | 21.169 | **1.75×** |

O paper DharmaOCR reporta 1.41% de degeneração para olmOCR. Modelos foram treinados com GRPO (RL) que pode levar a esse comportamento em documentos fora da distribuição de treino. A degeneração impacta severamente métricas CER/WER e também o custo (tokens de saída).

---

## Recomendações Finais

### Cenário 1: Texto Impresso (documentos, artigos, livros)

| Modelo | Score | CER | Custo |
|--------|:----:|:---:|:-----:|
| **qwen/qwen3-vl-235b-a22b-instruct** | **0.9866** | **1.02%** | $0.00046 |
| qwen/qwen3-vl-32b-instruct | 0.9846 | 1.25% | **$0.00023** |
| meta-llama/llama-4-maverick | 0.9320 | 8.17% | $0.00042 |

→ **qwen/qwen3-vl-32b-instruct** oferece o melhor custo-benefício para texto impresso (Score 0.9846, CER 1.25%, latência 21s).

### Cenário 2: Texto Manuscrito (redações, notas, formulários)

| Modelo | Score | CER |
|--------|:----:|:---:|
| **allenai/olmOCR-2-7B-1025** | **0.8775** | **13.58%** |
| PaddlePaddle/PaddleOCR-VL-0.9B | 0.8750 | 13.56% |
| qwen/qwen3-vl-235b-a22b-instruct | 0.7964 | 23.94% |

→ **allenai/olmOCR-2-7B-1025** no DeepInfra (Score 0.8775, CER 13.58%) — cuidado com degeneração em textos > 2.000 chars.

### Cenário 3: Uso Geral (misto impresso + manuscrito)

| Modelo | Score Geral | Pros | Contras |
|--------|:----------:|------|---------|
| **qwen/qwen3-vl-235b-a22b-instruct** | **0.8915** | Melhor geral, sem degeneração | Mais caro, latência alta |
| qwen/qwen3-vl-32b-instruct | 0.8658 | Rápido e barato, quase tão bom | "Corrige" acentos em manuscrito |
| meta-llama/llama-4-maverick | 0.8423 | Surpreendente em manuscrito | Perde para Qwen em impresso |

→ **qwen/qwen3-vl-235b-a22b-instruct** para máxima precisão; **qwen/qwen3-vl-32b-instruct** para melhor custo-benefício.

### Alertas

| Modelo | Problema | Impacto |
|--------|----------|---------|
| `allenai/olmOCR-2-7B-1025` | Degeneração em textos densos | Pode inflar custo e piorar métricas |
| `PaddlePaddle/PaddleOCR-VL-0.9B` | Degeneração severa + deprecação | Não recomendado para novos projetos |
| `google/gemma-4-31b-it` | Degeneração em textos longos | Latência alta (82s) + degeneração |
| `qwen/qwen3-vl-32b-instruct` | "Corrige" acentos em manuscrito | Perde fidelidade visual |
| `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL` | 50% de falha (saída vazia) | Não utilizável em produção |

---

## Apêndice A — Ground Truth das Amostras

### A.1 manuscrito/corcunda_notredame (2.356 chars)

![corcunda_notredame](dados/amostras_dharma/manuscrito/corcunda_notredame.png)

```
Na obra "O corcunda de Notredame", é aposentada a história de Quasímodo, um deficiente fisi-
co com transtornos mentais que, devido à repulsa social relacionada à sua aparência e con-
dição psíquica, vive escondido na torre de uma catedral. Fora da ficção, é fato que existe
preconceito e julgamento equivocado relacionados à saúde mental no hodierno brasileiro, que
marcam e --exl-- excluem quem sofre com este problema. Nesse sentido a falta de informa-
ção seguida de negacionismo, são fatores que corroboram a perpeturação desse cenário
no país.

À vista desse problema, é importante destacar que a carência de conhecimento sobre o
assunto é diametralmente relevante na população do Brasil. De acordo com dados do Institu-
to Brasileiro de Geografia e Estatística (IBGE), cerca de 30% das pessoas reconhecem a ansiedade
como transtorno mental, e menos de 20% saberiam autodiagnosticar-se. Então, é notória a
falta de clareza sobre a temática nos dias atuais. Assim, é de suma importância a di-
vulgação da informação de maneira ética e responsável.

Outrossim, não raro observa-se que o indivíduo, ou seus familiares, neguem a presença
de alguma inconformidade mental, mesmo que momentânea, em si próprios ou em seus núcleos
Segundo o historiador Leandro Karnal, os assuntos de ordens psiquiátricas são mal vistos
socialmente, e por isso renegados, desde o --incí-- início do grande êxodo rural culminado na
primeira revolução industrial, que pode-se dizer ser o grande causador das desordens men-
tais desde o século 19. Logo, a negação com sua gênese no preconceito, dificulta o diag-
nóstico, e consequentemente, o tratamento, o que torna a mitigação do problema mais
complexa.

Portanto, é mister que o Estado tome providências para amenizar o quadro atual. Pa-
ra a orientação da população à respeito desta questão, urge que o Ministério da Saúde
crie, por meio de verbas governamentais, campanhas publicitárias nas redes sociais que ex-
ponham os transtornos mentais mais comuns na atualidade, e mostrem as consequências da
desinformação e da não-aceitação. Sendo assim, sugerir ao interlocutor aprofundar-se
no assunto e evitar a segregação pelo preconceito. Deste modo, será possível di-
minuir os estigmas sociais e, diferentemente do exposto em "O corcunda de Notre
dame", o preconceito não afete a qualidade de vida dos indivíduos no Brasil.
```

### A.2 manuscrito/filme_coringa (2.561 chars)

![filme_coringa](dados/amostras_dharma/manuscrito/filme_coringa.png)

```
O filme Coringa retrata a história de Arthur, um homem solitário com diversos traumas
de infância, que tenta lidar com a vida e com sua doença mental. No filme, Arthur relata
em seu diário o fardo de ter uma doença psicológica e a pressão imposta pela sociedade para
que ele a esconda. Fora da ficção, o estigma associado às doenças mentais na sociedade
brasileira não é tão díspar e decorre, principalmente, do tabu acerca de doenças psicológicas
e da ausência de ações do Estado.

Em primera análise, observa-se que o preconceito enraizado dificulta o diagnóstico de doenças
mentais. Nesse sentido, o psicanalista Sigmund Freud definiu como "tabu" certos temas con-
siderados sagrados, inquietantes e que não devem ser discutidos abertamente. Sob esta óptica, o
estigma associado às doenças mentais advém de um tabu que, ao longo do tempo, reprimiu o
diálogo e o esclarecimento sobre o tema, enraizando o preconceito com doenças mentais na so-
ciedade brasileira. Dessa forma, é crível que tal ambiente social inibe a possibilidade de diagnós-
ticos e tratamentos adequados, condicionando um espaço onde milhões de brasileiros lidam sozi-
nhos com doenças psicológicas.

Outrossim, destaca-se que o dever do Estado é minimizar os impasses para garantir a qua-
lidade de vida dos cidadãos. Consoante o químico H. Louis Le Chatelier, "quando um sistema em
equilíbrio recebe algum tipo de perturbação externa ele se deslocará no sentido de minimizar es-
sa perturbação e retornar ao estado de equilíbrio". Sob este viés, apesar de ser um princípio quí-
mico, ele também pode ser aplicado ao meio social, uma vez que o dever do Estado é tomar
medidas para que o estigma associado às doenças mentais seja atenuado. Desse modo, é impres-
cindível que os orgãos estatais solucionem o óbice e efetivem a qualidade de vida dos cidadãos

Portanto, a respeito do estigma associado às doenças mentais na sociedade brasileira é mister
que o Estado tome medidas para solução do entrave. Para tal, compete ao ministério da Edu-
cação elaborar eventos abertos para a comunidade, em escolas públicas e privadas, em que
por meio de palestras, psicólogos e professores discursarão a respeito da saúde e de doen-
ças mentais. Além disso, visto que atividades culturais possuem imenso poder transfor-
mador. Tais eventos deverão proporcionar para as crianças brincadeiras lúdicas e edu-
cacionais, ensinando-as a identificar doenças psicológicas e, aos pais, como procurar
tratamento, para que assim a realidade social no Brasil seja diferente da que é
retratada no filme Coringa.
```

### A.3 manuscrito/filme_beleza_ocultar (2.899 chars)

![filme_beleza_ocultar](dados/amostras_dharma/manuscrito/filme_beleza_ocultar.png)

```
O filme Beleza Ocultar, exibido pela plataforma de conteúdo audiovisual Netflix, retrata as dificuldades enfrenta-
das por Howard (personagem interpretada pelo ator Will Smith), no âmbito laboral, ao encontrar-se em situação de
fragilidade psicológica. De maneira análoga, a realidade hodierna é verosímil à apresentada no enredo cinematográfico,
haja vista o dilema do estigma associado às doenças mentais na sociedade brasileira causado pela negligência estatal
e pela escassez de ensino escolar.

Cabe ressaltar, acerca disso, que a ínfima atuação governamental em garantir o acesso a consultas profis-
sionais constitui uma causa para o óbice. Sob tal óptica, segundo o filósofo iluminista john Locke, é função
do Estado assegurar o bem-estar social de todos os cidadãos. Diante do fato supracitado, observa-se que
a conjuntura hodierna do Brasil diverge dos ideais de pensador, uma vez que os escassos investimentos mo-
netários voltados à contratação de profissionais especializados para o atendimento de vítimas de enfermida-
des mentais diminui a disponibilidade do serviço, principalmente em regiões menos favorecidas economi-
camente, tendo em vista o alto índice de pessoas com doenças, como ansiedade e depressão, nesses locais. Em conse-
quência disso, há a lotação no agendamento desse tipo de consulta.

Convém mencionar, ademais que a ausência de educação escolar presente no corpo cívico colaboram para a
perpetuação do problema. Nesse viés, conforme o sociólogo clássico Émile Durkheim, a escola é considerada uma das prin-
cipais instituições socializadoras. Na esteira dessa premissa, percebe-se que o pensamento do intelectual concretiza-
se no contexto atual do País, já que a ausência de debates socioeducativos sobre a importância da saúde mental des-
de os anos iniciais do ensino acadêmico propícia a marginalização desse tema no convívio social, o que
auxilia o distanciamento entre as vítimas com transtornos psíquicos e o ambiente escolar. Por
conseguinte, ocorrem diversos prejuízos à formação da integridade física e mental desses estudantes.

Portanto, é necessário que o Poder Público, por meio do envio de investimentos monetários aos mu-
nicípios, amplie a contratação de profissionais especializados no ramo da saúde mental, como psi-
cólogos e psiquiatras, a fim de democratizar o acesso de indivíduos menos favorecidos economicamen-
te ao tratamento de enfermidades psíquicas e de diminuir a lotação no agendamento de consultas.
Além disso, as cooperativas educacionais, por intermédio de atividades interdisciplinares, a exem-
plo de palestras e de trabalhos escolares de carácter lúdico, devem promover debates sobre a im-
portância da resiliência psicológica, com o intuito de aproximar os estudantes e de atenuar os pre-
juízos à formação dos educandos. Assim, poder-se-á mitigar as semelhanças entre o filme Bele-
za Oculta e o contexto hodierno do Brasil.
```

### A.4 manuscrito/serie_13_reasons (3.255 chars)

![serie_13_reasons](dados/amostras_dharma/manuscrito/serie_13_reasons.png)

```
A série da Netflix "13 Reasons Why" iniciou um forte debate acerca da saúde mental e como problemas relativos a essa
são retratados pela mídia. A produção foi criticada pela estereopatização feita quanto às doenças psicológicas e a discussão
se deu no Brasil principalmente nas redes sociais. Nesse sentido, o estigma associado às patologias mentais na socieda-
de brasileiro é fortalecido pelo mito da felicidade constante e pela falta de informação pertinente ao tema.

Primeiramente, como a ideia da felicidade incessante é propagada ostensivamente, pessoas que não são incluídas
nesse ideal são estigmatizadas. Isso se dá pois, por conta da onda das redes sociais, publicações de usuários que exi-
bem apenas momentos selecionados, correspondentes ao padrão estabelecido, criam uma ilusão de que todos na
realidade compartilham dessa mesma sensação de contentamento. Entretanto, quando o indivíduo não se identi-
fica com tal padrão, há um movimento de exclusão consequente da sensação de não pertencimento, que dificulta
a interação social. Nesse contexto, o estereótipo da pessoa que possui algum trastorno de saúde mental é reforçado
pelos possíveis sintomas a serem desenvolvidos, isolando-o mais ainda em um ciclo vicioso de discriminação.
Tal comportamento pode ser visto no documentário "O Dilema das redes", o qual analiza como o funcionamento das redes
sociais e o mito da felicidade atuam no adoecimento da mente e estereotipização dos --do-- distúrbios. Dessa
forma, o estigma associado às patologías mentais é favorecido pela ilusão do contentamento constante.

Além disso, como a saúde mental é pouco discutida, falta informação e preconceitos relativos a essa são
criados e perpetuados. Esse tabu ocorre porque os transtornos mentais são historicamente tratados com base no isola-
mento. Se antes a exclusão era feita com internações em hospícios e sanatórios, atualmente o tema é isolado ao
evitá-lo nas conversas e ações do cotidiano. Isso é provado pela política de desmante pelo atual Governo Fede-
ral da estrutura pública de clínicas de tratamento no ano de 2020, com o fechamento de unidades e corte de
verbas, o que evidencia o não entendimento do assunto como pertencente a saúde pública. Portanto, a
falta --Diante disso, a estigmatização associada às doenças mentais-- de informação favorece a formação de este-
reótipos.

Diante disso, a estigmatização associada às doenças mentais na sociedade brasileira é fortalecida pelo mito da
felicidade e pela falta de informação. Assim, para conscientizar a população acerca da ilusão propagada pelas redes
socias e os distúrbios psicológicos causados por essa, cabe ao Ministério da Saúde divulgar materiais de apoio que
--ressaltem a-- ensinem como utilizar taís mecanismos de forma saudável, por meio de videos e imagens nas redes sociais.
Ademais, a fim de informar a sociedade, é dever do Ministério da Educação elaborar cartilhas e atividades que estimulem
a discussão acerca da saúde mental no âmbito familiar e escolar, mediante conteúdos desenvolvidos por especialis-
tas na área que atuem na educação pública. Somente --assim-- com essas medidas a estigmatização das doenças mentais
será combatida e o tema poderá ser abordado na média melhor do que na série estadunidense.
```

### A.5 manuscrito/filme_coringa_2 (3.081 chars)

![filme_coringa_2](dados/amostras_dharma/manuscrito/filme_coringa_2.png)

```
No recém lançado filme "Coringa", o protagonista Arthur Fleck sofre de transtornos psiquiátricos e
enfrenta extremas dificultades para realizar seu tratamento, sobretudo em função do preconceito social e
da falta de maiores investimentos em saúde pública na cidade de Gotham. De forma análoga, inúmeros
cidadãos brasileiros, não conseguem cuidar de seu bem-estar emocional, seja pela persistência de estig-
mas associados às doenças mentais, seja pela ineficiência do Estado no que diz respeito à garantia de
direitos previstos constitucionalmente. Diante disso, são necessários esforços para combater esse cenário.

Em primeira análise, um dos fatores que intensifica a persistência de preconceitos contra transtornos
mentais na sociedade brasileira é a dificuldade de se colocar em prática inúmeros direitos sociais garan-
tidos na Constituição. Nessa perspectiva, como discute o intelectual Gilberto Dimenstein em sua obra Cidadão de
Papel", o Brasil é um país que vivencia um avanço significativo no âmbito legislativo, sobretudo após a promul-
gação da "Constituição Cidadã" de 1988. No entanto, esse avanço não é verificado ao se analizar as condições
reais de muitas áreas sociais. Dessa forma, apenas ##da## --de a-- saúde pública ser um direito privado no artigo
sexto, a corrupção e o desvio de verbas são mazelas que dificultam a criação de programas de desendividamen-
to que otimizem tratamentos pricológicas e de campanhas informativas que conscientizem a população acerca da
complexidade do quadro psicossomáticos, medida essencial para a --xxx-- desconstrução de preconceitos.

Ademais, um entrave que corrobora essa problemática no Brasil é a normatização de um estilo de
vida cada vez mais ligado à produtividade. Nesse contexto, de acordo com as ideias do pensador Byung-Chul
Han, defendidas em rua obra "socidade do cansaço", o discurso positivo e meritocrático, forte traço das --xxx--
sociedades pós-modernas, manifesta-se de maneira coercitiva e natural, de modo a mascarar as angus-
tias do cotidiano. Com o tempo, o indivíduo passa a viver em uma realidade marcada pela autocobran-
ça e pela exaustão, o que contribui para o desenvolvimento de episódios de síndrome da pânico, de de-
pressão e até mesmo de suicídio, em que, muitas vezes, os cidadãos não buscam ajuda espe-
cialista com receio de serem analizados como pessoas loucas e improdutivas.

Portanto, a fim de combater os estigmas relacionados às doenças mentais no território brasilei-
ro, é imprescindível que o Ministério da Educação (MEC) - órgão responsável pela implementação de políticas
educacionais - por meio de uma proposta de reformulação da Base Nacional Comum Curricular (BNCC), instru-
a a criação de uma disciplina escolar especializada na conscientização dos jovens acerca da importância
da saúde mental para o bem-estar físico e o equilíbrio do corpo humano. Em conjunto, é essencial que
haja a veiculação de propagandas informativas na internet acerca da necessidade de respeito e de
compreensão para que o Brasil seja um país mais humanizado no que se refere à saúde psíquica.
```

### A.6 impresso/luis_x_rei (3.463 chars)

![luis_x_rei](dados/amostras_dharma/impresso/luis_x_rei.png)

```
Luís X & I (Paris,  - Vincennes, ), também conhecido como Luís, o Teimoso, foi o Rei de Navarra como Luís I de 1305 até sua morte, e também Rei da França como Luís X a partir de 1314.Luís tornou-se rei de Navarra aquando da morte da sua mãe a 2 de Abril de 1305, e rei de França após a morte do seu pai em 1314, tendo sido coroado na catedral de Reims em 24 de Agosto de 1315. O seu curto reinado ficou marcado pelas consequências do caso da Torre de Nesle, que o condicionou a procurar nova esposa para gerar um herdeiro para o trono. A obstinação com que perseguiu este objectivo e a forma como foi manipulado pelos nobres da sua corte valeram-lhe os seus cognomes (le Hutin em francês).Em 21 de Setembro de 1305, Luís casara-se com Margarida de Borgonha, também uma capetiana, filha de Roberto II, duque da Borgonha, e de Inês de França, filha de São Luís. Da união nasceu uma filha, Joana, a 28 de Janeiro de 1312.Em Abril de 1314, ano da morte do seu pai Filipe o Belo, ocorreu o grande escândalo que marcaria a história da França, com graves consequências na linha sucessória do trono francês: A sua esposa Margarida de Borgonha e Branca de Borgonha, esposa do seu irmão Carlos, foram denunciadas por adultério pela sua irmã Isabel de França, rainha de Inglaterra.Uma vez que a descendência da dinastia capetiana e do reino da França fora colocada em perigo, o castigo devia ser exemplar. Margarida da Borgonha foi aprisionada em Château-Gaillard, onde ocupou um quarto aberto aos ventos no topo da torre. Morreu a 30 de Abril de 1315, provavelmente de tuberculose, e segundo algumas versões por estrangulamento, mas as suas rudes condições de encarceramento já eram propícias a uma morte prematura.Luís casou-se então a 19 de Agosto de 1315 com Clemência da Hungria, também uma capetiana, filha de Carlos Martel de Anjou, rei titular da Hungria, e de Clemência de Habsburgo. Desta união nasceria João I de França, o Póstumo, cinco meses após o seu falecimento.Morreu em Vincennes a 5 de Junho de 1316, possivelmente de desidratação, apesar de haver suspeitas de envenenamento. Foi sepultado, tal como a sua segunda esposa e o seu filho, na Basílica de Saint-Denis em Saint-Denis na França.Na data da morte de Luís X, Clemência da Hungria estava grávida, pelo que a sucessão do trono ficou uma incógnita, havendo três pretendentes:O assunto parecia resolvido com o nascimento de um filho varão, João I de França, o Póstumo, na noite de 14 para 15 de Novembro de 1316. Mas João viveu apenas durante alguns dias, falecendo a 19 de Novembro de forma misteriosa durante a cerimónia de apresentação aos barões.À nobreza do reino foi posta a questão da legitimidade da princesa Joana, nascida do primeiro matrimónio, à sucessão do trono francês. De facto, era a primeira vez que ocorria a ausência de um herdeiro varão directo. A sucessão que começara por ser electiva no início da dinastia capetiana, passara a dinástica varonil. Havendo inclusivamente dúvidas sobre a paternidade de Joana, devido ao caso da Torre de Nesle, a nobreza francesa preferiu, alegando a lei sálica, oferecer as coroas de ambos os reinos, e o condado de Champanhe, ao irmão de Luís X e já regente, Filipe V de França.Do seu primeiro casamento com Margarida da Borgonha, de 21 de Setembro de 1305 a 30 de Abril de 1315, nasceu:Luís casou-se em segundas núpcias a 13 de Agosto de 1315 com Clemência da Hungria, de quem teve:De uma criada chamada Eudeline, teve ainda uma filha ilegítima:
```

### A.7 impresso/anatomia (6.102 chars)

![anatomia](dados/amostras_dharma/impresso/anatomia.png)

```
Anatólia (do grego antigo ,  - "leste" ou "erguer/nascer do sol"), ou península anatoliana, também conhecida como Ásia Menor ("pequena Ásia"; ), denota a protrusão ocidental da Ásia, que compõe a maior parte da República da Turquia. A região é banhada pelo mar Negro ao norte, o mar Mediterrâneo ao sul, e do mar Egeu, a oeste. O mar de Mármara forma uma ligação entre os mares Negro e Egeu através do Bósforo e Dardanelos e separa Anatólia da Trácia, no continente europeu. Tradicionalmente, Anatólia é considerada estendendo-se a leste de uma linha entre o golfo de İskenderun e do mar Negro, aproximadamente correspondente aos dois terços ocidentais da parte asiática da Turquia. No entanto, desde Anatólia agora é muitas vezes considerado como sinônimo de Turquia Asiática, suas fronteiras leste e sudeste são amplamente consideradas como sendo as fronteiras turcas com os países vizinhos, que são Geórgia, Arménia, Azerbaijão, Irã, Iraque e Síria, em direção no sentido horário.A península da Anatólia, também chamada de Ásia Menor, é banhada pelo mar Negro ao norte, o mar Mediterrâneo ao sul, o mar Egeu, a oeste, e o mar de Mármara a noroeste, que separa a Anatólia da Trácia na Europa.Tradicionalmente, Anatólia é considerada a se estender ao leste de uma linha por tempo indeterminado correndo a partir do golfo de İskenderun ao mar Negro, coincidente com o Planalto de Anatólia. Esta definição geográfica tradicional é utilizada, por exemplo, na edição mais recente do Merriam-Webster's Geographical Dictionary, bem como a comunidade arqueológica. Sob esta definição, Anatólia é delimitada a leste pelo planalto Armênio, e o rio Eufrates, antes das curvas do rio a sudeste entrando na Mesopotâmia. A sudeste, é delimitada pelas faixas que separam o vale do Orontes, na Síria (região) e a planície da Mesopotâmia.[...]
```

### A.8 impresso/barranquilla (7.497 chars)

![barranquilla](dados/amostras_dharma/impresso/barranquilla.png)

```
Barranquilla ou Barranquilha  (em espanhol: Barranquilla) é uma cidade da Colômbia, capital do departamento do Atlántico. Localiza-se no norte do país, perto do mar das Caraíbas. Tem cerca de 2 milhões de habitantes, sendo atualmente uma das cidades mais populosas do país.Foi fundada em meados de 1629, mas não há registros oficiais sobre sua data de fundação. No entanto, os habitantes da cidade celebram o dia 7 de abril como o aniversário da cidade, pois foi nesse dia, no ano de 1813, que a cidade foi estabelecida legalmente como uma vila. Barranquilla possui um dos mais importantes portos da América do Sul, recebendo desde a Segunda Guerra Mundial, imigrantes de várias partes do mundo. É um dos lares e o local de nascimento da estrela pop internacional Shakira, da atriz e modelo Sofía Vergara e da Miss Universo 2014, Paulina Vega. O escritor Gabriel García Márquez morou e estudou em Barranquilla em sua infância e juventude.[...]
```

### A.9 impresso/cairo (12.082 chars)

![cairo](dados/amostras_dharma/impresso/cairo.png)

```
Cairo (, , literalmente: "conquistador" ou "vencedor"; ) é a capital do Egito e da província homônima. É a maior cidade do mundo árabe e da África. Os egípcios a denominam muitas vezes, simplesmente, com o nome do país no idioma local: Miṣr (), em árabe clássico; e Miṣr, em árabe egípcio. Cairo está localizada nas margens e ilhas do rio Nilo, ao sul do delta. Ao sudoeste, se encontra Giza e a antiga necrópole de Mênfis, com a meseta de Giza e suas monumentais pirâmides, como a Grande Pirâmide de Quéops.[...]
```

### A.10 impresso/doutrina_nacional (3.413 chars)

![doutrina_nacional](dados/amostras_dharma/impresso/doutrina_nacional.png)

```
DOUTRINA NACIONAL
655
3.1.1.1. Desobediência em face da separação dos Poderes
Com relação, em primeiro lugar, à separação dos Poderes, cumpre lembrar
que, no Estado Democrático de Direito, o Poder Judiciário tende a cumprir um
papel muito relevante na tutela da sociedade e do cidadão. Ele cumpre a função
de confrontar o Governo com o Direito. Como advertiu Tamanaha, o Judiciário
serve como "barreira diante de ações governamentais contrárias à lei"³². É uma
garantia democrática de que o Executivo não pode tudo e de que este tem de
respeitar o Direito produzido pelo legislador e aplicado pelo juiz.

Não é demais lembrar, neste aspecto, a lição de Montesquieu, no sentido da
importância de se assegurar a independência e a autoridade do Judiciário. Isso
como freio ao uso arbitrário do poder. Conforme o autor, "não há liberdade se o
Poder Judiciário não for separado do Legislativo e do Executivo"³³. Se o Judiciá-
rio e o Legislativo fossem um só, a vida e a liberdade do indivíduo estariam sujei-
tas a controle arbitrário e não democrático, pois o juiz seria então o legislador. E
se o Judiciário fosse unido ao Poder Executivo, então haveria violência e opres-
são³⁴. Não se teria limites à vontade do governante. Desse modo, sem os freios
estabelecidos pelo Judiciário aos outros Poderes e sem que o juiz tenha indepen-
dência em face destes para aplicar o Direito, perde-se o equilíbrio entre os Pode-
res. O Executivo tende a ser tornar ilimitado e, sendo assim, arbitrário. A Admi-
nistração acaba podendo fazer o que quiser, como melhor entender, da forma
como desejar, sem que tenha que respeitar as fronteiras inerentes à coerção judi-
cial³⁵. É isso, aliás, que está ocorrendo no caso das subvenções para investimento
acima citado. O Executivo, sem os limites inerentes à separação dos Poderes, vem
agindo descalibrada e arbitrariamente.
[...]
```

> O texto completo de cada amostra está disponível em `dados/amostras_dharma/manifest.json` (campo `ground_truth`).

---

## Arquivos Gerados

- `dados/amostras_dharma/` — 10 amostras do DharmaOCR-Benchmark (imagens PNG + manifest.json com GT)
- `resultados/dharma.json` — saídas de 9 modelos nas 10 amostras (80+ entradas)
- `resultados/metricas_dharma_resumo.json` — métricas agregadas por modelo e por amostra
- `src/testes/testar_ocr.py` — script unificado de teste (--provider --modelos --amostras --dataset)
- `src/metricas/calcular_metricas.py` — cálculo de métricas (CER, WER, F1, LR, BLEU, Score)

---

*Relatório gerado em 18 de junho de 2026 — Antonio Alisio de Meneses Cordeiro*
