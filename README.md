# OCR PT-BR Benchmark

Benchmark de modelos OCR/ VLM em texto em português brasileiro.

## Setup

```bash
python -m pip install -r requirements.txt
```

Defina as variáveis de ambiente com as chaves de API:

```bash
export OPENROUTER_API_KEY="sk-..."
export DEEPINFRA_API_KEY="..."
```

## Geração de Amostras

```bash
# Gera documento_teste.png (impresso) e documento_manuscrito.png em assets/
python src/geracao/gerar_imagem_teste.py
python src/geracao/gerar_manuscrito.py

# Extrai 10 amostras do DharmaOCR-Benchmark para dados/amostras_dharma/
python src/geracao/extrair_amostras.py
```

## Execução de Testes

```bash
python src/testes/testar_ocr.py --help
```

### Exemplos

Testar modelos em uma imagem única:
```bash
python src/testes/testar_ocr.py \
    --modelos "qwen/qwen3-vl-32b-instruct,openrouter" \
    --amostras assets/documento_teste.png \
    --output resultados/impresso.json
```

Testar múltiplos modelos com lista em arquivo:
```bash
python src/testes/testar_ocr.py \
    --modelos-list configs/modelos_openrouter_5.txt \
    --amostras assets/documento_teste.png \
    --output resultados/impresso.json
```

Testar dataset completo (Dharma, 10 amostras):
```bash
python src/testes/testar_ocr.py \
    --dataset dharma \
    --modelos-list configs/modelos_dharma_9.txt \
    --output resultados/dharma.json \
    --resume
```

Testar com provedor específico:
```bash
python src/testes/testar_ocr.py \
    --provider deepinfra \
    --modelos-list configs/modelos_deepinfra_3.txt \
    --amostras assets/documento_teste.png
```

Reproduzir todos os resultados originais:
```bash
make testar-impresso
make testar-manuscrito
make testar-dharma
make testar-deepinfra
make testar-bressay
```

## Cálculo de Métricas

```bash
python src/metricas/calcular_metricas.py --entrada resultados/impresso.json
python src/metricas/calcular_metricas.py --entrada resultados/dharma.json --saida resultados/dharma_metricas.json
```

Para formatos antigos sem ground truth no JSON:
```bash
python src/metricas/calcular_metricas.py \
    --entrada resultados/deepinfra.json \
    --ground-truth assets/ground_truth.txt
```

Todas de uma vez:
```bash
make metricas
```

## Estrutura do Projeto

```
src/
├── geracao/          # Geração de imagens de teste
│   ├── gerar_imagem_teste.py
│   ├── gerar_manuscrito.py
│   └── extrair_amostras.py
├── testes/
│   └── testar_ocr.py      # Script unificado de teste
└── metricas/
    └── calcular_metricas.py # Cálculo de CER, WER, F1, BLEU, Score
dados/
├── amostras_dharma/    # Dataset Dharma (versionado)
└── amostras_manuscritas/ # Amostras locais (versionado)
resultados/             # Resultados de testes em JSON
configs/                # Listas de modelos por provedor
assets/                 # Imagens de teste e ground truths
Makefile                # Comandos de reprodução
```

## Métricas

- **CER** (Character Error Rate): distância Levenshtein / total de chars
- **WER** (Word Error Rate): distância Levenshtein entre tokens / total de tokens
- **F1**: média harmônica entre precisão e recall de palavras (bag-of-words)
- **Levenshtein Ratio**: similaridade normalizada de caracteres
- **BLEU**: n-gram precision em nível de caractere
- **Score**: (Levenshtein Ratio + BLEU) / 2

---

**Autor:** [Alisio](https://github.com/alisio)  
**Dataset:** [DharmaOCR-Benchmark](https://huggingface.co/datasets/Dharma-AI/DharmaOCR-Benchmark) (Cardoso et al., 2026)  
**Licença:** MIT  
**IA utilizada:** Este projeto foi desenvolvido com assistência de IA generativa.
