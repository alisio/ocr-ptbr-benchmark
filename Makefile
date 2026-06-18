# Makefile para reprodução de testes OCR

# ============================================================
# GERAÇÃO DE AMOSTRAS
# ============================================================

.PHONY: gerar-assets gerar-dharma

gerar-assets:
	python src/geracao/gerar_imagem_teste.py --output assets
	python src/geracao/gerar_manuscrito.py --output assets

gerar-dharma:
	python src/geracao/extrair_amostras.py --output dados/amostras_dharma

# ============================================================
# TESTES ORIGINAIS (reprodução dos resultados existentes)
# ============================================================

# resultados/impresso.json: 5 modelos OpenRouter no documento_teste.png
.PHONY: testar-impresso
testar-impresso:
	python src/testes/testar_ocr.py \
		--amostras assets/documento_teste.png \
		--modelos-list configs/modelos_openrouter_5.txt \
		--output resultados/impresso.json

# resultados/manuscrito.json: 3 modelos (mixed providers) no documento_manuscrito.png
.PHONY: testar-manuscrito
testar-manuscrito:
	python src/testes/testar_ocr.py \
		--amostras assets/documento_manuscrito.png \
		--modelos-list configs/modelos_mixed_3.txt \
		--output resultados/manuscrito.json

# resultados/dharma.json: 9 modelos (OpenRouter + DeepInfra) nas 10 amostras
.PHONY: testar-dharma
testar-dharma:
	python src/testes/testar_ocr.py \
		--dataset dharma \
		--modelos-list configs/modelos_dharma_9.txt \
		--output resultados/dharma.json \
		--resume

# resultados/deepinfra.json: 3 modelos DeepInfra no documento_teste.png
.PHONY: testar-deepinfra
testar-deepinfra:
	python src/testes/testar_ocr.py \
		--amostras assets/documento_teste.png \
		--modelos-list configs/modelos_deepinfra_3.txt \
		--output resultados/deepinfra.json

# resultados/bressay.json: 3 modelos em 3 páginas do Bressay
.PHONY: testar-bressay
testar-bressay:
	python src/testes/testar_ocr.py \
		--dataset bressay \
		--modelos-list configs/modelos_mixed_3.txt \
		--output resultados/bressay.json

# ============================================================
# MÉTRICAS
# ============================================================

.PHONY: metricas metricas-dharma

metricas:
	python src/metricas/calcular_metricas.py --entrada resultados/impresso.json --saida resultados/impresso_metricas.json
	python src/metricas/calcular_metricas.py --entrada resultados/manuscrito.json --saida resultados/manuscrito_metricas.json
	python src/metricas/calcular_metricas.py --entrada resultados/dharma.json --saida resultados/dharma_metricas.json
	python src/metricas/calcular_metricas.py --entrada resultados/deepinfra.json --saida resultados/deepinfra_metricas.json
	python src/metricas/calcular_metricas.py --entrada resultados/bressay.json --saida resultados/bressay_metricas.json

metricas-dharma:
	python src/metricas/calcular_metricas.py --entrada resultados/dharma.json --saida resultados/dharma_metricas.json

# ============================================================
# INSTALAÇÃO
# ============================================================

.PHONY: install

install:
	python -m pip install -r requirements.txt

# ============================================================
# LIMPEZA
# ============================================================

.PHONY: clean

clean:
	rm -rf assets/dados __pycache__ */__pycache__
