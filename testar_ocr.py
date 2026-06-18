import requests
import base64
import time
import json
import os
import sys

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    print("ERRO: Defina OPENROUTER_API_KEY no ambiente")
    sys.exit(1)

MODELOS = [
    "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen/qwen3-vl-32b-instruct",
    "mistralai/mistral-large-2512",
    "meta-llama/llama-4-maverick",
    "google/gemma-4-31b-it",
]

PROMPT = "Extraia todo o texto desta imagem exatamente como aparece, preservando a estrutura original (parágrafos, listas, tabelas). Nao adicione comentarios nem explicacoes. Retorne apenas o texto extraido."

def testar_modelo(modelo, imagem_b64):
    inicio = time.time()
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": modelo,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{imagem_b64}"
                                },
                            },
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 2048,
            },
            timeout=120,
        )
        latencia = time.time() - inicio
        resp.raise_for_status()
        data = resp.json()

        texto = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return {
            "modelo": modelo,
            "texto": texto,
            "latencia": round(latencia, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "erro": None,
        }
    except Exception as e:
        latencia = time.time() - inicio
        return {
            "modelo": modelo,
            "texto": None,
            "latencia": round(latencia, 2),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "erro": str(e),
        }

caminho_imagem = "/Users/alisio/temp/ocr_results/documento_teste.png"
with open(caminho_imagem, "rb") as f:
    imagem_b64 = base64.b64encode(f.read()).decode("utf-8")

with open("/Users/alisio/temp/ocr_results/ground_truth.txt") as f:
    GROUND_TRUTH = f.read().strip()

resultados = []
for modelo in MODELOS:
    print(f"\n=== Testando: {modelo} ===")
    resultado = testar_modelo(modelo, imagem_b64)
    if resultado["erro"]:
        print(f"  ERRO: {resultado['erro']}")
    else:
        print(f"  Latencia: {resultado['latencia']}s")
        print(f"  Tokens: {resultado['prompt_tokens']} in / {resultado['completion_tokens']} out")
        texto_exib = resultado["texto"][:200].replace("\n", "\\n")
        print(f"  Texto (inicio): {texto_exib}")
    resultados.append(resultado)

saida = {
    "ground_truth": GROUND_TRUTH,
    "resultados": resultados,
}

with open("/Users/alisio/temp/ocr_results/resultados_brutos.json", "w") as f:
    json.dump(saida, f, indent=2, ensure_ascii=False)

print("\n\nResultados salvos em resultados_brutos.json")
