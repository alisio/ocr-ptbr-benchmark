import requests
import base64
import time
import json
import os
import sys
from openai import OpenAI

DEEPINFRA_KEY = os.environ.get("DEEPINFRA_API_KEY")
if not DEEPINFRA_KEY:
    print("ERRO: Defina DEEPINFRA_API_KEY no ambiente")
    sys.exit(1)

client = OpenAI(
    api_key=DEEPINFRA_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)

MODELOS = [
    "allenai/olmOCR-2-7B-1025",
    "PaddlePaddle/PaddleOCR-VL-0.9B",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL",
]

PROMPT = "Extraia todo o texto desta imagem exatamente como aparece, preservando a estrutura original (paragrafos, listas, tabelas). Nao adicione comentarios nem explicacoes. Retorne apenas o texto extraido."

caminho_imagem = "/Users/alisio/temp/ocr_results/documento_teste.png"
with open(caminho_imagem, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
    data_url = f"data:image/png;base64,{b64}"

precos = {
    "allenai/olmOCR-2-7B-1025": (0.09, 0.19),
    "PaddlePaddle/PaddleOCR-VL-0.9B": (0.14, 0.80),
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL": (0.20, 0.60),
}

resultados = []
for modelo in MODELOS:
    print(f"\n=== Testando: {modelo} ===")
    inicio = time.time()
    try:
        resp = client.chat.completions.create(
            model=modelo,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0,
            max_tokens=2048,
        )
        latencia = time.time() - inicio
        texto = resp.choices[0].message.content
        usage = resp.usage
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0

        r = {
            "modelo": modelo,
            "texto": texto,
            "latencia": round(latencia, 2),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "erro": None,
        }
        print(f"  OK | Lat: {latencia:.2f}s | Tokens: {pt} in / {ct} out")
        print(f"  Texto: {texto[:150].replace(chr(10), ' ')}...")

    except Exception as e:
        latencia = time.time() - inicio
        r = {
            "modelo": modelo,
            "texto": None,
            "latencia": round(latencia, 2),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "erro": str(e),
        }
        print(f"  ERRO: {e}")
    
    resultados.append(r)

saida = {
    "resultados": resultados,
}

with open("/Users/alisio/temp/ocr_results/resultados_deepinfra.json", "w") as f:
    json.dump(saida, f, indent=2, ensure_ascii=False)

print("\n\nResultados salvos em resultados_deepinfra.json")
