import requests, base64, time, json, os, sys
from openai import OpenAI

OR_KEY = os.environ.get("OPENROUTER_API_KEY")
DI_KEY = os.environ.get("DEEPINFRA_API_KEY")

if not OR_KEY or not DI_KEY:
    print("ERRO: Defina OPENROUTER_API_KEY e DEEPINFRA_API_KEY")
    sys.exit(1)

# Models: OCR specialized vs VLM
MODELOS = [
    ("allenai/olmOCR-2-7B-1025", "deepinfra"),
    ("qwen/qwen3-vl-32b-instruct", "openrouter"),
    ("qwen/qwen3-vl-235b-a22b-instruct", "openrouter"),
]

PROMPT = "Extraia TODO o texto escrito a mão nesta imagem exatamente como aparece. Preserve a estrutura (listas, paragrafos). Nao adicione comentarios. Retorne apenas o texto."

caminho = "/Users/alisio/temp/ocr_results/documento_manuscrito.png"
with open(caminho, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
    data_url = f"data:image/png;base64,{b64}"

client_or = OpenAI(api_key=OR_KEY, base_url="https://openrouter.ai/api/v1")
client_di = OpenAI(api_key=DI_KEY, base_url="https://api.deepinfra.com/v1/openai")

precos = {
    "allenai/olmOCR-2-7B-1025": (0.09, 0.19),
    "qwen/qwen3-vl-32b-instruct": (0.104, 0.416),
    "qwen/qwen3-vl-235b-a22b-instruct": (0.20, 0.88),
}

resultados = []
for modelo, provider in MODELOS:
    print(f"\n=== {modelo} ({provider}) ===")
    client = client_di if provider == "deepinfra" else client_or
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
            max_tokens=1024,
        )
        latencia = time.time() - inicio
        texto = resp.choices[0].message.content
        usage = resp.usage
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0
        r = {
            "modelo": modelo,
            "provider": provider,
            "texto": texto,
            "latencia": round(latencia, 2),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "erro": None,
        }
        print(f"  OK | {latencia:.1f}s | {pt} in / {ct} out")
        print(f"  Saida: {texto[:300]}")
    except Exception as e:
        latencia = time.time() - inicio
        r = {
            "modelo": modelo, "provider": provider,
            "texto": None, "latencia": round(latencia, 2),
            "prompt_tokens": 0, "completion_tokens": 0, "erro": str(e),
        }
        print(f"  ERRO: {e}")
    resultados.append(r)

with open("/Users/alisio/temp/ocr_results/resultados_manuscrito.json", "w") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print("\n\nResultados salvos!")
