import base64, json, os, time, requests

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
DEEPINFRA_KEY = os.environ.get("DEEPINFRA_KEY", "")

BASE = "/Users/alisio/temp/ocr_results/bressay_sample/bressay/data/pages"
PAGES = ["7574-030", "4114-032", "8520-014"]

PROMPT = "Extraia TODO o texto desta imagem exatamente como aparece, preservando a estrutura original (parágrafos, quebras de linha). Nao adicione comentarios nem explicacoes. Retorne apenas o texto extraido."

MODELS = [
    {
        "name": "allenai/olmOCR-2-7B-1025",
        "provider": "deepinfra",
        "url": "https://api.deepinfra.com/v1/openai/chat/completions",
        "key": DEEPINFRA_KEY,
        "model": "allenai/olmOCR-2-7B-1025"
    },
    {
        "name": "qwen/qwen3-vl-32b-instruct",
        "provider": "openrouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key": OPENROUTER_KEY,
        "model": "qwen/qwen3-vl-32b-instruct"
    },
    {
        "name": "qwen/qwen3-vl-235b-a22b-instruct",
        "provider": "openrouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key": OPENROUTER_KEY,
        "model": "qwen/qwen3-vl-235b-a22b-instruct"
    }
]

resultados = []

for m in MODELS:
    for pid in PAGES:
        img_path = f"{BASE}/{pid}/{pid}.png"
        gt_path = f"{BASE}/{pid}/{pid}.txt"

        with open(gt_path) as f:
            ground_truth = f.read().strip()

        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        print(f"\n=== {m['name']} | {pid} ===")
        print(f"  GT: {len(ground_truth)} chars, {len(ground_truth.split())} words")

        data_url = f"data:image/png;base64,{b64}"
        body = {
            "model": m["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": PROMPT}
                    ]
                }
            ],
            "temperature": 0,
            "max_tokens": 4096
        }

        headers = {
            "Authorization": f"Bearer {m['key']}",
            "Content-Type": "application/json"
        }

        t0 = time.time()
        try:
            r = requests.post(m["url"], headers=headers, json=body, timeout=120)
            lat = time.time() - t0
            r.raise_for_status()
            data = r.json()
            texto = data["choices"][0]["message"]["content"].strip()
            pt = data.get("usage", {}).get("prompt_tokens", 0)
            ct = data.get("usage", {}).get("completion_tokens", 0)
            print(f"  OK | {lat:.1f}s | {pt} in / {ct} out")
            print(f"  Output ({len(texto)} chars): {texto[:150]}...")
        except Exception as e:
            lat = time.time() - t0
            texto = ""
            pt = ct = 0
            print(f"  ERRO | {lat:.1f}s | {e}")

        resultados.append({
            "modelo": m["name"],
            "provedor": m["provider"],
            "pagina": pid,
            "texto": texto,
            "ground_truth": ground_truth,
            "latencia": round(lat, 2),
            "prompt_tokens": pt,
            "completion_tokens": ct
        })

        # Rate limiting
        time.sleep(1)

with open("/Users/alisio/temp/ocr_results/resultados_bressay.json", "w") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print("\n\n=== RESUMO ===")
for r in resultados:
    gt = r["ground_truth"]
    texto = r["texto"]
    print(f"{r['modelo']:40s} | {r['pagina']:10s} | {len(texto):5d}/{len(gt):5d} chars | {r['latencia']:5.1f}s")
