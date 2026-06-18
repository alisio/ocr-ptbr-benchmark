"""Testa 3 modelos OCR nas 10 amostras do DharmaOCR-Benchmark (com save incremental)."""
import base64, json, os, time, requests

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
DEEPINFRA_KEY = os.environ.get("DEEPINFRA_KEY", "")

BASE = "/Users/alisio/temp/ocr_results/amostras_dharma"
OUTPUT = "/Users/alisio/temp/ocr_results/resultados_dharma.json"

with open(f"{BASE}/manifest.json") as f:
    manifest = json.load(f)

PROMPT = "Extraia TODO o texto desta imagem exatamente como aparece, preservando a estrutura original (paragrafos, quebras de linha). Nao adicione comentarios nem explicacoes. Retorne apenas o texto extraido."

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

# Load existing results to resume
resultados = []
if os.path.exists(OUTPUT):
    with open(OUTPUT) as f:
        resultados = json.load(f)
    print(f"Carregados {len(resultados)} resultados existentes")

def already_done(modelo, nome):
    return any(r["modelo"] == modelo and r["nome"] == nome for r in resultados)

for m in MODELS:
    for amostra in manifest:
        tipo = amostra["tipo"]
        nome = amostra["nome"]
        
        if already_done(m["name"], nome):
            print(f"  SKIP (ja feito): {m['name'][:35]} | {tipo}/{nome}")
            continue
        
        img_rel = amostra["imagem"]
        img_path = f"{BASE}/{img_rel}"
        gt = amostra["ground_truth"]
        
        # Check for original (unresized) backup
        orig_path = img_path.replace(".png", "_original.png")
        if os.path.exists(orig_path):
            img_to_use = orig_path
        else:
            img_to_use = img_path

        print(f"\n=== {m['name']} | {tipo}/{nome} ===", flush=True)
        print(f"  GT: {len(gt)} chars, {len(gt.split())} words", flush=True)
        print(f"  Imagem: {img_to_use}", flush=True)

        with open(img_to_use, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

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
            "max_tokens": 8192
        }

        headers = {
            "Authorization": f"Bearer {m['key']}",
            "Content-Type": "application/json"
        }

        # Increase timeout for large images
        timeout_sec = 300
        t0 = time.time()
        try:
            r = requests.post(m["url"], headers=headers, json=body, timeout=timeout_sec)
            lat = time.time() - t0
            r.raise_for_status()
            data = r.json()
            texto = data["choices"][0]["message"]["content"].strip()
            pt = data.get("usage", {}).get("prompt_tokens", 0)
            ct = data.get("usage", {}).get("completion_tokens", 0)
            print(f"  OK | {lat:.1f}s | {pt} in / {ct} out | output: {len(texto)} chars", flush=True)
        except Exception as e:
            lat = time.time() - t0
            texto = ""
            pt = ct = 0
            print(f"  ERRO | {lat:.1f}s | {e}", flush=True)
            # Still save the error result

        resultados.append({
            "modelo": m["name"],
            "provedor": m["provider"],
            "tipo": tipo,
            "nome": nome,
            "texto": texto,
            "ground_truth": gt,
            "latencia": round(lat, 2),
            "prompt_tokens": pt,
            "completion_tokens": ct
        })

        # Save incrementally
        with open(OUTPUT, "w") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)

        # Rate limiting
        time.sleep(1.5)

print("\n\n=== RESUMO ===")
for r in resultados:
    gt = r["ground_truth"]
    texto = r["texto"]
    status = "OK" if texto else "ERRO"
    print(f"{r['modelo']:40s} | {r['tipo']:10s}/{r['nome']:25s} | {status} | {len(texto):5d}/{len(gt):5d} chars | {r['latencia']:5.1f}s")
