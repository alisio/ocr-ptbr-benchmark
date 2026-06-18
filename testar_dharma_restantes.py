"""Testa modelos restantes (gemma, paddle, nemotron + completar mistral-large)."""
import base64, json, os, time, requests

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
DEEPINFRA_KEY = os.environ.get("DEEPINFRA_KEY", "")

BASE = "/Users/alisio/temp/ocr_results/amostras_dharma"
OUTPUT = "/Users/alisio/temp/ocr_results/resultados_dharma.json"

with open(f"{BASE}/manifest.json") as f:
    manifest = json.load(f)

PROMPT = "Extraia TODO o texto desta imagem exatamente como aparece, preservando a estrutura original (paragrafos, quebras de linha). Nao adicione comentarios nem explicacoes. Retorne apenas o texto extraido."

# Só os modelos faltantes
NOVOS = [
    {"name": "mistralai/mistral-large-2512", "provider": "openrouter",
     "url": "https://openrouter.ai/api/v1/chat/completions", "key": OPENROUTER_KEY,
     "model": "mistralai/mistral-large-2512"},
    {"name": "google/gemma-4-31b-it", "provider": "openrouter",
     "url": "https://openrouter.ai/api/v1/chat/completions", "key": OPENROUTER_KEY,
     "model": "google/gemma-4-31b-it"},
    {"name": "PaddlePaddle/PaddleOCR-VL-0.9B", "provider": "deepinfra",
     "url": "https://api.deepinfra.com/v1/openai/chat/completions", "key": DEEPINFRA_KEY,
     "model": "PaddlePaddle/PaddleOCR-VL-0.9B"},
    {"name": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL", "provider": "deepinfra",
     "url": "https://api.deepinfra.com/v1/openai/chat/completions", "key": DEEPINFRA_KEY,
     "model": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL"},
]

resultados = []
if os.path.exists(OUTPUT):
    with open(OUTPUT) as f:
        resultados = json.load(f)

def already_done(modelo, nome):
    return any(r["modelo"] == modelo and r["nome"] == nome and r.get("texto", "") for r in resultados)

for m in NOVOS:
    for amostra in manifest:
        nome = amostra["nome"]
        tipo = amostra["tipo"]
        
        if already_done(m["name"], nome):
            print(f"  SKIP: {m['name'][:40]} | {tipo}/{nome}")
            continue
        
        # Remove entries from prior failed attempts
        resultados = [r for r in resultados if not (r["modelo"] == m["name"] and r["nome"] == nome)]
        
        img_path = f"{BASE}/{amostra['imagem']}"
        gt = amostra["ground_truth"]

        print(f"\n[{m['provider']}] {m['name']} | {tipo}/{nome}", flush=True)

        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        data_url = f"data:image/png;base64,{b64}"
        body = {
            "model": m["model"],
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": PROMPT
                }]}],
            "temperature": 0,
            "max_tokens": 8192
        }

        headers = {"Authorization": f"Bearer {m['key']}", "Content-Type": "application/json"}

        t0 = time.time()
        try:
            r = requests.post(m["url"], headers=headers, json=body, timeout=600)
            lat = time.time() - t0
            r.raise_for_status()
            data = r.json()
            texto = data["choices"][0]["message"]["content"].strip()
            pt = data.get("usage", {}).get("prompt_tokens", 0)
            ct = data.get("usage", {}).get("completion_tokens", 0)
            print(f"  OK | {lat:.1f}s | {pt}/{ct} | {len(texto)} chars", flush=True)
        except Exception as e:
            lat = time.time() - t0
            texto = ""
            pt = ct = 0
            print(f"  ERRO | {lat:.1f}s | {e}", flush=True)

        resultados.append({
            "modelo": m["name"], "provedor": m["provider"], "tipo": tipo, "nome": nome,
            "texto": texto, "ground_truth": gt,
            "latencia": round(lat, 2), "prompt_tokens": pt, "completion_tokens": ct
        })
        with open(OUTPUT, "w") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        time.sleep(1.5)

print("\n\n=== FINAL ===")
by_model = {}
for r in resultados:
    by_model.setdefault(r["modelo"], []).append(r)
for m, rs in sorted(by_model.items()):
    ok = sum(1 for r in rs if r.get("texto", ""))
    print(f"{m[:45]:45s}: {ok}/{len(rs)} OK")
