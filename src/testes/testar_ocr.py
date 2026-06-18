#!/usr/bin/env python3
"""Script unificado para testar OCR em imagens."""
import requests
import base64
import time
import json
import os
import sys
import argparse
from openai import OpenAI

PROMPT_PADRAO = "Extraia TODO o texto desta imagem exatamente como aparece, preservando a estrutura original (paragrafos, quebras de linha). Nao adicione comentarios nem explicacoes. Retorne apenas o texto extraido."

PROVEDEORES = {
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "use_openai": False,
    },
    "deepinfra": {
        "url": "https://api.deepinfra.com/v1/openai/chat/completions",
        "key_env": "DEEPINFRA_API_KEY",
        "use_openai": False,
    },
}

def get_client(provider):
    """Retorna client OpenAI ou None se usar requests direto."""
    config = PROVEDEORES.get(provider)
    if not config:
        raise ValueError(f"Provider desconhecido: {provider}")
    return config

def testar_modelo(modelo, provider, img_path, key):
    """Testa um modelo específico em uma imagem."""
    config = get_client(provider)

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    data_url = f"data:image/png;base64,{b64}"

    if provider == "openrouter":
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": modelo,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_PADRAO},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 8192,
        }
        t0 = time.time()
        try:
            resp = requests.post(config["url"], headers=headers, json=body, timeout=300)
            latencia = time.time() - t0
            resp.raise_for_status()
            data = resp.json()
            texto = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {
                "modelo": modelo,
                "provedor": provider,
                "texto": texto,
                "latencia": round(latencia, 2),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "erro": None,
            }
        except Exception as e:
            return {
                "modelo": modelo,
                "provedor": provider,
                "texto": None,
                "latencia": round(time.time() - t0, 2),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "erro": str(e),
            }
    else:
        client = OpenAI(api_key=key, base_url=config["url"])
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=modelo,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT_PADRAO},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                temperature=0,
                max_tokens=8192,
            )
            latencia = time.time() - t0
            texto = resp.choices[0].message.content
            usage = resp.usage
            return {
                "modelo": modelo,
                "provedor": provider,
                "texto": texto,
                "latencia": round(latencia, 2),
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "erro": None,
            }
        except Exception as e:
            return {
                "modelo": modelo,
                "provedor": provider,
                "texto": None,
                "latencia": round(time.time() - t0, 2),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "erro": str(e),
            }

def carregar_manifest(dataset_path):
    """Carrega manifesto do dataset."""
    manifest_path = os.path.join(dataset_path, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest não encontrado: {manifest_path}")
    with open(manifest_path) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Testar OCR em imagens")
    parser.add_argument("--provider", "-p", choices=["openrouter", "deepinfra", "both"], default="openrouter", help="Provider a usar")
    parser.add_argument("--modelos", "-m", help="Modelos separados por vírgula")
    parser.add_argument("--modelos-list", help="Arquivo com lista de modelos (um por linha)")
    parser.add_argument("--amostras", "-a", help="Imagem única ou diretório de amostras")
    parser.add_argument("--dataset", "-d", choices=["dharma", "manuscrito", "impresso", "bressay"], help="Dataset pré-definido")
    parser.add_argument("--output", "-o", default="resultados/resultados.json", help="Arquivo de saída")
    parser.add_argument("--prompt", default=PROMPT_PADRAO, help="Prompt customizado")
    parser.add_argument("--resume", action="store_true", help="Reaproveitar resultados existentes")
    args = parser.parse_args()

    # Build modelos list
    modelos = []
    if args.modelos:
        for entry in args.modelos.split(","):
            parts = entry.strip().split(",")
            if len(parts) == 2:
                modelos.append({"name": parts[0], "provider": parts[1]})
            else:
                modelos.append({"name": parts[0], "provider": args.provider})
    elif args.modelos_list:
        with open(args.modelos_list) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(",")
                    if len(parts) == 2:
                        modelos.append({"name": parts[0], "provider": parts[1]})
                    else:
                        modelos.append({"name": parts[0], "provider": args.provider})
    else:
        # Default modelos
        modelos = [
            {"name": "qwen/qwen3-vl-235b-a22b-instruct", "provider": "openrouter"},
            {"name": "qwen/qwen3-vl-32b-instruct", "provider": "openrouter"},
            {"name": "mistralai/mistral-large-2512", "provider": "openrouter"},
        ]

    resultados = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            resultados = json.load(f)
        print(f"Reaproveitando {len(resultados)} resultados")

    def ja_feito(modelo, nome):
        return any(r["modelo"] == modelo and r["nome"] == nome for r in resultados)

    # Determinar amostras
    amostras = []
    if args.dataset:
        dataset_path = f"dados/{args.dataset}" if args.dataset != "dharma" else "dados/amostras_dharma"
        manifest = carregar_manifest(dataset_path)
        for item in manifest:
            amostras.append({
                "nome": item["nome"],
                "tipo": item["tipo"],
                "imagem": os.path.join(dataset_path, item["imagem"]),
                "ground_truth": item["ground_truth"],
            })
    elif args.amostras:
        if os.path.isfile(args.amostras):
            # Imagem única - procurar ground truth
            base = os.path.dirname(args.amostras)
            nome = os.path.splitext(os.path.basename(args.amostras))[0]
            gt_path = os.path.join(base, f"{nome}.txt") or os.path.join(base, f"ground_truth.txt")
            if os.path.exists(gt_path):
                with open(gt_path) as f:
                    gt = f.read()
            else:
                gt = ""
            amostras.append({"nome": nome, "tipo": "custom", "imagem": args.amostras, "ground_truth": gt})
        else:
            # Diretório de amostras
            for root, _, files in os.walk(args.amostras):
                for f in files:
                    if f.endswith((".png", ".jpg", ".jpeg")):
                        nome = os.path.splitext(f)[0]
                        img_path = os.path.join(root, f)
                        gt_path = img_path.rsplit(".", 1)[0] + ".txt"
                        if os.path.exists(gt_path):
                            with open(gt_path) as gf:
                                gt = gf.read()
                        else:
                            gt = ""
                        tipo = os.path.basename(root)
                        amostras.append({"nome": nome, "tipo": tipo, "imagem": img_path, "ground_truth": gt})
    else:
        print("Erro: especifique --amostras ou --dataset")
        sys.exit(1)

    # Executar testes
    total = len(modelos) * len(amostras)
    count = 0
    for modelo_info in modelos:
        modelo = modelo_info["name"]
        provider = modelo_info["provider"]
        key = os.environ.get(PROVEDEORES[provider]["key_env"])
        if not key:
            print(f"Erro: defina {PROVEDEORES[provider]['key_env']} no ambiente")
            continue

        for amostra in amostras:
            if ja_feito(modelo, amostra["nome"]):
                print(f"SKIP: {modelo} | {amostra['nome']}")
                continue

            print(f"\n[{provider}] {modelo} | {amostra['tipo']}/{amostra['nome']}")
            print(f"  GT: {len(amostra['ground_truth'])} chars")

            resultado = testar_modelo(modelo, provider, amostra["imagem"], key)
            resultado["nome"] = amostra["nome"]
            resultado["tipo"] = amostra["tipo"]
            resultado["ground_truth"] = amostra["ground_truth"]

            if resultado["erro"]:
                print(f"  ERRO: {resultado['erro']}")
            else:
                texto = resultado["texto"] or ""
                print(f"  OK | {resultado['latencia']}s | {resultado['prompt_tokens']}/{resultado['completion_tokens']} tokens | {len(texto)} chars")

            resultados.append(resultado)
            count += 1

            # Save incremental
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False)

            time.sleep(1.5)

    print(f"\n=== RESUMO ===")
    print(f"Total: {len(resultados)} testes ({count} novos)")

if __name__ == "__main__":
    main()