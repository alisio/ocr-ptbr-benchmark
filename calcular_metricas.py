import json
import math

with open("/Users/alisio/temp/ocr_results/resultados_brutos.json") as f:
    dados = json.load(f)

GT = dados["ground_truth"]
resultados = dados["resultados"]

def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = range(len(b) + 1)
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            custo = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + custo))
        prev = curr
    return prev[-1]

def tokenizar(texto):
    import re
    tokens = re.findall(r'\S+', texto)
    return tokens

def palavras(texto):
    import re
    return re.findall(r'\b\w+\b', texto.lower())

gt_palavras_set = set(palavras(GT))
gt_tokens = tokenizar(GT)
gt_palavras = palavras(GT)
total_chars = len(GT)
total_words = len(gt_palavras)
total_tokens = len(gt_tokens)

print("=" * 100)
print(f"{'Modelo':<42} {'CER':>8} {'WER':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Exact':>8} {'Lat(s)':>8} {'Custo':>10}")
print("=" * 100)

for r in resultados:
    modelo = r["modelo"]
    texto = r["texto"] or ""
    latencia = r["latencia"]

    tokens_in = r["prompt_tokens"]
    tokens_out = r["completion_tokens"]
    precos = {
        "qwen/qwen3-vl-235b-a22b-instruct": (0.20, 0.88),
        "qwen/qwen3-vl-32b-instruct": (0.104, 0.416),
        "mistralai/mistral-large-2512": (0.50, 1.50),
        "meta-llama/llama-4-maverick": (0.15, 0.60),
        "google/gemma-4-31b-it": (0.12, 0.35),
    }
    p_in, p_out = precos.get(modelo, (0, 0))
    custo = (tokens_in * p_in + tokens_out * p_out) / 1_000_000

    texto = texto.strip()
    gt_strip = GT.strip()

    edit_dist_chars = levenshtein(texto, gt_strip)
    cer = edit_dist_chars / max(len(gt_strip), 1)

    hip_tokens = tokenizar(texto)
    gt_tok = tokenizar(gt_strip)
    edit_dist_words = levenshtein(hip_tokens, gt_tok)
    wer = edit_dist_words / max(len(gt_tok), 1)

    hip_palavras = palavras(texto)
    gt_pal = palavras(gt_strip)
    hip_set = set(hip_palavras)
    gt_set = set(gt_pal)

    corretas = len(hip_set & gt_set)
    precision = corretas / max(len(hip_set), 1)
    recall = corretas / max(len(gt_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    exact = int(texto == gt_strip)

    nome_curto = modelo
    print(f"{nome_curto:<42} {cer:>8.4f} {wer:>8.4f} {precision:>8.4f} {recall:>8.4f} {f1:>8.4f} {exact:>8} {latencia:>8.2f} ${custo:>8.5f}")

print("=" * 100)
print(f"\nGround truth: {total_chars} chars, {total_words} palavras, {total_tokens} tokens")

print("\n\n=== DETALHAMENTO POR MODELO ===")
print("=" * 100)

for r in resultados:
    modelo = r["modelo"]
    texto = (r["texto"] or "").strip()
    gt_strip = GT.strip()

    print(f"\n--- {modelo} ---")

    if texto == gt_strip:
        print("  STATUS: EXACT MATCH")
    else:
        gt_lines = gt_strip.split('\n')
        hip_lines = texto.split('\n')

        print(f"  Linhas GT: {len(gt_lines)}, Linhas saída: {len(hip_lines)}")
        print(f"  Chars GT: {len(gt_strip)}, Chars saída: {len(texto)}")

        dist = levenshtein(texto, gt_strip)
        print(f"  Distância Levenshtein (chars): {dist}")

        import difflib
        differ = difflib.SequenceMatcher(None, gt_strip, texto)
        diff_ratio = differ.ratio()
        print(f"  Similaridade (difflib): {diff_ratio:.4f}")

        gt_w = palavras(gt_strip)
        hip_w = palavras(texto)
        gt_set = set(gt_w)
        hip_set = set(hip_w)

        faltando = gt_set - hip_set
        extras = hip_set - gt_set

        if faltando:
            print(f"  Palavras GT não encontradas: {sorted(faltando)[:15]}")
        if extras:
            print(f"  Palavras extras na saída: {sorted(extras)[:15]}")

        if len(gt_lines) <= 20:
            print("\n  LINHAS GT x SAÍDA:")
            matcher = difflib.SequenceMatcher(None, gt_strip.split('\n'), texto.split('\n'))
            for op, i1, i2, j1, j2 in matcher.get_opcodes():
                if op == 'equal':
                    pass
                elif op == 'replace':
                    for k in range(i1, i2):
                        print(f"  - [GT]: {gt_lines[k]}")
                    for k in range(j1, j2):
                        print(f"  + [SAIDA]: {hip_lines[k]}")
                elif op == 'delete':
                    for k in range(i1, i2):
                        print(f"  - [GT]: {gt_lines[k]}")
                elif op == 'insert':
                    for k in range(j1, j2):
                        print(f"  + [SAIDA]: {hip_lines[k]}")

print("\n\n=== RANKING FINAL (por WER) ===")
print("=" * 70)

ranking = []
for r in resultados:
    modelo = r["modelo"]
    texto = (r["texto"] or "").strip()
    gt_strip = GT.strip()

    hip_tokens = tokenizar(texto)
    gt_tok = tokenizar(gt_strip)
    edit_dist_words = levenshtein(hip_tokens, gt_tok)
    wer = edit_dist_words / max(len(gt_tok), 1)

    edit_dist_chars = levenshtein(texto, gt_strip)
    cer = edit_dist_chars / max(len(gt_strip), 1)

    hip_palavras = palavras(texto)
    gt_pal = palavras(gt_strip)
    hip_set = set(hip_palavras)
    gt_set = set(gt_pal)
    corretas = len(hip_set & gt_set)
    precision = corretas / max(len(hip_set), 1)
    recall = corretas / max(len(gt_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    tokens_in = r["prompt_tokens"]
    tokens_out = r["completion_tokens"]
    precos = {
        "qwen/qwen3-vl-235b-a22b-instruct": (0.20, 0.88),
        "qwen/qwen3-vl-32b-instruct": (0.104, 0.416),
        "mistralai/mistral-large-2512": (0.50, 1.50),
        "meta-llama/llama-4-maverick": (0.15, 0.60),
        "google/gemma-4-31b-it": (0.12, 0.35),
    }
    p_in, p_out = precos.get(modelo, (0, 0))
    custo = (tokens_in * p_in + tokens_out * p_out) / 1_000_000

    ranking.append((wer, cer, f1, modelo, r["latencia"], custo, precision, recall))

ranking.sort(key=lambda x: x[0])

print(f"{'#':>2} {'Modelo':<42} {'CER':>8} {'WER':>8} {'F1':>8} {'Lat(s)':>8} {'Custo':>10}")
print("-" * 90)
for i, (wer, cer, f1, modelo, lat, custo, prec, rec) in enumerate(ranking, 1):
    print(f"{i:>2} {modelo:<42} {cer:>8.4f} {wer:>8.4f} {f1:>8.4f} {lat:>8.2f} ${custo:>8.5f}")

print()
print("Legenda:")
print("  CER = Character Error Rate (menor = melhor)")
print("  WER = Word Error Rate (menor = melhor)")
print("  F1  = Média harmônica Precision/Recall (maior = melhor)")
print("  Custo = USD gasto por requisição (tokens * preço / 1M)")
