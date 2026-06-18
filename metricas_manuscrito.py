import json, re

with open("/Users/alisio/temp/ocr_results/ground_truth_manuscrito.txt") as f:
    GT = f.read().strip()
with open("/Users/alisio/temp/ocr_results/resultados_manuscrito.json") as f:
    resultados = json.load(f)

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
    return re.findall(r'\S+', texto)

def palavras(texto):
    return re.findall(r'\b\w+\b', texto.lower())

precos = {
    "allenai/olmOCR-2-7B-1025": (0.09, 0.19),
    "qwen/qwen3-vl-32b-instruct": (0.104, 0.416),
    "qwen/qwen3-vl-235b-a22b-instruct": (0.20, 0.88),
}

print(f"{'Modelo':<50} {'CER':>8} {'WER':>8} {'F1':>8} {'Lat(s)':>8} {'Custo':>10} {'Exact':>8}")
print("=" * 100)

for r in resultados:
    modelo = r["modelo"]
    texto = (r["texto"] or "").strip()
    gt = GT.strip()
    
    if not texto:
        print(f"{modelo:<50} {'SEM SAIDA':>8}")
        continue
    
    cer = levenshtein(texto, gt) / max(len(gt), 1)
    
    hip_tok = tokenizar(texto)
    gt_tok = tokenizar(gt)
    wer = levenshtein(hip_tok, gt_tok) / max(len(gt_tok), 1)
    
    hip_pal = palavras(texto)
    gt_pal = palavras(gt)
    hip_set = set(hip_pal)
    gt_set = set(gt_pal)
    corretas = len(hip_set & gt_set)
    prec = corretas / max(len(hip_set), 1)
    rec = corretas / max(len(gt_set), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    
    p_in, p_out = precos.get(modelo, (0, 0))
    custo = (r["prompt_tokens"] * p_in + r["completion_tokens"] * p_out) / 1_000_000
    
    exact = 1 if texto == gt else 0
    
    print(f"{modelo:<50} {cer:>8.4f} {wer:>8.4f} {f1:>8.4f} {r['latencia']:>8.2f} ${custo:>8.5f} {exact:>8}")

print(f"\n\nGround truth ({len(GT)} chars, {len(palavras(GT))} palavras):")
print(GT)

print("\n\n=== ANÁLISE DETALHADA ===")
for r in resultados:
    modelo = r["modelo"]
    texto = (r["texto"] or "").strip()
    gt = GT.strip()
    
    if not texto:
        continue
    
    print(f"\n--- {modelo} ---")
    
    # Check what differs
    gt_words = set(palavras(gt))
    out_words = set(palavras(texto))
    
    faltando = gt_words - out_words
    extras = out_words - gt_words
    
    if faltando:
        print(f"  Palavras GT ausentes: {sorted(faltando)}")
    if extras:
        print(f"  Palavras extras: {sorted(extras)}")
    
    if texto == gt:
        print("  ✅ EXACT MATCH")
    else:
        # Show line differences
        gt_lines = gt.split('\n')
        out_lines = texto.split('\n')
        for i in range(max(len(gt_lines), len(out_lines))):
            g = gt_lines[i] if i < len(gt_lines) else '(missing)'
            o = out_lines[i] if i < len(out_lines) else '(missing)'
            if g != o:
                print(f"  Linha {i}:")
                print(f"    GT:  {repr(g)}")
                print(f"    Out: {repr(o)}")
