import json, re

with open("/Users/alisio/temp/ocr_results/resultados_bressay.json") as f:
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

def palavras(texto):
    return re.findall(r'\b\w+\b', texto.lower())

precos = {
    "allenai/olmOCR-2-7B-1025": (0.09, 0.19),
    "qwen/qwen3-vl-32b-instruct": (0.104, 0.416),
    "qwen/qwen3-vl-235b-a22b-instruct": (0.20, 0.88),
}

print(f"{'Modelo':<40} {'Pagina':<12} {'CER':>8} {'WER':>8} {'F1':>8} {'Lat(s)':>8} {'Custo':>12} {'Exact':>6}")
print("=" * 100)

for r in resultados:
    modelo = r["modelo"]
    pid = r["pagina"]
    texto = (r["texto"] or "").strip()
    gt = r["ground_truth"].strip()
    
    cer = levenshtein(texto, gt) / max(len(gt), 1)
    
    hip_tok = texto.split()
    gt_tok = gt.split()
    wer = levenshtein(hip_tok, gt_tok) / max(len(gt_tok), 1)
    
    hip_pal = set(palavras(texto))
    gt_pal = set(palavras(gt))
    corretas = len(hip_pal & gt_pal)
    prec = corretas / max(len(hip_pal), 1)
    rec = corretas / max(len(gt_pal), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    
    p_in, p_out = precos.get(modelo, (0, 0))
    custo = (r["prompt_tokens"] * p_in + r["completion_tokens"] * p_out) / 1_000_000
    
    exact = 1 if texto == gt else 0
    
    print(f"{modelo:<40} {pid:<12} {cer:>8.4f} {wer:>8.4f} {f1:>8.4f} {r['latencia']:>8.2f} ${custo:>10.5f} {exact:>6}")

# Per-model averages
print("\n\n=== MÉDIAS POR MODELO ===")
from collections import defaultdict
agg = defaultdict(lambda: {"cer": [], "wer": [], "f1": [], "lat": [], "custo": []})
for r in resultados:
    m = r["modelo"]
    texto = (r["texto"] or "").strip()
    gt = r["ground_truth"].strip()
    cer = levenshtein(texto, gt) / max(len(gt), 1)
    hip_tok = texto.split()
    gt_tok = gt.split()
    wer = levenshtein(hip_tok, gt_tok) / max(len(gt_tok), 1)
    hip_pal = set(palavras(texto))
    gt_pal = set(palavras(gt))
    corretas = len(hip_pal & gt_pal)
    prec = corretas / max(len(hip_pal), 1)
    rec = corretas / max(len(gt_pal), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    p_in, p_out = precos.get(m, (0, 0))
    custo = (r["prompt_tokens"] * p_in + r["completion_tokens"] * p_out) / 1_000_000
    
    agg[m]["cer"].append(cer)
    agg[m]["wer"].append(wer)
    agg[m]["f1"].append(f1)
    agg[m]["lat"].append(r["latencia"])
    agg[m]["custo"].append(custo)

print(f"{'Modelo':<40} {'CER':>8} {'WER':>8} {'F1':>8} {'Lat(s)':>8} {'Custo':>12}")
print("=" * 80)
for m, v in sorted(agg.items(), key=lambda x: sum(x[1]["wer"])/len(x[1]["wer"])):
    print(f"{m:<40} {sum(v['cer'])/len(v['cer']):>8.4f} {sum(v['wer'])/len(v['wer']):>8.4f} {sum(v['f1'])/len(v['f1']):>8.4f} {sum(v['lat'])/len(v['lat']):>8.2f} ${sum(v['custo'])/len(v['custo']):>10.5f}")

# Per-page averages
print("\n\n=== MÉDIAS POR PÁGINA ===")
pag_agg = defaultdict(list)
for r in resultados:
    pid = r["pagina"]
    texto = (r["texto"] or "").strip()
    gt = r["ground_truth"].strip()
    cer = levenshtein(texto, gt) / max(len(gt), 1)
    hip_tok = texto.split()
    gt_tok = gt.split()
    wer = levenshtein(hip_tok, gt_tok) / max(len(gt_tok), 1)
    pag_agg[pid].append((r["modelo"][:30], cer, wer))

for pid in ["7574-030", "4114-032", "8520-014"]:
    print(f"\n--- {pid} ---")
    for nome, cer, wer in pag_agg[pid]:
        print(f"  {nome:<35} CER={cer:.4f}  WER={wer:.4f}")

# Show first-line comparison
print("\n\n=== ERROS NA PRIMEIRA FRASE (título/tema) ===")
for r in resultados:
    gt_first = r["ground_truth"].strip().split('\n')[0]
    out_first = (r["texto"] or "").strip().split('\n')[0]
    if gt_first != out_first:
        print(f"\n  {r['modelo']:40s} | {r['pagina']}")
        print(f"    GT:  {gt_first[:100]}")
        print(f"    Out: {out_first[:100]}")
