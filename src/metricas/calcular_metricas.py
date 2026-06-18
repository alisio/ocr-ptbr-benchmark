#!/usr/bin/env python3
"""Calcula métricas OCR para resultados de testes."""
import json
import re
import argparse
import sys
from collections import defaultdict

PRECOS_MODELOS = {
    "allenai/olmOCR-2-7B-1025": (0.09, 0.19),
    "qwen/qwen3-vl-235b-a22b-instruct": (0.20, 0.88),
    "qwen/qwen3-vl-32b-instruct": (0.104, 0.416),
    "mistralai/mistral-large-2512": (0.50, 1.50),
    "meta-llama/llama-4-maverick": (0.15, 0.60),
    "google/gemma-4-31b-it": (0.12, 0.35),
    "PaddlePaddle/PaddleOCR-VL-0.9B": (0.14, 0.80),
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL": (0.20, 0.60),
}

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

def tokenizar(texto):
    return re.findall(r'\S+', texto)

def normalize(texto):
    return re.sub(r'\s+', ' ', texto).strip().lower()

def word_f1(hyp_words, ref_words):
    hyp_set = set(hyp_words)
    ref_set = set(ref_words)
    if not ref_set and not hyp_set:
        return 1.0
    if not ref_set:
        return 0.0
    inter = len(hyp_set & ref_set)
    prec = inter / len(hyp_set) if hyp_set else 0
    rec = inter / len(ref_set) if ref_set else 0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def try_bleu(ref_text, hyp_text):
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothie = SmoothingFunction().method1
        ref = [list(normalize(ref_text))]
        hyp = list(normalize(hyp_text))
        if len(hyp) < 4 or len(ref[0]) < 4:
            return 0.0
        return sentence_bleu(ref, hyp, smoothing_function=smoothie,
                            weights=(0.25, 0.25, 0.25, 0.25))
    except ImportError:
        return None

def calcular(gt, pred):
    if not pred:
        return {"cer": 1.0, "wer": 1.0, "f1": 0.0, "lv_ratio": 0.0, "bleu": 0.0, "score": 0.0, "deg": 0.0, "exact": 0}

    gt_s = gt.strip()
    pred_s = pred.strip()
    gt_norm = normalize(gt_s)
    pred_norm = normalize(pred_s)

    # Character-level
    edit_chars = levenshtein(pred_s, gt_s)
    cer = edit_chars / max(len(gt_s), 1)

    # Word-level (token)
    gt_tokens = tokenizar(gt_s)
    pred_tokens = tokenizar(pred_s)
    edit_words = levenshtein(pred_tokens, gt_tokens)
    wer = edit_words / max(len(gt_tokens), 1)

    # Word F1 (bag-of-words)
    gt_pal = palavras(gt_s)
    pred_pal = palavras(pred_s)
    f1 = word_f1(pred_pal, gt_pal)

    # Levenshtein Ratio (normalized similarity)
    lv_dist = levenshtein(gt_norm, pred_norm)
    lv_ratio = max(0, 1 - lv_dist / max(len(gt_norm), len(pred_norm), 1))

    # BLEU
    bleu = try_bleu(gt_s, pred_s) or 0.0

    # Composite score
    score = (lv_ratio + bleu) / 2

    # Degeneracao (output/GT length ratio)
    deg = len(pred_s) / max(len(gt_s), 1)

    exact = 1 if pred_s == gt_s else 0

    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "f1": round(f1, 4),
        "lv_ratio": round(lv_ratio, 4),
        "bleu": round(bleu, 4),
        "score": round(score, 4),
        "deg": round(deg, 4),
        "exact": exact,
    }

def get_custo(modelo, prompt_tokens, completion_tokens):
    p_in, p_out = PRECOS_MODELOS.get(modelo, (0, 0))
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000

def print_tabela(rows, headers):
    col_widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    sep = " | ".join("=" * w for w in col_widths)
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print(sep)
    for row in rows:
        line = " | ".join(str(r).ljust(w) for r, w in zip(row, col_widths))
        print(line)

def processar(resultados, verbose=True):
    # Group by sample or model
    if "ground_truth" in resultados[0]:
        # Per-sample results
        for r in resultados:
            gt = r.get("ground_truth", "")
            pred = r.get("texto", "") or ""
            r["metricas"] = calcular(gt, pred)
            lat = r.get("latencia", 0)
            pt = r.get("prompt_tokens", 0)
            ct = r.get("completion_tokens", 0)
            r["custo"] = round(get_custo(r.get("modelo", ""), pt, ct), 6)

    # Build per-model agg
    by_model = defaultdict(list)
    for r in resultados:
        by_model[r.get("modelo", "unknown")].append(r)

    all_metrics = {}
    rows = []
    for modelo, items in sorted(by_model.items()):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  {modelo}")
            print(f"{'='*60}")

        totals = {"cer": 0, "wer": 0, "f1": 0, "score": 0, "lv_ratio": 0, "bleu": 0}
        count = 0
        per_sample = {}
        custos = []
        lats = []

        for r in items:
            if "metricas" not in r:
                gt = r.get("ground_truth", "")
                pred = r.get("texto", "") or ""
                r["metricas"] = calcular(gt, pred)
            m = r["metricas"]
            nome = r.get("nome", r.get("pagina", "?"))
            tipo = r.get("tipo", "?")
            per_sample[nome] = m
            for k in totals:
                totals[k] += m[k]
            count += 1
            custos.append(r.get("custo", 0))
            lats.append(r.get("latencia", 0))

            if verbose:
                status = "✓" if m["score"] > 0.5 else "✗" if m["score"] > 0 else "?"
                print(f"  {status} {tipo:12s} {nome:25s} Score={m['score']:.4f} CER={m['cer']:.4f} WER={m['wer']:.4f} F1={m['f1']:.4f}")

        avg = {k: round(totals[k] / count, 4) if count else 0 for k in totals}
        avg_custo = sum(custos) / len(custos) if custos else 0
        avg_lat = sum(lats) / len(lats) if lats else 0
        all_metrics[modelo] = {"metrics": avg, "per_sample": per_sample}

        if verbose:
            print(f"\n  -> Média: Score={avg['score']:.4f} CER={avg['cer']:.4f} WER={avg['wer']:.4f} F1={avg['f1']:.4f} (n={count})")
            
        rows.append((avg["wer"], avg["cer"], avg["score"], avg["f1"], avg_lat, avg_custo, modelo))

    if verbose and rows:
        rows.sort(key=lambda x: x[0])
        print(f"\n\n{'='*60}")
        print("  RANKING (por WER)")
        print(f"{'='*60}")
        print_tabela(
            [(i, r[6][:40], r[1], r[0], r[3], r[2], r[4], f"${r[5]:.5f}") for i, r in enumerate(rows, 1)],
            ["#", "Modelo", "CER", "WER", "F1", "Score", "Lat(s)", "Custo"]
        )

    return all_metrics

def main():
    parser = argparse.ArgumentParser(description="Calcular métricas OCR")
    parser.add_argument("--entrada", "-i", required=True, help="Arquivo JSON com resultados")
    parser.add_argument("--saida", "-o", help="Arquivo de saída para métricas (opcional)")
    parser.add_argument("--ground-truth", "-g", help="Arquivo de ground truth externo (para formatos antigos)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Modo silencioso")
    args = parser.parse_args()

    if not os.path.exists(args.entrada):
        print(f"Erro: arquivo não encontrado: {args.entrada}")
        sys.exit(1)

    with open(args.entrada) as f:
        dados = json.load(f)

    # Detect format: list of results or nested object
    gt_global = None
    if isinstance(dados, dict):
        gt_global = dados.get("ground_truth", dados.get("GT"))
        resultados = dados.get("resultados", dados.get("results", []))
    elif isinstance(dados, list):
        resultados = dados
    else:
        print("Erro: formato de dados não reconhecido")
        sys.exit(1)

    # Override GT from external file if provided
    if args.ground_truth:
        with open(args.ground_truth) as f:
            gt_global = f.read().strip()

    # Inject global ground_truth into each result entry if missing
    if gt_global:
        for r in resultados:
            if "ground_truth" not in r or not r["ground_truth"]:
                r["ground_truth"] = gt_global

    all_metrics = processar(resultados, verbose=not args.quiet)

    if args.saida:
        os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
        with open(args.saida, "w") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"\nMétricas salvas em {args.saida}")

if __name__ == "__main__":
    import os
    main()