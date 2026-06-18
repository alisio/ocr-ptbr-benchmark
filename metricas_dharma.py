"""Calcula métricas para todos os modelos no resultados_dharma.json."""
import json, re
from collections import defaultdict
try:
    import Levenshtein
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-Levenshtein"])
    import Levenshtein

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    import nltk
    nltk.download('punkt_tbl')
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smoothie = SmoothingFunction().method1

def normalize(t):
    return re.sub(r'\s+', ' ', t).strip().lower()

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

def calc_metrics(gt, pred):
    if not pred:
        return {"lv_ratio": 0, "bleu": 0, "score": 0, "cer": 1, "wer": 1, "f1": 0, "deg": 0}
    
    gt_n = normalize(gt)
    pred_n = normalize(pred)
    gt_words = gt_n.split()
    pred_words = pred_n.split()
    
    # Levenshtein Ratio (character-level similarity)
    lv_dist = Levenshtein.distance(gt_n, pred_n)
    lv_ratio = max(0, 1 - lv_dist / max(len(gt_n), len(pred_n), 1))
    
    # BLEU char-level (n-gram precision up to 4)
    ref = [list(gt_n)]
    hyp = list(pred_n)
    if len(hyp) < 4 or len(ref[0]) < 4:
        bleu = 0.0
    else:
        try:
            bleu = sentence_bleu(ref, hyp, smoothing_function=smoothie, 
                                weights=(0.25, 0.25, 0.25, 0.25))
        except:
            bleu = 0.0
    
    score = (lv_ratio + bleu) / 2
    
    # CER (Character Error Rate)
    cer = lv_dist / max(len(gt_n), 1)
    
    # WER (Word Error Rate)
    wer = Levenshtein.distance(gt_words, pred_words) / max(len(gt_words), 1)
    
    # F1 word-level
    f1 = word_f1(pred_words, gt_words)
    
    # Degeneracao: ratio output/GT
    deg = len(pred) / max(len(gt), 1)
    
    return {
        "lv_ratio": round(lv_ratio, 4),
        "bleu": round(bleu, 4),
        "score": round(score, 4),
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "f1": round(f1, 4),
        "deg": round(deg, 4)
    }

BASE = "/Users/alisio/temp/ocr_results"
with open(f"{BASE}/resultados_dharma.json") as f:
    resultados = json.load(f)

# Group by modelo
by_model = defaultdict(list)
for r in resultados:
    by_model[r["modelo"]].append(r)

all_metrics = {}
for modelo, items in sorted(by_model.items()):
    print(f"\n{'='*60}")
    print(f"  {modelo}")
    print(f"{'='*60}")
    
    total = {"lv_ratio": 0, "bleu": 0, "score": 0, "cer": 0, "wer": 0, "f1": 0, "deg": 0}
    count = 0
    per_sample = {}
    
    for r in items:
        gt = r["ground_truth"]
        pred = r.get("texto", "")
        nome = r["nome"]
        m = calc_metrics(gt, pred)
        per_sample[nome] = m
        for k in total:
            total[k] += m[k]
        count += 1
        
        tipo = r.get("tipo", "?")
        status = "✓" if pred else "✗"
        print(f"  {status} {tipo:12s} {nome:25s} Score={m['score']:.4f} CER={m['cer']:.4f} WER={m['wer']:.4f} F1={m['f1']:.4f}" + 
              (f" DEG={m['deg']:.2f}" if m['deg'] > 1.5 or m['deg'] < 0.5 else ""))
    
    metrics = {}
    for k in total:
        metrics[k] = round(total[k] / count, 4) if count else 0
    all_metrics[modelo] = {"metrics": metrics, "per_sample": per_sample}
    
    score = metrics["score"]
    cer = metrics["cer"]
    wer = metrics["wer"]
    f1 = metrics["f1"]
    deg = metrics["deg"]
    print(f"\n  -> Média: Score={score:.4f} CER={cer:.4f} WER={wer:.4f} F1={f1:.4f} DEG={deg:.2f} (n={count})")

# Save
with open(f"{BASE}/metricas_dharma_resumo.json", "w") as f:
    json.dump(all_metrics, f, indent=2, ensure_ascii=False)

# Ranking
print(f"\n\n{'='*60}")
print("  RANKING GERAL (Score DharmmaOCR)")
print(f"{'='*60}")
ranking = sorted(all_metrics.items(), key=lambda x: x[1]["metrics"]["score"], reverse=True)
for i, (m, d) in enumerate(ranking, 1):
    s = d["metrics"]
    print(f"  {i}. {m:45s} Score={s['score']:.4f} CER={s['cer']:.4f} WER={s['wer']:.4f} F1={s['f1']:.4f}")
    print(f"     LV={s['lv_ratio']:.4f} BLEU={s['bleu']:.4f} DEG={s['deg']:.2f}")
