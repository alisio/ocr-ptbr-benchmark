"""Extrai 10 amostras do DharmaOCR-Benchmark para teste de OCR."""
import requests, io, base64, os, json
from PIL import Image
import pandas as pd

BASE = "/Users/alisio/temp/ocr_results/amostras_dharma"
os.makedirs(f"{BASE}/manuscrito", exist_ok=True)
os.makedirs(f"{BASE}/impresso", exist_ok=True)

# Rows to extract: (file_number, row_index, tipo, nome)
AMOSTRAS = [
    (8, 3, "manuscrito", "corcunda_notredame"),
    (8, 9, "manuscrito", "filme_coringa"),
    (8, 14, "manuscrito", "filme_beleza_ocultar"),
    (11, 0, "manuscrito", "serie_13_reasons"),
    (11, 25, "manuscrito", "filme_coringa_2"),
    (0, 0, "impresso", "luis_x_rei"),
    (0, 3, "impresso", "anatomia"),
    (8, 13, "impresso", "barranquilla"),
    (8, 33, "impresso", "cairo"),
    (0, 41, "impresso", "doutrina_nacional"),
]

def load_file(fnum):
    url = f"https://huggingface.co/datasets/Dharma-AI/DharmaOCR-Benchmark/resolve/main/data/test-{fnum:05d}-of-00012.parquet"
    print(f"  Baixando file {fnum}...")
    r = requests.get(url, timeout=300)
    return pd.read_parquet(io.BytesIO(r.content))

# Group downloads by file
files_needed = set(f for f, _, _, _ in AMOSTRAS)
dataframes = {}
for fnum in sorted(files_needed):
    dataframes[fnum] = load_file(fnum)

# Extract each sample
manifest = []
for fnum, row_idx, tipo, nome in AMOSTRAS:
    df = dataframes[fnum]
    row = df.iloc[row_idx]
    b64 = row['image_base64']
    gt = row['assistant_without_json']
    
    img_data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_data))
    
    # Save image
    ext = "png" if b64.startswith("iVBOR") else "jpg"
    img_path = f"{BASE}/{tipo}/{nome}.{ext}"
    img.save(img_path)
    
    # Save ground truth
    gt_path = f"{BASE}/{tipo}/{nome}.txt"
    with open(gt_path, "w") as f:
        f.write(gt.strip())
    
    manifest.append({
        "id": fnum * 100 + row_idx,
        "nome": nome,
        "tipo": tipo,
        "imagem": f"{tipo}/{nome}.{ext}",
        "ground_truth": gt.strip(),
        "dimensoes": f"{img.size[0]}x{img.size[1]}",
        "mode": img.mode,
        "chars": len(gt.strip()),
        "palavras": len(gt.strip().split()),
    })
    print(f"  OK: {tipo}/{nome}.{ext} ({img.size[0]}x{img.size[1]}, {len(gt.strip())} chars)")

with open(f"{BASE}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"\nTotal: {len(manifest)} amostras extraidas em {BASE}/")
