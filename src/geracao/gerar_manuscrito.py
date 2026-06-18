# Autor: Alisio (https://github.com/alisio)
# Licenca: MIT

from PIL import Image, ImageDraw, ImageFont
import os
import argparse

GROUND_TRUTH = """Lista de Compras

Hoje fui ao mercado e comprei:
- 2 kg de arroz
- 1 lata de oleo
- 3 pacotes de macarrao
- 1 kg de acucar
- 2 litros de leite

Total aproximado: R$ 85,00

Nao esquecer de pagar a conta de luz!
Ass: Joao"""

LARGURA = 700
ALTURA = 800
COR_FUNDO = (255, 255, 252)
COR_TEXTO = (30, 25, 20)

def gerar(output_dir):
    img = Image.new('RGB', (LARGURA, ALTURA), COR_FUNDO)
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Apple Chancery.ttf", 36)
        font_body1 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf", 22)
        font_body2 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Brush Script.ttf", 24)
        font_sig = ImageFont.truetype("/System/Library/Fonts/Supplemental/SnellRoundhand.ttc", 26)
    except:
        font_title = font_body1 = font_body2 = font_sig = ImageFont.load_default()

    y = 40

    def desenha_linha(texto, font, y, cor=COR_TEXTO):
        draw.text((50, y), texto, fill=cor, font=font)
        bbox = draw.textbbox((50, y), texto, font=font)
        espaco = int(font.size * 1.1)
        return y + espaco

    y = desenha_linha("Lista de Compras", font_title, y)
    y += 10

    y = desenha_linha("Hoje fui ao mercado e comprei:", font_body2, y)
    y += 5

    y = desenha_linha("- 2 kg de arroz", font_body1, y)
    y = desenha_linha("- 1 lata de oleo", font_body1, y)
    y = desenha_linha("- 3 pacotes de macarrao", font_body1, y)
    y = desenha_linha("- 1 kg de acucar", font_body1, y)
    y = desenha_linha("- 2 litros de leite", font_body1, y)
    y += 10

    y = desenha_linha("Total aproximado: R$ 85,00", font_body2, y)
    y += 15

    y = desenha_linha("Nao esquecer de pagar a conta de luz!", font_body1, y)
    y += 5

    y = desenha_linha("Ass: Joao", font_sig, y)

    os.makedirs(output_dir, exist_ok=True)
    img.save(os.path.join(output_dir, "documento_manuscrito.png"))
    with open(os.path.join(output_dir, "ground_truth_manuscrito.txt"), "w") as f:
        f.write(GROUND_TRUTH)

    print(f"Imagem manuscrita gerada em {output_dir}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerar imagem manuscrita de teste OCR")
    parser.add_argument("--output", "-o", default="assets", help="Diretório de saída")
    args = parser.parse_args()
    gerar(args.output)