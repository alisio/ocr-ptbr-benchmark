# Autor: Alisio (https://github.com/alisio)
# Licenca: MIT

from PIL import Image, ImageDraw, ImageFont
import os
import argparse

GROUND_TRUTH = """Relatório de Vendas - Primeiro Trimestre 2026

Análise de Desempenho Comercial

O primeiro trimestre de 2026 apresentou resultados expressivos para a empresa. As vendas totais alcançaram R$ 2.450.000,00, superando a meta estabelecida em 15%. O crescimento foi impulsionado principalmente pelo lançamento de novos produtos e pela expansão da equipe comercial.

Os principais fatores que contribuíram para este resultado foram:

1. Campanha de marketing digital com foco em redes sociais.
2. Treinamento intensivo da equipe de vendas em janeiro.
3. Parcerias estratégicas com três novos distribuidores regionais.

A tabela abaixo resume o desempenho por categoria:

Categoria     | Meta (R$) | Realizado (R$)
Eletrônicos   | 800.000   | 920.000
Vestuário     | 500.000   | 580.000
Alimentos     | 300.000   | 450.000"""

LARGURA = 900
ALTURA = 1200
COR_FUNDO = (255, 255, 255)
COR_TEXTO = (0, 0, 0)

def gerar(output_dir):
    img = Image.new('RGB', (LARGURA, ALTURA), COR_FUNDO)
    draw = ImageDraw.Draw(img)

    try:
        font_titulo = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_corpo = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font_titulo = ImageFont.load_default()
        font_corpo = ImageFont.load_default()

    y = 40

    def desenha(texto, font, y, cor=COR_TEXTO, espaco_extra=0):
        draw.text((50, y), texto, fill=cor, font=font)
        bbox = draw.textbbox((50, y), texto, font=font)
        return y + (bbox[3] - bbox[1]) + espaco_extra

    y = desenha("Relatório de Vendas - Primeiro Trimestre 2026", font_titulo, y, espaco_extra=35)
    y = desenha("Análise de Desempenho Comercial", font_corpo, y, espaco_extra=25)

    paragrafos = [
        "O primeiro trimestre de 2026 apresentou resultados expressivos para a empresa. As vendas totais alcançaram R$ 2.450.000,00, superando a meta estabelecida em 15%. O crescimento foi impulsionado principalmente pelo lançamento de novos produtos e pela expansão da equipe comercial.",
        "Os principais fatores que contribuíram para este resultado foram:",
    ]

    for p in paragrafos:
        palavras = p.split()
        linha = ""
        for palavra in palavras:
            teste = linha + " " + palavra if linha else palavra
            bbox = draw.textbbox((50, y), teste, font=font_corpo)
            if bbox[2] - bbox[0] > LARGURA - 100:
                y = desenha(linha, font_corpo, y, espaco_extra=4)
                linha = palavra
            else:
                linha = teste
        if linha:
            y = desenha(linha, font_corpo, y, espaco_extra=12)

    y = desenha("1. Campanha de marketing digital com foco em redes sociais.", font_corpo, y, espaco_extra=4)
    y = desenha("2. Treinamento intensivo da equipe de vendas em janeiro.", font_corpo, y, espaco_extra=4)
    y = desenha("3. Parcerias estratégicas com três novos distribuidores regionais.", font_corpo, y, espaco_extra=20)

    y = desenha("A tabela abaixo resume o desempenho por categoria:", font_corpo, y, espaco_extra=15)

    cabecalho = ["Categoria", "Meta (R$)", "Realizado (R$)"]
    linhas_tabela = [
        ["Eletrônicos", "800.000", "920.000"],
        ["Vestuário", "500.000", "580.000"],
        ["Alimentos", "300.000", "450.000"],
    ]

    col_x = [50, 350, 580]
    col_w = [280, 200, 200]

    for i, item in enumerate(cabecalho):
        draw.text((col_x[i], y), item, fill=(0, 0, 0), font=font_titulo)
    y += 35

    for linha in linhas_tabela:
        for i, item in enumerate(linha):
            draw.text((col_x[i], y), item, fill=(0, 0, 0), font=font_corpo)
        y += 30
        for i in range(len(col_x)):
            draw.line([(col_x[i], y-5), (col_x[i]+col_w[i], y-5)], fill=(200,200,200), width=1)
        y += 5

    os.makedirs(output_dir, exist_ok=True)
    img.save(os.path.join(output_dir, "documento_teste.png"))
    with open(os.path.join(output_dir, "ground_truth.txt"), "w") as f:
        f.write(GROUND_TRUTH)

    print(f"Imagem e ground truth gerados em {output_dir}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerar imagem de teste OCR")
    parser.add_argument("--output", "-o", default="assets", help="Diretório de saída")
    args = parser.parse_args()
    gerar(args.output)