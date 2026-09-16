#!/usr/bin/env python3
"""
Define o avatar da BatataAI a partir de uma imagem PNG.

A imagem é recortada em quadrado (centralizada, com um leve viés para cima
para pegar o rosto), redimensionada, otimizada e embutida em web/index.html
como data URI. Isso mantém a BatataAI portátil: continua sendo um único
arquivo HTML, sem rota nova no backend e sem depender de internet.

Uso:
    python tools/definir-avatar.py web/cat.png
    python tools/definir-avatar.py web/cat.png --tamanho 192 --foco 0.42
    python tools/definir-avatar.py --remover

Só depende da biblioteca padrão (zlib + base64).
"""

import argparse
import base64
import struct
import sys
import zlib
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
INDEX = RAIZ / "web" / "index.html"

INICIO = "/* BATATAAI-AVATAR-START */"
FIM = "/* BATATAAI-AVATAR-END */"

SEM_AVATAR = f"{INICIO}\n/* Nenhum avatar definido ainda. */\n{FIM}"


# -------------------------------------------------- leitura de PNG

def ler_png(caminho: Path):
    """Devolve (largura, altura, pixels RGB como bytearray)."""
    dados = caminho.read_bytes()
    if dados[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{caminho.name} não é um PNG.")

    largura = altura = None
    profundidade = tipo_cor = interlace = None
    idat = bytearray()
    paleta = None
    pos = 8

    while pos < len(dados):
        (tamanho,) = struct.unpack(">I", dados[pos:pos + 4])
        tipo = dados[pos + 4:pos + 8]
        corpo = dados[pos + 8:pos + 8 + tamanho]
        pos += 12 + tamanho  # 4 tamanho + 4 tipo + corpo + 4 CRC

        if tipo == b"IHDR":
            largura, altura, profundidade, tipo_cor, _, _, interlace = struct.unpack(
                ">IIBBBBB", corpo
            )
        elif tipo == b"PLTE":
            paleta = corpo
        elif tipo == b"IDAT":
            idat += corpo
        elif tipo == b"IEND":
            break

    if largura is None:
        raise ValueError("PNG sem cabeçalho IHDR.")
    if profundidade != 8:
        raise ValueError(
            f"Só sei ler PNG de 8 bits por canal (este tem {profundidade}). "
            "Reexporte a imagem como PNG de 8 bits."
        )
    if interlace:
        raise ValueError("PNG entrelaçado (Adam7) não é suportado. Reexporte sem entrelaçamento.")

    canais = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(tipo_cor)
    if canais is None:
        raise ValueError(f"Tipo de cor PNG não suportado: {tipo_cor}")
    if tipo_cor == 3 and paleta is None:
        raise ValueError("PNG com paleta sem bloco PLTE.")

    bruto = zlib.decompress(bytes(idat))
    bpp = canais
    largura_linha = largura * bpp

    # Desfaz os filtros linha a linha.
    saida = bytearray(altura * largura_linha)
    anterior = bytearray(largura_linha)
    origem = 0

    for y in range(altura):
        filtro = bruto[origem]
        origem += 1
        linha = bytearray(bruto[origem:origem + largura_linha])
        origem += largura_linha

        if filtro == 1:      # Sub
            for i in range(bpp, largura_linha):
                linha[i] = (linha[i] + linha[i - bpp]) & 0xFF
        elif filtro == 2:    # Up
            for i in range(largura_linha):
                linha[i] = (linha[i] + anterior[i]) & 0xFF
        elif filtro == 3:    # Average
            for i in range(largura_linha):
                esq = linha[i - bpp] if i >= bpp else 0
                linha[i] = (linha[i] + ((esq + anterior[i]) >> 1)) & 0xFF
        elif filtro == 4:    # Paeth
            for i in range(largura_linha):
                a = linha[i - bpp] if i >= bpp else 0
                b = anterior[i]
                c = anterior[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                linha[i] = (linha[i] + pr) & 0xFF
        elif filtro != 0:
            raise ValueError(f"Filtro PNG desconhecido: {filtro}")

        saida[y * largura_linha:(y + 1) * largura_linha] = linha
        anterior = linha

    # Normaliza tudo para RGB, achatando transparência sobre branco.
    rgb = bytearray(largura * altura * 3)
    for i in range(largura * altura):
        base = i * bpp
        if tipo_cor == 0:                     # cinza
            v = saida[base]
            r = g = b = v
        elif tipo_cor == 4:                   # cinza + alfa
            v, a = saida[base], saida[base + 1]
            v = (v * a + 255 * (255 - a)) // 255
            r = g = b = v
        elif tipo_cor == 2:                   # RGB
            r, g, b = saida[base], saida[base + 1], saida[base + 2]
        elif tipo_cor == 3:                   # paleta
            idx = saida[base] * 3
            r, g, b = paleta[idx], paleta[idx + 1], paleta[idx + 2]
        else:                                 # RGBA
            r, g, b, a = saida[base:base + 4]
            if a != 255:
                r = (r * a + 255 * (255 - a)) // 255
                g = (g * a + 255 * (255 - a)) // 255
                b = (b * a + 255 * (255 - a)) // 255
        rgb[i * 3:i * 3 + 3] = bytes((r, g, b))

    return largura, altura, rgb


# ------------------------------------------- recorte e redimensionamento

def recortar_quadrado(largura, altura, rgb, foco_x=0.5, foco_y=0.42, zoom=1.0):
    """
    Recorta um quadrado da imagem sem esticar nada.

    zoom=1.0 pega o maior quadrado possível; valores menores aproximam.
    foco_x/foco_y (0..1) dizem em que ponto da imagem o quadrado é centrado.
    """
    lado = max(8, min(round(min(largura, altura) * zoom), min(largura, altura)))

    x0 = round(largura * foco_x - lado / 2)
    y0 = round(altura * foco_y - lado / 2)
    x0 = max(0, min(largura - lado, x0))
    y0 = max(0, min(altura - lado, y0))

    if lado == largura and lado == altura and x0 == 0 and y0 == 0:
        return lado, rgb

    corte = bytearray(lado * lado * 3)
    for y in range(lado):
        origem = ((y0 + y) * largura + x0) * 3
        corte[y * lado * 3:(y + 1) * lado * 3] = rgb[origem:origem + lado * 3]
    return lado, corte


def redimensionar(lado, rgb, destino):
    """Redução por média de área (box filter). Não distorce: entra e sai quadrado."""
    if destino >= lado:
        return lado, rgb

    saida = bytearray(destino * destino * 3)
    escala = lado / destino

    for ty in range(destino):
        y0 = int(ty * escala)
        y1 = max(y0 + 1, int((ty + 1) * escala))
        for tx in range(destino):
            x0 = int(tx * escala)
            x1 = max(x0 + 1, int((tx + 1) * escala))

            sr = sg = sb = n = 0
            for y in range(y0, y1):
                base = (y * lado + x0) * 3
                for _ in range(x1 - x0):
                    sr += rgb[base]
                    sg += rgb[base + 1]
                    sb += rgb[base + 2]
                    base += 3
                    n += 1

            i = (ty * destino + tx) * 3
            saida[i] = sr // n
            saida[i + 1] = sg // n
            saida[i + 2] = sb // n

    return destino, saida


# -------------------------------------------------- escrita de PNG

def _bloco(tipo: bytes, corpo: bytes) -> bytes:
    return (struct.pack(">I", len(corpo)) + tipo + corpo
            + struct.pack(">I", zlib.crc32(tipo + corpo) & 0xFFFFFFFF))


def escrever_png(lado, rgb) -> bytes:
    """Codifica RGB de 8 bits com filtro Paeth, que comprime bem foto."""
    largura_linha = lado * 3
    bruto = bytearray()
    anterior = bytearray(largura_linha)

    for y in range(lado):
        linha = rgb[y * largura_linha:(y + 1) * largura_linha]
        bruto.append(4)  # Paeth
        codificada = bytearray(largura_linha)
        for i in range(largura_linha):
            a = linha[i - 3] if i >= 3 else 0
            b = anterior[i]
            c = anterior[i - 3] if i >= 3 else 0
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            codificada[i] = (linha[i] - pr) & 0xFF
        bruto += codificada
        anterior = linha

    ihdr = struct.pack(">IIBBBBB", lado, lado, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + _bloco(b"IHDR", ihdr)
            + _bloco(b"IDAT", zlib.compress(bytes(bruto), 9))
            + _bloco(b"IEND", b""))


# -------------------------------------------------- injeção no HTML

def _trocar_bloco(html: str, novo: str) -> str:
    i = html.index(INICIO)
    f = html.index(FIM) + len(FIM)
    return html[:i] + novo + html[f:]


def _trocar_favicon(html: str, href: str) -> str:
    """
    Troca a tag do favicon.

    O fim da tag precisa ser procurado ignorando o que está entre aspas: o
    favicon antigo era um SVG embutido e continha ">" dentro do href, o que
    faria um index(">") simples cortar a tag no lugar errado.
    """
    i = html.index('<link rel="icon"')

    aspas = None
    j = i
    while j < len(html):
        c = html[j]
        if aspas:
            if c == aspas:
                aspas = None
        elif c in ('"', "'"):
            aspas = c
        elif c == ">":
            break
        j += 1
    else:
        raise ValueError("Tag <link rel='icon'> sem fechamento.")

    return html[:i] + f'<link rel="icon" href="{href}">' + html[j + 1:]


def main():
    ap = argparse.ArgumentParser(description="Define o avatar da BatataAI.")
    ap.add_argument("imagem", nargs="?", help="PNG de origem (ex.: web/cat.png)")
    ap.add_argument("--tamanho", type=int, default=192,
                    help="lado do avatar em pixels (padrão: 192)")
    ap.add_argument("--aplicar-favicon", action="store_true",
                    help="também usa a imagem no favicon (por padrão o favicon "
                         "continua sendo o emoji de batata da marca)")
    ap.add_argument("--favicon", type=int, default=64,
                    help="lado do favicon em pixels, se --aplicar-favicon (padrão: 64)")
    ap.add_argument("--foco-x", type=float, default=0.50,
                    help="ponto focal horizontal, 0=esquerda 1=direita (padrão: 0.50)")
    ap.add_argument("--foco-y", type=float, default=0.46,
                    help="ponto focal vertical, 0=topo 1=base (padrão: 0.46)")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="fração da imagem usada no recorte; menor = mais perto (padrão: 1.0)")
    ap.add_argument("--enquadramento", type=float, default=118,
                    help="zoom do CSS dentro do círculo, em porcento (padrão: 118)")
    ap.add_argument("--pos-x", type=float, default=52,
                    help="posição horizontal do enquadramento, em porcento (padrão: 52)")
    ap.add_argument("--pos-y", type=float, default=44,
                    help="posição vertical do enquadramento, em porcento (padrão: 44)")
    ap.add_argument("--remover", action="store_true",
                    help="tira o avatar; a interface volta a mostrar a letra 'B'")
    args = ap.parse_args()

    html = INDEX.read_text(encoding="utf-8")
    if INICIO not in html or FIM not in html:
        sys.exit("Marcadores do avatar não encontrados em web/index.html.")

    if args.remover:
        # O favicon não é mexido: ele pertence à marca (batata), não ao avatar.
        html = _trocar_bloco(html, SEM_AVATAR)
        INDEX.write_text(html, encoding="utf-8")
        print("Avatar removido; a letra 'B' voltou. O favicon não foi alterado.")
        return

    if not args.imagem:
        ap.error("informe a imagem, ou use --remover")

    origem = Path(args.imagem)
    if not origem.is_absolute():
        origem = RAIZ / origem
    if not origem.exists():
        sys.exit(f"Imagem não encontrada: {origem}")

    largura, altura, rgb = ler_png(origem)
    print(f"origem: {largura}x{altura}")

    lado, quadrado = recortar_quadrado(
        largura, altura, rgb, args.foco_x, args.foco_y, args.zoom
    )
    print(f"recorte: {lado}x{lado} "
          f"(zoom {args.zoom}, foco {args.foco_x}/{args.foco_y})")

    tarefas = [("avatar", args.tamanho)]
    if args.aplicar_favicon:
        tarefas.append(("favicon", args.favicon))

    saidas = {}
    for nome, destino in tarefas:
        n, px = redimensionar(lado, quadrado, destino)
        png = escrever_png(n, px)
        saidas[nome] = png
        print(f"{nome}: {n}x{n}  {len(png) / 1024:.1f} KB")

    b64 = base64.b64encode(saidas["avatar"]).decode()
    bloco = (
        f"{INICIO}\n"
        f"/* Avatar gerado a partir de {origem.name} "
        f"({args.tamanho}x{args.tamanho}). Regerar: "
        f"python tools/definir-avatar.py {args.imagem} */\n"
        ":root{\n"
        f'  --avatar-image:url("data:image/png;base64,{b64}");\n'
        "  --avatar-text-color:transparent;\n"
        f"  --avatar-zoom:{args.enquadramento}%;\n"
        f"  --avatar-position:{args.pos_x}% {args.pos_y}%;\n"
        "}\n"
        f"{FIM}"
    )

    html = _trocar_bloco(html, bloco)

    onde = "no card de boas-vindas e ao lado de cada resposta"
    if args.aplicar_favicon:
        html = _trocar_favicon(
            html,
            "data:image/png;base64," + base64.b64encode(saidas["favicon"]).decode(),
        )
        onde += " e no favicon"

    INDEX.write_text(html, encoding="utf-8")

    print(f"web/index.html: {len(html) / 1024:.0f} KB")
    print(f"Avatar aplicado {onde}.")


if __name__ == "__main__":
    main()
