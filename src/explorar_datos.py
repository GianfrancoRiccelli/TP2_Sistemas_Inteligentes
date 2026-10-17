"""Exploración del dataset CCSN.

Cuenta las imágenes por clase, verifica que todas sean 256x256 RGB
y grafica la distribución de clases en results/exploracion/.

Uso (desde la raíz del repo):
    python src/explorar_datos.py
"""

import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # solo se guarda la figura, no se abre ventana
import matplotlib.pyplot as plt
from PIL import Image

from config import CLASES, NOMBRES_CLASES, RUTA_DATASET, RUTA_RESULTADOS, TAMANIO_IMAGEN

EXTENSIONES = {".jpg", ".jpeg", ".png"}

# Colores de la figura (una por resolución, en orden fijo)
COLORES_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
COLOR_TEXTO = "#0b0b0b"
COLOR_TEXTO_SECUNDARIO = "#52514e"
COLOR_GRILLA = "#e4e3df"
COLOR_FONDO = "#fcfcfb"


def describir_formato(tamanio, modo):
    return f"{tamanio[0]}x{tamanio[1]} {modo}"


def verificar_dataset():
    """Recorre cada carpeta de clase.

    Devuelve, por clase, un Counter de formatos ("400x400 RGB" -> cantidad)
    y una lista de problemas (archivos que no se pueden leer, carpetas de más o de menos).
    """
    formatos = {}
    problemas = []

    carpetas = {p.name for p in RUTA_DATASET.iterdir() if p.is_dir()}
    for extra in sorted(carpetas - set(CLASES)):
        problemas.append(f"Carpeta inesperada: {extra}")

    for clase in CLASES:
        formatos[clase] = Counter()
        carpeta = RUTA_DATASET / clase
        if not carpeta.is_dir():
            problemas.append(f"Falta la carpeta de la clase {clase}")
            continue

        for ruta in sorted(p for p in carpeta.iterdir() if p.is_file()):
            if ruta.suffix.lower() not in EXTENSIONES:
                problemas.append(f"Archivo que no es imagen: {ruta.relative_to(RUTA_DATASET)}")
                continue
            try:
                with Image.open(ruta) as img:
                    img.verify()  # detecta archivos dañados
                with Image.open(ruta) as img:
                    formatos[clase][describir_formato(img.size, img.mode)] += 1
            except Exception as e:
                problemas.append(f"No se pudo abrir {ruta.relative_to(RUTA_DATASET)}: {e}")

    return formatos, problemas


def imprimir_resumen(formatos, problemas):
    esperado = describir_formato(TAMANIO_IMAGEN, "RGB")
    conteos = {c: sum(formatos[c].values()) for c in CLASES}
    total = sum(conteos.values())
    todos = sorted(set().union(*formatos.values()))

    print(f"Dataset: {RUTA_DATASET}\n")
    encabezado = f"{'Clase':<6}{'Nombre':<18}{'Imágenes':>9}{'%':>8}"
    encabezado += "".join(f"{f:>14}" for f in todos)
    print(encabezado)
    for clase in CLASES:
        n = conteos[clase]
        fila = f"{clase:<6}{NOMBRES_CLASES[clase]:<18}{n:>9}{100 * n / total:>7.1f}%"
        fila += "".join(f"{formatos[clase][f]:>14}" for f in todos)
        print(fila)
    fila_total = f"{'Total':<24}{total:>9}{'':>8}"
    fila_total += "".join(f"{sum(formatos[c][f] for c in CLASES):>14}" for f in todos)
    print(fila_total + "\n")

    mayor = max(conteos, key=conteos.get)
    menor = min(conteos, key=conteos.get)
    print(
        f"Clase mayoritaria: {mayor} ({conteos[mayor]}), minoritaria: {menor} ({conteos[menor]}), "
        f"relación {conteos[mayor] / conteos[menor]:.2f}\n"
    )

    fuera = {c: conteos[c] - formatos[c][esperado] for c in CLASES}
    n_fuera = sum(fuera.values())
    if n_fuera:
        print(f"Verificación {esperado}: NO. {n_fuera} de {total} imágenes tienen otro formato.")
        for clase in CLASES:
            if fuera[clase]:
                print(f"  - {clase}: {fuera[clase]} de {conteos[clase]}")
    else:
        print(f"Verificación {esperado}: OK, todas las imágenes tienen ese formato.")

    if problemas:
        print(f"\nSe encontraron {len(problemas)} problemas:")
        for p in problemas:
            print(f"  - {p}")

    return n_fuera


def graficar_distribucion(formatos, ruta_salida):
    # Barras horizontales apiladas por formato, ordenadas de mayor a menor
    conteos = {c: sum(formatos[c].values()) for c in CLASES}
    orden = sorted(CLASES, key=lambda c: conteos[c])
    etiquetas = [f"{NOMBRES_CLASES[c]} ({c})" for c in orden]
    total = sum(conteos.values())
    # El formato más frecuente va primero (pegado al eje)
    series = sorted(
        set().union(*formatos.values()),
        key=lambda f: -sum(formatos[c][f] for c in CLASES),
    )

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=COLOR_FONDO)
    ax.set_facecolor(COLOR_FONDO)

    izquierda = [0] * len(orden)
    for formato, color in zip(series, COLORES_SERIES):
        valores = [formatos[c][formato] for c in orden]
        ax.barh(
            etiquetas,
            valores,
            left=izquierda,
            height=0.7,
            color=color,
            edgecolor=COLOR_FONDO,  # separación de 2px entre segmentos
            linewidth=1,
            label=formato,
            zorder=2,
        )
        izquierda = [i + v for i, v in zip(izquierda, valores)]

    for y, clase in enumerate(orden):
        n = conteos[clase]
        ax.text(
            n + 4,
            y,
            f"{n}  ({100 * n / total:.1f}%)",
            va="center",
            fontsize=9,
            color=COLOR_TEXTO_SECUNDARIO,
        )

    ax.set_title(
        f"Distribución de clases en CCSN ({total} imágenes)",
        loc="left",
        fontsize=12,
        color=COLOR_TEXTO,
        pad=12,
    )
    ax.set_xlabel("Cantidad de imágenes", color=COLOR_TEXTO_SECUNDARIO)
    ax.set_xlim(0, max(conteos.values()) * 1.2)
    ax.grid(axis="x", color=COLOR_GRILLA, linewidth=0.8, zorder=0)
    ax.tick_params(colors=COLOR_TEXTO_SECUNDARIO, length=0)
    for lado in ("top", "right", "bottom"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color(COLOR_GRILLA)
    if len(series) > 1:
        leyenda = ax.legend(title="Resolución", loc="lower right", frameon=False, fontsize=9)
        leyenda.get_title().set_color(COLOR_TEXTO_SECUNDARIO)
        for texto in leyenda.get_texts():
            texto.set_color(COLOR_TEXTO_SECUNDARIO)

    fig.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=150, facecolor=COLOR_FONDO)
    plt.close(fig)


def main():
    if not RUTA_DATASET.is_dir():
        print(f"No se encontró el dataset en {RUTA_DATASET}")
        print("Revisá la ruta o definí la variable de entorno TP2_DATASET.")
        return 1

    formatos, problemas = verificar_dataset()
    n_fuera = imprimir_resumen(formatos, problemas)

    ruta_figura = RUTA_RESULTADOS / "exploracion" / "distribucion_clases.png"
    graficar_distribucion(formatos, ruta_figura)
    print(f"\nGráfico guardado en {ruta_figura}")

    # Código de salida distinto de 0 si la verificación no pasa
    return 1 if (problemas or n_fuera) else 0


if __name__ == "__main__":
    sys.exit(main())
