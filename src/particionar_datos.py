"""Partición estratificada 60/20/20 del dataset CCSN.

Divide las imágenes en entrenamiento, validación y prueba (con semilla fija),
las copia a data/train, data/val y data/test (una subcarpeta por clase)
y guarda las cantidades por clase en data/cantidades_por_clase.csv.

Primero se separa el conjunto de prueba y después el resto se divide en
entrenamiento y validación, como se describe en la entrega parcial.

Uso (desde la raíz del repo):
    python src/particionar_datos.py
    python src/particionar_datos.py --sobrescribir   # borra y rearma data/train, data/val y data/test
"""

import argparse
import csv
import shutil
import sys

from sklearn.model_selection import train_test_split

from config import (
    CLASES,
    CONJUNTOS,
    NOMBRES_CLASES,
    PROPORCION_TEST,
    PROPORCION_VAL,
    RUTA_DATASET,
    RUTA_DATOS,
    SEED,
)

EXTENSIONES = {".jpg", ".jpeg"}
RUTA_CSV = RUTA_DATOS / "cantidades_por_clase.csv"


def listar_imagenes():
    """Devuelve las rutas de todas las imágenes y la clase de cada una."""
    rutas, etiquetas = [], []
    for clase in CLASES:
        carpeta = RUTA_DATASET / clase
        imagenes = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES)
        rutas.extend(imagenes)
        etiquetas.extend([clase] * len(imagenes))
    return rutas, etiquetas


def particionar(rutas, etiquetas):
    """Partición estratificada en dos pasos: primero prueba, después validación."""
    resto_x, test_x, resto_y, test_y = train_test_split(
        rutas,
        etiquetas,
        test_size=PROPORCION_TEST,
        stratify=etiquetas,
        random_state=SEED,
    )
    # La validación es el 20% del total, o sea 0.2 / 0.8 = 25% de lo que queda
    train_x, val_x, train_y, val_y = train_test_split(
        resto_x,
        resto_y,
        test_size=PROPORCION_VAL / (1 - PROPORCION_TEST),
        stratify=resto_y,
        random_state=SEED,
    )
    return {
        "train": (train_x, train_y),
        "val": (val_x, val_y),
        "test": (test_x, test_y),
    }


def copiar(particion):
    for conjunto, (rutas, etiquetas) in particion.items():
        for clase in CLASES:
            (RUTA_DATOS / conjunto / clase).mkdir(parents=True, exist_ok=True)
        for ruta, clase in zip(rutas, etiquetas):
            shutil.copy2(ruta, RUTA_DATOS / conjunto / clase / ruta.name)
        print(f"Copiadas {len(rutas)} imágenes a {RUTA_DATOS / conjunto}")


def contar(particion):
    """Cantidad de imágenes por clase en cada conjunto."""
    return {
        conjunto: {c: etiquetas.count(c) for c in CLASES}
        for conjunto, (_, etiquetas) in particion.items()
    }


def guardar_csv(conteos):
    with open(RUTA_CSV, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["clase", "nombre", *CONJUNTOS, "total"])
        for clase in CLASES:
            fila = [conteos[conj][clase] for conj in CONJUNTOS]
            escritor.writerow([clase, NOMBRES_CLASES[clase], *fila, sum(fila)])
        totales = [sum(conteos[conj].values()) for conj in CONJUNTOS]
        escritor.writerow(["Total", "", *totales, sum(totales)])


def imprimir_resumen(conteos):
    print(f"\n{'Clase':<6}{'Nombre':<18}" + "".join(f"{c:>8}" for c in CONJUNTOS) + f"{'total':>8}")
    for clase in CLASES:
        fila = [conteos[conj][clase] for conj in CONJUNTOS]
        print(f"{clase:<6}{NOMBRES_CLASES[clase]:<18}" + "".join(f"{n:>8}" for n in fila) + f"{sum(fila):>8}")
    totales = [sum(conteos[conj].values()) for conj in CONJUNTOS]
    total = sum(totales)
    print(f"{'Total':<24}" + "".join(f"{n:>8}" for n in totales) + f"{total:>8}")
    print(f"{'%':<24}" + "".join(f"{100 * n / total:>7.1f}%" for n in totales))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sobrescribir",
        action="store_true",
        help="borra data/train, data/val y data/test si ya existen y los vuelve a armar",
    )
    args = parser.parse_args()

    if not RUTA_DATASET.is_dir():
        print(f"No se encontró el dataset en {RUTA_DATASET}")
        print("Revisá la ruta o definí la variable de entorno TP2_DATASET.")
        return 1

    existentes = [RUTA_DATOS / c for c in CONJUNTOS if (RUTA_DATOS / c).exists()]
    if existentes:
        if not args.sobrescribir:
            print("Ya existe una partición en:")
            for ruta in existentes:
                print(f"  - {ruta}")
            print("No se modificó nada. Para rearmarla, usá --sobrescribir.")
            return 1
        for ruta in existentes:
            shutil.rmtree(ruta)
        print("Se borró la partición anterior.")

    rutas, etiquetas = listar_imagenes()
    print(f"Dataset: {RUTA_DATASET} ({len(rutas)} imágenes)")

    particion = particionar(rutas, etiquetas)

    # Control: ninguna imagen puede quedar en dos conjuntos
    nombres = [set(f"{y}/{x.name}" for x, y in zip(*particion[c])) for c in CONJUNTOS]
    assert not (nombres[0] & nombres[1] or nombres[0] & nombres[2] or nombres[1] & nombres[2])

    copiar(particion)

    conteos = contar(particion)
    guardar_csv(conteos)
    imprimir_resumen(conteos)
    print(f"\nCantidades guardadas en {RUTA_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
