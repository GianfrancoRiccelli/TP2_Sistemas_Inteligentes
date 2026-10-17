"""Configuración compartida: rutas, clases y semilla."""

import os
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parent.parent

# El dataset se usa desde donde está, fuera del repo (../CCSN/CCSN_v2).
# Se puede cambiar con la variable de entorno TP2_DATASET (por ejemplo, en Colab).
RUTA_DATASET = Path(os.environ.get("TP2_DATASET", RAIZ_REPO.parent / "CCSN" / "CCSN_v2"))

RUTA_RESULTADOS = RAIZ_REPO / "results"

SEED = 42

TAMANIO_IMAGEN = (256, 256)

# Abreviatura de la carpeta -> nombre en español
NOMBRES_CLASES = {
    "Ac": "Altocúmulo",
    "As": "Altostrato",
    "Cb": "Cumulonimbo",
    "Cc": "Cirrocúmulo",
    "Ci": "Cirro",
    "Cs": "Cirrostrato",
    "Ct": "Estela de avión",
    "Cu": "Cúmulo",
    "Ns": "Nimbostrato",
    "Sc": "Estratocúmulo",
    "St": "Estrato",
}
CLASES = sorted(NOMBRES_CLASES)
