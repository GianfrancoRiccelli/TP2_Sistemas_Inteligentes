"""ConvNet desde cero. Entrada 256x256x3.

Cuatro bloques de convolución, ReLU y pooling recorren la imagen y arman
mapas de características. Después un aplanado y una capa densa clasifican
en las 11 clases. Los pesos arrancan al azar: no se reutiliza otra red.

Uso (desde la raíz del repo):
    python src/cnn.py           # evalúa entrenamiento y validación
    python src/cnn.py --final   # evalúa entrenamiento y prueba, una sola vez
"""

import argparse
import sys

import tensorflow as tf

from config import CLASES
from utils import cargar_conjuntos, entrenar, evaluar, fijar_semillas

NOMBRE = "cnn_desde_cero"  # carpeta de salida: results/cnn_desde_cero
TAMANIO = (256, 256)
BATCH_SIZE = 32
EPOCAS = 100  # tope; el early stopping puede cortar antes
PACIENCIA = 10  # épocas seguidas sin mejorar la loss de validación antes de parar
LEARNING_RATE = 1e-3


def bloque(filtros):
    """Convolución 3x3 con ReLU y pooling que reduce alto y ancho a la mitad."""
    return [
        # padding same: la convolución no achica la imagen. El achique lo hace el pooling.
        tf.keras.layers.Conv2D(filtros, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
    ]


def construir():
    """Arma y compila la red convolucional."""
    fijar_semillas()
    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*TAMANIO, 3)),
            # 256 -> 128 -> 64 -> 32 -> 16. El último mapa queda de 16x16x128.
            *bloque(32),
            *bloque(64),
            *bloque(128),
            *bloque(128),
            # Pasa ese mapa a un vector para poder usar capas densas.
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            # Apaga la mitad de las neuronas al entrenar: el vector aplanado tiene
            # millones de pesos para solo 1.525 imágenes.
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(CLASES), activation="softmax"),
        ],
        name=NOMBRE,
    )
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--final",
        action="store_true",
        help="evalúa entrenamiento y prueba; sin este flag, entrenamiento y validación",
    )
    args = parser.parse_args()
    # Sin --final no se mira la prueba: las decisiones de diseño se toman con validación.
    conjuntos_eval = ("train", "test") if args.final else ("train", "val")

    # Lee las tres carpetas, redimensiona a 256x256 y escala los píxeles a [0, 1].
    conjuntos = cargar_conjuntos(tamanio=TAMANIO, batch_size=BATCH_SIZE)
    modelo = construir()
    # Entrena con early stopping sobre la loss de validación y guarda topología e historial.
    historial = entrenar(
        modelo,
        conjuntos,
        NOMBRE,
        epocas=EPOCAS,
        paciencia=PACIENCIA,
        parametros={
            "tamaño de imagen": f"{TAMANIO[0]}x{TAMANIO[1]}",
            "batch size": BATCH_SIZE,
            "evaluación": "/".join(conjuntos_eval),
        },
    )
    # Gráfico de loss, matriz de confusión y precisión y recall por clase.
    evaluar(modelo, conjuntos, historial, NOMBRE, conjuntos_eval=conjuntos_eval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
