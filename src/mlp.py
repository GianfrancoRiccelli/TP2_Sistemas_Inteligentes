"""MLP con backpropagation. Modelo de referencia.

La imagen se reduce a 64x64 y se aplana a un vector de 12.288 valores.
Cada píxel entra como un atributo independiente: la red no usa la forma
bidimensional de la imagen. Sirve para comparar contra las convolucionales.

Uso (desde la raíz del repo):
    python src/mlp.py           # evalúa entrenamiento y validación
    python src/mlp.py --final   # evalúa entrenamiento y prueba, una sola vez
"""

import argparse
import sys

import tensorflow as tf

from config import CLASES
from utils import cargar_conjuntos, entrenar, evaluar, fijar_semillas

NOMBRE = "mlp"  # carpeta de salida: results/mlp
TAMANIO = (64, 64)  # 256x256 aplanado serían unas 200 mil entradas
BATCH_SIZE = 32
EPOCAS = 100  # tope; el early stopping puede cortar antes
PACIENCIA = 10  # épocas seguidas sin mejorar la loss de validación antes de parar
LEARNING_RATE = 1e-3


def construir():
    """Arma y compila el perceptrón multicapa."""
    # Misma semilla en la inicialización de los pesos, para poder repetir la corrida.
    fijar_semillas()
    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*TAMANIO, 3)),
            # De matriz (64, 64, 3) a un vector: una neurona de entrada por valor de píxel.
            tf.keras.layers.Flatten(),
            # Capas ocultas con sigmoide, como en el perceptrón multicapa visto en clase.
            tf.keras.layers.Dense(128, activation="sigmoid"),
            tf.keras.layers.Dense(64, activation="sigmoid"),
            # Una neurona por clase. Softmax deja las 11 salidas como probabilidades que suman 1.
            tf.keras.layers.Dense(len(CLASES), activation="softmax"),
        ],
        name=NOMBRE,
    )
    # Entropía cruzada: el error entre la clase real (one-hot) y la probabilidad predicha.
    # Adam aplica la corrección de ese error (backpropagation) con learning rate fijo.
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

    # Lee data/train, data/val y data/test, redimensiona a 64x64 y escala los píxeles a [0, 1].
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
