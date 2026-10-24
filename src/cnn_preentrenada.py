"""ConvNet preentrenada: MobileNetV2 de ImageNet, base congelada, salida de 11 clases.

Se reutilizan filtros ya entrenados para detectar bordes y texturas, y solo se
entrena una capa nueva de 11 clases. La primera corrida descarga los pesos de ImageNet.

Uso (desde la raíz del repo):
    python src/cnn_preentrenada.py           # evalúa entrenamiento y validación
    python src/cnn_preentrenada.py --final   # evalúa entrenamiento y prueba, una sola vez
"""

import argparse
import sys

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from config import CLASES
from utils import cargar_conjuntos, entrenar, evaluar, fijar_semillas

NOMBRE = "cnn_preentrenada"  # carpeta de salida: results/cnn_preentrenada
TAMANIO = (224, 224)  # tamaño de entrada con el que se entrenó MobileNetV2
BATCH_SIZE = 32
EPOCAS = 30  # solo se entrena el cabezal, así que converge antes que una red desde cero
PACIENCIA = 5  # épocas seguidas sin mejorar la loss de validación antes de parar
LEARNING_RATE = 1e-3


def construir():
    """Arma la red: base de ImageNet congelada y una salida nueva de 11 clases."""
    fijar_semillas()
    # include_top=False saca la clasificación original de 1000 clases de ImageNet.
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(*TAMANIO, 3))
    # Congela los filtros. El entrenamiento ajusta únicamente las capas que se agregan abajo.
    base.trainable = False
    modelo = tf.keras.Sequential(
        [
            base,
            # Resume cada mapa de filtros en un número. El cabezal queda chico.
            tf.keras.layers.GlobalAveragePooling2D(),
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

    # preprocess_input escala a [-1, 1], que es lo que espera MobileNetV2.
    # No es la división por 255 que usan el MLP y la ConvNet desde cero.
    conjuntos = cargar_conjuntos(
        tamanio=TAMANIO,
        batch_size=BATCH_SIZE,
        preprocesar=preprocess_input,
    )
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
            "backbone": "MobileNetV2 ImageNet, base congelada",
            "evaluación": "/".join(conjuntos_eval),
        },
    )
    # Gráfico de loss, matriz de confusión y precisión y recall por clase.
    evaluar(modelo, conjuntos, historial, NOMBRE, conjuntos_eval=conjuntos_eval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
