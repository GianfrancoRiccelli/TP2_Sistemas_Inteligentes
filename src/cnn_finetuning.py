"""ConvNet preentrenada con ajuste fino y aumento de datos.

Parte de la misma MobileNetV2 de ImageNet que cnn_preentrenada.py, con dos cambios
para atacar el sobreajuste que se vio en esa corrida (72 % en entrenamiento contra
40 % en prueba):

1. Aumento de datos: en cada época las fotos de entrenamiento se espejan, rotan,
   acercan y cambian de contraste al azar, así la red no ve dos veces la misma imagen.
   Estas capas solo actúan al entrenar; en la evaluación las imágenes pasan sin cambios.
2. Ajuste fino: además de la capa de salida, se entrenan las últimas capas de
   MobileNetV2 con un learning rate bajo, para adaptar sus filtros a texturas de nubes.

Uso (desde la raíz del repo):
    python src/cnn_finetuning.py           # evalúa entrenamiento y validación
    python src/cnn_finetuning.py --final   # evalúa entrenamiento y prueba, una sola vez
"""

import argparse
import sys

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from config import CLASES, SEED
from utils import cargar_conjuntos, entrenar, evaluar, fijar_semillas

NOMBRE = "cnn_finetuning"  # carpeta de salida: results/cnn_finetuning
TAMANIO = (224, 224)  # tamaño de entrada con el que se entrenó MobileNetV2
BATCH_SIZE = 32
EPOCAS = 50
PACIENCIA = 8
# Bajo a propósito: más alto rompería los filtros que ya trae la red de ImageNet.
LEARNING_RATE = 1e-4
CAPAS_ENTRENABLES = 40  # últimas capas de MobileNetV2 que se descongelan


def aumento_de_datos():
    """Transformaciones al azar que solo se aplican durante el entrenamiento.

    Solo espejado horizontal: dar vuelta una foto de cielo de arriba abajo
    deja el horizonte arriba, algo que no pasa en fotos reales.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.05, seed=SEED),  # hasta unos 18 grados
            tf.keras.layers.RandomZoom(0.15, seed=SEED),
            tf.keras.layers.RandomContrast(0.2, value_range=(-1, 1), seed=SEED),
        ],
        name="aumento_de_datos",
    )


def construir():
    """Arma la red: aumento de datos, MobileNetV2 parcialmente descongelada y salida de 11 clases."""
    fijar_semillas()
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(*TAMANIO, 3))
    base.trainable = True
    # Se congelan todas menos las últimas CAPAS_ENTRENABLES. Las primeras detectan
    # bordes y texturas generales, que sirven igual para nubes.
    for capa in base.layers[:-CAPAS_ENTRENABLES]:
        capa.trainable = False
    # Las capas de normalización por lotes quedan congeladas siempre. Si se entrenan,
    # usan las estadísticas del lote al entrenar y las de ImageNet al evaluar, y la red
    # se comporta distinto en cada caso (en Keras 3 no alcanza con training=False).
    for capa in base.layers:
        if isinstance(capa, tf.keras.layers.BatchNormalization):
            capa.trainable = False

    entrada = tf.keras.Input(shape=(*TAMANIO, 3))
    x = aumento_de_datos()(entrada)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    salida = tf.keras.layers.Dense(len(CLASES), activation="softmax")(x)

    modelo = tf.keras.Model(entrada, salida, name=NOMBRE)
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
    conjuntos_eval = ("train", "test") if args.final else ("train", "val")

    # preprocess_input escala a [-1, 1], que es lo que espera MobileNetV2.
    conjuntos = cargar_conjuntos(
        tamanio=TAMANIO,
        batch_size=BATCH_SIZE,
        preprocesar=preprocess_input,
    )
    modelo = construir()
    historial = entrenar(
        modelo,
        conjuntos,
        NOMBRE,
        epocas=EPOCAS,
        paciencia=PACIENCIA,
        parametros={
            "tamaño de imagen": f"{TAMANIO[0]}x{TAMANIO[1]}",
            "batch size": BATCH_SIZE,
            "backbone": f"MobileNetV2 ImageNet, últimas {CAPAS_ENTRENABLES} capas entrenables",
            "aumento de datos": "espejado horizontal, rotación 5 %, zoom 15 %, contraste 20 %",
            "dropout antes de la salida": 0.3,
            "evaluación": "/".join(conjuntos_eval),
        },
    )
    evaluar(modelo, conjuntos, historial, NOMBRE, conjuntos_eval=conjuntos_eval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
