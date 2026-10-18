"""Funciones compartidas por los tres modelos: carga de datos, entrenamiento y evaluación.

Flujo típico en el script de cada modelo:

    conjuntos = cargar_conjuntos(tamanio=(256, 256))
    modelo = ...  # construir y compilar (loss="categorical_crossentropy")
    historial = entrenar(modelo, conjuntos, "cnn_desde_cero")
    evaluar(modelo, conjuntos, historial, "cnn_desde_cero")

Regla del TP: el conjunto de prueba no se usa para decisiones de diseño. Mientras se
prueban variantes, evaluar con conjuntos_eval=("train", "val") y dejar ("train", "test")
para la corrida final, cuando el modelo ya está fijo.
"""

import csv
import io
import os

# Tienen que definirse antes de importar TensorFlow.
# 2 = ocultar los mensajes informativos y de advertencia de TensorFlow (los errores se siguen viendo)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Desactiva las optimizaciones oneDNN: saca su aviso y hace los resultados reproducibles entre corridas
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib

matplotlib.use("Agg")  # solo se guardan las figuras, no se abren ventanas
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from config import CLASES, CONJUNTOS, NOMBRES_CLASES, RUTA_DATOS, RUTA_RESULTADOS, SEED

tf.get_logger().setLevel("ERROR")  # advertencias del lado de Python

NOMBRES_CONJUNTOS = {"train": "entrenamiento", "val": "validación", "test": "prueba"}

# Colores de las figuras
COLOR_ENTRENAMIENTO = "#2a78d6"
COLOR_VALIDACION = "#eb6834"
COLOR_TEXTO = "#0b0b0b"
COLOR_TEXTO_SECUNDARIO = "#52514e"
COLOR_GRILLA = "#e4e3df"
COLOR_FONDO = "#fcfcfb"
# Escala secuencial de un solo tono (azul) para la matriz de confusión: 0 = color de fondo
MAPA_AZUL = LinearSegmentedColormap.from_list(
    "azul", [COLOR_FONDO, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------


def fijar_semillas(seed=SEED):
    """Fija las semillas de Python, NumPy y TensorFlow."""
    tf.keras.utils.set_random_seed(seed)


def cargar_conjuntos(tamanio=(256, 256), batch_size=32, preprocesar=None):
    """Carga data/train, data/val y data/test como tf.data.Dataset.

    - Todas las imágenes se redimensionan a `tamanio` (en el dataset original hay
      imágenes de 400x400 y de 256x256).
    - Las etiquetas vienen en one-hot, con las clases en el orden de config.CLASES.
    - Solo el conjunto de entrenamiento se mezcla (con la semilla fija).
    - `preprocesar`: función que se aplica a las imágenes (valores 0-255 en float32).
      Por defecto se escala a [0, 1]. Para la red preentrenada, pasar el
      preprocess_input del backbone.

    La data augmentation no va acá: conviene agregarla como capas del modelo
    (RandomFlip, RandomRotation, etc.), que solo actúan durante el entrenamiento.
    """
    fijar_semillas()
    if preprocesar is None:
        preprocesar = lambda x: x / 255.0

    conjuntos = {}
    for conjunto in CONJUNTOS:
        ds = tf.keras.utils.image_dataset_from_directory(
            str(RUTA_DATOS / conjunto),
            labels="inferred",
            label_mode="categorical",
            class_names=CLASES,
            color_mode="rgb",
            image_size=tamanio,
            batch_size=batch_size,
            shuffle=(conjunto == "train"),
            seed=SEED,
        )
        ds = ds.map(lambda x, y: (preprocesar(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        conjuntos[conjunto] = ds.prefetch(tf.data.AUTOTUNE)
    return conjuntos


def calcular_pesos_clase():
    """Pesos por clase inversamente proporcionales a su frecuencia en entrenamiento.

    Para pasar como class_weight a entrenar() si el desbalance afecta a las clases chicas.
    """
    cantidades = np.array(
        [sum(1 for p in (RUTA_DATOS / "train" / c).iterdir() if p.is_file()) for c in CLASES]
    )
    pesos = cantidades.sum() / (len(CLASES) * cantidades)
    return {i: float(p) for i, p in enumerate(pesos)}


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------


def entrenar(
    modelo,
    conjuntos,
    nombre_modelo,
    epocas=100,
    paciencia=10,
    class_weight=None,
    callbacks_extra=None,
    parametros=None,
):
    """Entrena un modelo ya compilado, con EarlyStopping sobre la loss de validación.

    Se queda con los pesos de la mejor época (restore_best_weights). Guarda en
    results/<nombre_modelo>/ la topología y los parámetros (config.txt) y la loss
    por época (historial.csv).

    `parametros`: diccionario con datos extra para config.txt (por ejemplo, el
    tamaño de imagen o el batch size usados al cargar los datos).
    """
    carpeta = RUTA_RESULTADOS / nombre_modelo
    carpeta.mkdir(parents=True, exist_ok=True)

    parada_temprana = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=paciencia,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks = [parada_temprana, *(callbacks_extra or [])]

    historial = modelo.fit(
        conjuntos["train"],
        validation_data=conjuntos["val"],  # nunca el conjunto de prueba
        epochs=epocas,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    mejor_epoca = int(np.argmin(historial.history["val_loss"])) + 1
    _guardar_historial(historial, carpeta / "historial.csv")
    _guardar_configuracion(
        modelo,
        carpeta / "config.txt",
        {
            "épocas máximas": epocas,
            "épocas entrenadas": len(historial.history["loss"]),
            "mejor época (menor val_loss)": mejor_epoca,
            "paciencia EarlyStopping": paciencia,
            "class_weight": "sí" if class_weight else "no",
            "callbacks extra": ", ".join(type(c).__name__ for c in callbacks_extra or []) or "ninguno",
            **(parametros or {}),
        },
    )
    return historial


def _guardar_historial(historial, ruta):
    claves = list(historial.history)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["epoca", *claves])
        for i in range(len(historial.history[claves[0]])):
            escritor.writerow([i + 1, *(historial.history[k][i] for k in claves)])


def _guardar_configuracion(modelo, ruta, datos):
    resumen = io.StringIO()
    modelo.summary(print_fn=lambda linea, **_: resumen.write(linea + "\n"))

    optimizador = modelo.optimizer
    config_opt = optimizador.get_config()
    lr = config_opt.get("learning_rate")
    if isinstance(lr, dict):  # learning rate con schedule
        lr = lr.get("class_name", lr)

    with open(ruta, "w", encoding="utf-8") as f:
        f.write(f"Modelo: {modelo.name}\n\n")
        f.write(resumen.getvalue())
        f.write("\nEntrenamiento\n")
        f.write(f"  optimizador: {type(optimizador).__name__}\n")
        f.write(f"  learning rate inicial: {lr}\n")
        f.write(f"  loss: {modelo.loss}\n")
        for clave, valor in datos.items():
            f.write(f"  {clave}: {valor}\n")


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------


def predecir(modelo, ds):
    """Devuelve (clases reales, clases predichas) como índices, en modo inferencia.

    Las etiquetas y las predicciones se toman del mismo lote, así que sirve aunque
    el conjunto esté mezclado. Las capas de augmentation y Dropout no actúan acá.
    """
    reales, predichas = [], []
    for x, y in ds:
        probabilidades = modelo.predict_on_batch(x)
        reales.append(np.argmax(y, axis=1))
        predichas.append(np.argmax(probabilidades, axis=1))
    return np.concatenate(reales), np.concatenate(predichas)


def evaluar(modelo, conjuntos, historial, nombre_modelo, conjuntos_eval=("train", "test")):
    """Guarda en results/<nombre_modelo>/ los resultados del modelo.

    - loss.png: loss de entrenamiento y validación por época.
    - Por cada conjunto evaluado (por defecto entrenamiento y prueba):
      cm_<conjunto>.png (matriz de confusión) y metricas_<conjunto>.csv
      (precisión y recuperación por clase).
    """
    carpeta = RUTA_RESULTADOS / nombre_modelo
    carpeta.mkdir(parents=True, exist_ok=True)

    graficar_loss(historial, nombre_modelo, carpeta / "loss.png")

    for conjunto in conjuntos_eval:
        reales, predichas = predecir(modelo, conjuntos[conjunto])
        exactitud = float(np.mean(reales == predichas))
        print(f"{nombre_modelo} - {NOMBRES_CONJUNTOS[conjunto]}: exactitud {exactitud:.3f}")

        graficar_matriz_confusion(
            reales,
            predichas,
            f"{nombre_modelo}: matriz de confusión ({NOMBRES_CONJUNTOS[conjunto]})",
            carpeta / f"cm_{conjunto}.png",
        )
        guardar_metricas(reales, predichas, exactitud, carpeta / f"metricas_{conjunto}.csv")

    print(f"Resultados guardados en {carpeta}")


def guardar_metricas(reales, predichas, exactitud, ruta):
    """CSV con precisión y recuperación por clase, el promedio macro y la exactitud."""
    precision, recall, _, soporte = precision_recall_fscore_support(
        reales, predichas, labels=range(len(CLASES)), zero_division=0
    )
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["clase", "nombre", "precision", "recuperacion", "soporte"])
        for i, clase in enumerate(CLASES):
            escritor.writerow(
                [clase, NOMBRES_CLASES[clase], f"{precision[i]:.4f}", f"{recall[i]:.4f}", int(soporte[i])]
            )
        escritor.writerow(
            ["promedio macro", "", f"{precision.mean():.4f}", f"{recall.mean():.4f}", int(soporte.sum())]
        )
        escritor.writerow(["exactitud", "", "", "", f"{exactitud:.4f}"])


def graficar_loss(historial, nombre_modelo, ruta):
    loss = historial.history["loss"]
    val_loss = historial.history["val_loss"]
    epocas = np.arange(1, len(loss) + 1)
    mejor = int(np.argmin(val_loss)) + 1

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLOR_FONDO)
    ax.set_facecolor(COLOR_FONDO)
    ax.plot(epocas, loss, color=COLOR_ENTRENAMIENTO, linewidth=2, label="Entrenamiento")
    ax.plot(epocas, val_loss, color=COLOR_VALIDACION, linewidth=2, label="Validación")
    ax.axvline(mejor, color=COLOR_TEXTO_SECUNDARIO, linewidth=1, linestyle="--")
    ax.annotate(
        f"mejor época: {mejor}",
        xy=(mejor, 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -4),
        textcoords="offset points",
        va="top",
        fontsize=9,
        color=COLOR_TEXTO_SECUNDARIO,
    )

    ax.set_title(f"{nombre_modelo}: loss por época", loc="left", fontsize=12, color=COLOR_TEXTO, pad=12)
    ax.set_xlabel("Época", color=COLOR_TEXTO_SECUNDARIO)
    ax.set_ylabel("Loss (entropía cruzada)", color=COLOR_TEXTO_SECUNDARIO)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color=COLOR_GRILLA, linewidth=0.8)
    ax.tick_params(colors=COLOR_TEXTO_SECUNDARIO, length=0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(COLOR_GRILLA)
    leyenda = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for texto in leyenda.get_texts():
        texto.set_color(COLOR_TEXTO_SECUNDARIO)

    fig.tight_layout()
    fig.savefig(ruta, dpi=150, facecolor=COLOR_FONDO)
    plt.close(fig)


def graficar_matriz_confusion(reales, predichas, titulo, ruta):
    """Matriz de confusión con cantidades en cada celda.

    El color de cada celda es la proporción dentro de su fila (la recuperación en la
    diagonal), para poder comparar clases con distinta cantidad de imágenes.
    """
    cm = confusion_matrix(reales, predichas, labels=range(len(CLASES)))
    totales_fila = cm.sum(axis=1, keepdims=True)
    proporcion = np.divide(cm, totales_fila, out=np.zeros(cm.shape), where=totales_fila > 0)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=COLOR_FONDO)
    imagen = ax.imshow(proporcion, cmap=MAPA_AZUL, vmin=0, vmax=1)

    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            if cm[i, j] == 0:
                continue
            color = "white" if proporcion[i, j] > 0.5 else COLOR_TEXTO
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(len(CLASES)), CLASES)
    ax.set_yticks(range(len(CLASES)), CLASES)
    ax.set_xlabel("Clase predicha", color=COLOR_TEXTO_SECUNDARIO)
    ax.set_ylabel("Clase real", color=COLOR_TEXTO_SECUNDARIO)
    ax.set_title(titulo, loc="left", fontsize=12, color=COLOR_TEXTO, pad=12)
    ax.tick_params(colors=COLOR_TEXTO_SECUNDARIO, length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    # Líneas finas de separación entre celdas
    ax.set_xticks(np.arange(-0.5, len(CLASES)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASES)), minor=True)
    ax.grid(which="minor", color=COLOR_FONDO, linewidth=2)
    ax.tick_params(which="minor", length=0)

    barra = fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)
    barra.set_label("Proporción de la fila", color=COLOR_TEXTO_SECUNDARIO)
    barra.ax.tick_params(colors=COLOR_TEXTO_SECUNDARIO, length=0)
    barra.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(ruta, dpi=150, facecolor=COLOR_FONDO)
    plt.close(fig)
