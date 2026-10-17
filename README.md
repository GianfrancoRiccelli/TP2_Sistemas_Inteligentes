# TP2 Sistemas Inteligentes: clasificación de tipos de nubes

Trabajo Práctico N.º 2 de **Inteligencia Artificial** (UTN FRBA, curso K5672, 2.º cuatrimestre de 2026), **Grupo 16**.

El objetivo es clasificar automáticamente una fotografía de cielo tomada desde tierra en uno de **11 tipos de nube**, usando redes neuronales artificiales. Se entrenan y comparan tres modelos: un perceptrón multicapa (MLP) entrenado con backpropagation, una red convolucional construida desde cero y una red convolucional preentrenada.

> **Estado:** en desarrollo. Este README describe el plan de trabajo; las secciones de uso y resultados se completan a medida que avanza la implementación.

## Problema

La Organización Meteorológica Mundial (OMM) clasifica las nubes en diez géneros según su altura y su aspecto. Esa identificación se usa en el pronóstico de corto plazo, en los reportes para la aviación (por ejemplo, la presencia de cumulonimbos) y en estudios de clima y de energía solar. Hoy la hacen a ojo observadores entrenados, por lo que depende de la experiencia de cada persona y es inconsistente entre estaciones.

No se puede resolver con reglas explícitas: lo que distingue, por ejemplo, un altocúmulo de un cirrocúmulo o un estrato de un nimbostrato es la textura, la distribución y la opacidad de la nube, y esas propiedades cambian con la iluminación, la hora y el ángulo de la toma. Por eso se usa un sistema que aprende esas regularidades a partir de ejemplos etiquetados.

- **Entrada:** imagen RGB de 256×256 píxeles.
- **Salida:** una de las 11 clases (la red devuelve una probabilidad por clase y se elige la mayor).

## Dataset

Se usa **CCSN (Cirrus Cumulus Stratus Nimbus) v2**, publicado en Harvard Dataverse bajo licencia CC0: 2.543 imágenes JPEG de 256×256 RGB, etiquetadas por profesionales meteorológicos y organizadas en una carpeta por clase. Incluye los 10 géneros de la OMM más las estelas de condensación de aviones (contrails).

| Clase | Abreviatura | Imágenes |
|---|---|---:|
| Altocúmulo | Ac | 221 |
| Altostrato | As | 188 |
| Cumulonimbo | Cb | 242 |
| Cirrocúmulo | Cc | 268 |
| Cirro | Ci | 139 |
| Cirrostrato | Cs | 287 |
| Estela de avión | Ct | 200 |
| Cúmulo | Cu | 182 |
| Nimbostrato | Ns | 274 |
| Estratocúmulo | Sc | 340 |
| Estrato | St | 202 |
| **Total** | | **2.543** |

El conjunto está moderadamente desbalanceado: la clase mayor (Sc, 340) tiene casi 2,5 veces más ejemplos que la menor (Ci, 139). Algunas clases se confunden incluso a simple vista, como Ac y Cc, St y Ns, o As y Cs.

## Metodología

### Preparación y partición de los datos

- Los píxeles se normalizan al rango [0, 1]. La red preentrenada usa el preprocesamiento propio de su arquitectura.
- Las etiquetas se codifican como un vector de 11 posiciones (one-hot).
- La partición es **estratificada 60 / 20 / 20** (entrenamiento / validación / prueba), con una semilla fija para que sea reproducible. Primero se separa el conjunto de prueba y luego el resto se divide en entrenamiento y validación. Los tres modelos usan la misma partición.
- El **conjunto de prueba no se usa para ninguna decisión de diseño** (arquitectura, hiperparámetros, cantidad de épocas). Esas decisiones se toman con el conjunto de validación, y la prueba se evalúa una sola vez, al final.
- Si se usa aumento de datos, se aplica solo al conjunto de entrenamiento.

### Modelos

1. **MLP con backpropagation:** red multicapa densa con aprendizaje supervisado por corrección de errores (regla delta generalizada). Recibe la imagen reducida y aplanada en un vector, y tiene 11 neuronas de salida con Softmax. Es el modelo de referencia; no aprovecha la estructura bidimensional de la imagen.
2. **Red convolucional desde cero:** capas de convolución, ReLU y pooling, seguidas de una capa de aplanado y una salida de 11 neuronas con Softmax. Recibe la imagen completa de 256×256×3.
3. **Red convolucional preentrenada (transfer learning):** reutiliza las capas convolucionales de una arquitectura entrenada sobre ImageNet y reemplaza la salida por una de 11 clases. Se espera que sirva por el tamaño acotado del dataset.

La topología final de cada red (capas, neuronas, activaciones, optimizador, tasa de aprendizaje) se define durante la experimentación usando el conjunto de validación.

### Resultados que se reportan por modelo

- Topología de la red y valores de los parámetros de entrenamiento.
- Gráfico de la evolución de la loss durante el entrenamiento.
- Matriz de confusión, precisión y recuperación (recall) por clase, **sobre el conjunto de entrenamiento y sobre el de prueba**.

## Herramientas

Python, TensorFlow/Keras, scikit-learn, NumPy y matplotlib.

## Integrantes

Thomas Leandro Pereyra · Valentina Luján Rau Robayo · José Gerónimo González Peñaflor · María Guadalupe Fernandez · Agustina Brenda Alaniz Kvitkin · Gianfranco Ricceli · Juan Ignacio Mosmann

## Referencias

- Liu, P. (2019). *Cirrus Cumulus Stratus Nimbus (CCSN) Database*, V2. Harvard Dataverse. https://doi.org/10.7910/DVN/CADDPD
- Zhang, J. L., Liu, P., Zhang, F., & Song, Q. Q. (2018). CloudNet: Ground-Based Cloud Classification With Deep Convolutional Neural Network. *Geophysical Research Letters*, 45, 8665–8672. https://doi.org/10.1029/2018GL077787
