# Proyecto de Machine Learning con JAX en Google Colab

Este proyecto utiliza **Google Colab** para implementar y entrenar un modelo de clasificación multicategoría con **JAX** y herramientas relacionadas. El objetivo principal es construir, entrenar y evaluar un modelo de red neuronal utilizando las herramientas optimizadas de JAX para hardware acelerado (GPU/TPU).
(https://repository-images.githubusercontent.com/154739597/90607180-e100-11e9-8642-c65819bec604)
---

## Tecnologías y Librerías Utilizadas

### 1. **Google Colab**
Google Colab es el entorno de ejecución utilizado en este proyecto. Ofrece acceso gratuito a GPUs y TPUs, haciendo posible el desarrollo de modelos complejos de machine learning de manera eficiente y escalable.

### 2. **Librerías Utilizadas**
- **JAX**: Una biblioteca de Python optimizada para cálculo numérico y aprendizaje automático, diseñada para aprovechar hardware acelerado.  
  - Submódulos destacados:
    - `jax.numpy`: Versión de NumPy optimizada para JAX.
    - `jax.random`: Para generación reproducible de números aleatorios.
    - `jax.nn.one_hot`: Para la codificación de etiquetas en formato One-Hot.
    - `jax.grad`: Para el cálculo automático de gradientes.
    - `jax.jit`: Para la compilación Just-In-Time (JIT), mejorando el rendimiento.

- **Optax**: Biblioteca de optimización compatible con JAX que implementa diversos algoritmos de optimización, como Adam y SGD.
  - Herramientas clave utilizadas:
    - `optax.adam`: Optimizador Adam.
    - `optax.softmax_cross_entropy`: Función de pérdida para problemas de clasificación.

- **Stax**: Biblioteca dentro de JAX para definir redes neuronales de manera secuencial.
  - Componentes destacados:
    - `stax.serial`: Permite construir un modelo como una secuencia de capas.
    - Capas utilizadas: `stax.Dense`, `stax.Relu`, y `stax.Softmax`.

- **Scikit-learn**: Utilizada para el preprocesamiento de datos.
  - Herramientas destacadas:
    - `StandardScaler`: Para la normalización de los datos (media 0 y desviación estándar 1).

- **Matplotlib**: Biblioteca para la visualización de datos.
  - Herramientas utilizadas:
    - `matplotlib.pyplot`: Para gráficos básicos.
    - `ConfusionMatrixDisplay`: Para generar la matriz de confusión y evaluar el rendimiento del modelo.

---

## Estructura del Proyecto

### 1. **Preprocesamiento de Datos**
- **Estandarización**: Los datos se normalizan utilizando `StandardScaler` para asegurar que todas las características tengan una escala similar.
- **Codificación One-Hot**: Las etiquetas categóricas se convierten en vectores One-Hot usando `jax.nn.one_hot`, lo que permite su uso en la función de pérdida de clasificación.

### 2. **Definición del Modelo**
El modelo se construye utilizando `stax.serial` como una red secuencial con las siguientes capas:
- **Capas densas (`stax.Dense`)**:
  - Primera capa: 16 neuronas con activación ReLU.
  - Segunda capa: 8 neuronas con activación ReLU.
  - Salida: 3 neuronas con activación Softmax (para problemas de clasificación multiclase).
- **Función de Activación (`stax.Relu`)**: Aplicada después de las capas densas ocultas.
- **Salida (`stax.Softmax`)**: Convierte las predicciones en probabilidades.

### 3. **Entrenamiento del Modelo**
- **Función de Pérdida**: Calculada usando `optax.softmax_cross_entropy`, que mide la discrepancia entre las predicciones del modelo y las etiquetas verdaderas en formato One-Hot.
- **Optimizador**: Se utiliza el optimizador Adam (`optax.adam`) con un learning rate de 0.001.
- **Gradientes**: Calculados automáticamente con `jax.grad`, y las actualizaciones de parámetros son aceleradas con `jax.jit`.

### 4. **Evaluación del Modelo**
- **Pérdida en el conjunto de prueba**: Calculada con la misma función de pérdida utilizada en el entrenamiento.
- **Precisión del modelo**: Calculada comparando las etiquetas predichas con las reales.
- **Matriz de Confusión**: Visualizada con `ConfusionMatrixDisplay` de Matplotlib para evaluar el rendimiento del modelo en cada clase.

---

## Cómo Ejecutar el Proyecto

1. **Requisitos Previos**
   - Tener acceso a Google Colab.
   - Familiaridad básica con Python y librerías como JAX, Optax y Matplotlib.

2. **Pasos para Ejecutar**
   - Subir el archivo `.ipynb` a Google Colab.
   - Instalar cualquier dependencia adicional que no esté preinstalada en Colab.
   - Configurar el entorno con GPU o TPU:
     - Menú: `Entorno de ejecución > Cambiar tipo de entorno de ejecución > Acelerador de hardware > GPU/TPU`.
   - Ejecutar las celdas en orden secuencial.

---

## Resultados Esperados

- **Modelo Entrenado**:
  - Pérdida en el conjunto de entrenamiento y prueba se reducen progresivamente.
  - Precisión razonable en el conjunto de prueba.

- **Evaluación Gráfica**:
  - **Matriz de Confusión**: Proporciona una visión clara de los aciertos y errores del modelo para cada clase.

---

## Recursos Adicionales

- **Documentación de JAX**: [https://jax.readthedocs.io/](https://jax.readthedocs.io/)
- **Optax**: [https://optax.readthedocs.io/](https://optax.readthedocs.io/)
- **Google Colab**: [https://colab.research.google.com/](https://colab.research.google.com/)
---
# Recursos Extras 
- **QuictStart Jax** : [https://docs.jax.dev/en/latest/quickstart.html?utm_source=chatgpt.com](https://docs.jax.dev/en/latest/quickstart.html?utm_source=chatgpt.com)
- **ML Engineer comparison** : [https://softwaremill.com/ml-engineer-comparison-of-pytorch-tensorflow-jax-and-flax/?utm_source=chatgpt.com](https://softwaremill.com/ml-engineer-comparison-of-pytorch-tensorflow-jax-and-flax/?utm_source=chatgpt.com)
- **Pytorch Jax Tensorflow Comparison** : [https://www.restack.io/p/pytorch-answer-comparison-pytorch-jax-tensorflow](https://www.restack.io/p/pytorch-answer-comparison-pytorch-jax-tensorflow)
- **¿Qué es JAX?** : [https://es.eitca.org/artificial-intelligence/eitc-ai-gcml-google-cloud-machine-learning/google-cloud-ai-platform/introduction-to-jax/examination-review-introduction-to-jax/what-is-jax-and-how-does-it-speed-up-machine-learning-tasks/](https://es.eitca.org/artificial-intelligence/eitc-ai-gcml-google-cloud-machine-learning/google-cloud-ai-platform/introduction-to-jax/examination-review-introduction-to-jax/what-is-jax-and-how-does-it-speed-up-machine-learning-tasks/)
---

## Contribuciones

Este proyecto está diseñado para ayudar a estudiantes y desarrolladores a familiarizarse con JAX y herramientas relacionadas en Google Colab.
---
