# Regresión Lineal con Descenso del Gradiente en Python

Este proyecto es una implementación desde cero del algoritmo de **Descenso del Gradiente** para resolver un problema de **Regresión Lineal Múltiple**. Se utiliza el famoso dataset **Iris** para predecir la longitud del pétalo de una flor a partir de otras tres características.

El objetivo principal es tanto demostrar el funcionamiento interno del algoritmo como validar su correcta implementación al comparar los resultados con la librería `scikit-learn`.

---

## 🧠 Algoritmo Implementado: Descenso del Gradiente

El Descenso del Gradiente es un algoritmo de optimización que minimiza iterativamente una función de costo. La intuición es simple: para encontrar el mínimo de una función (el "valle de una montaña"), uno debe dar pasos repetidamente en la dirección opuesta al gradiente (la "pendiente") en el punto actual.

En este proyecto, se utiliza para minimizar el **Error Cuadrático Medio (MSE)**, ajustando los pesos (`W`) y el sesgo (`b`) del modelo lineal `y = W*X + b`.

---

## ✨ Características del Proyecto

* **Implementación desde Cero**: El algoritmo de Descenso del Gradiente está escrito utilizando únicamente la librería **NumPy** para operaciones matemáticas.
* **Visualización de Resultados**:
    1.  **Curva de Costo**: Una gráfica que muestra la evolución del error a lo largo de las iteraciones, permitiendo verificar la convergencia del algoritmo.
    2.  **Plano de Regresión 3D**: Una visualización del modelo ajustado frente a los puntos de datos reales.
* **Validación del Modelo**: Una tabla comparativa que muestra los parámetros calculados por nuestra implementación frente a los obtenidos con el modelo `LinearRegression` de `scikit-learn`, confirmando la exactitud del código.

---

## 🔧 Requisitos

Necesitarás tener las siguientes librerías de Python instaladas:
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`

Puedes instalarlas usando pip:
pip install numpy pandas scikit-learn matplotlib

## 🚀 Cómo Ejecutar
Simplemente clona el repositorio y ejecuta el script de Python desde tu terminal:
`python Gradiente_descendente.py`
El script se ejecutará, mostrará las gráficas y finalmente imprimirá en la consola la tabla comparativa de los resultados.
