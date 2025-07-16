#Gómez Meza Jorge Ángel
#Grupo 1 - Módulo 1: Inducción

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# a) Cargamos y preparamos el dataset Iris
# Cargamos los datos como un DataFrame de pandas
iris = load_iris(as_frame=True)
df = iris.frame

# Renombramos columnas para poder facilitar el acceso
df.columns = [col.lower().replace(" (cm)", "").replace(" ", "_") for col in df.columns]

# Extraemos las variables independientes (X) y la dependiente (y) 
X = df[['sepal_width', 'petal_width', 'sepal_length']].values
y = df['petal_length'].values

# b) Implementamos el Descenso del Gradiente 
def run_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Implementa el algoritmo de descenso del gradiente para regresión lineal.
    """
    n_samples, n_features = X.shape
    
    # 1. Inicializamos los parámetros (W y b) con ceros 
    weights = np.zeros(n_features)
    bias = 0
    cost_history = []

    # 2. Repetimos para K iteraciones
    for _ in range(n_iterations):
        # Calculamos las predicciones con el modelo lineal: y_hat = X * W + b 
        y_predicted = np.dot(X, weights) + bias
        
        # Calculamos el costo (Error Cuadrático Medio) 
        cost = (1 / (2 * n_samples)) * np.sum((y_predicted - y)**2)
        cost_history.append(cost)
        
        # Calculamos los gradientes parciales
        # Derivada respecto a los pesos (W)
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        # Derivada respecto al sesgo (b)
        db = (1 / n_samples) * np.sum(y_predicted - y)
        
        # Actualizamos los parámetros
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
    return weights, bias, cost_history

# Aquí entrenamos nuestro modelo
learning_rate = 0.01
n_iterations = 1500
custom_weights, custom_bias, cost_history = run_gradient_descent(X, y, learning_rate, n_iterations)

# c) Mostramos nuestros resultados
# 1. Evolución de la función de costo f(W, b)
print("--- Evolución de la Función de Costo ---")
plt.figure(figsize=(10, 6))
plt.plot(range(n_iterations), cost_history)
plt.title('Evolución de la Función de Costo (MSE) durante el Entrenamiento')
plt.xlabel('Número de Iteraciones')
plt.ylabel('Costo (Error Cuadrático Medio)')
plt.grid(True)
plt.show()

# 2. Gráfica de los puntos reales y el plano ajustado 
print("\n--- Gráfica del Ajuste del Modelo ---")
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Puntos de datos reales
ax.scatter(X[:, 0], X[:, 1], y, color='red', marker='o', label='Datos Reales')

# Creamos una malla para graficar el plano de regresión
x_surf, y_surf = np.meshgrid(
    np.linspace(X[:, 0].min(), X[:, 0].max(), 10),
    np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
)
# Usamos el valor medio de 'sepal_length' para visualizar el plano en 3D
z_surf = (custom_weights[0] * x_surf + 
          custom_weights[1] * y_surf + 
          custom_weights[2] * X[:, 2].mean() + 
          custom_bias)

ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.5, cmap='viridis', label='Plano de Regresión Ajustado')

ax.set_xlabel('Sepal Width (x1)')
ax.set_ylabel('Petal Width (x2)')
ax.set_zlabel('Petal Length (y)')
ax.set_title('Plano de Regresión vs. Puntos Reales')
plt.show()

# 3. Comparamos los resultados con LinearRegression de sklearn 
print("\n--- Comparación de Resultados con scikit-learn ---")
# Entrenamos el modelo de scikit-learn
sklearn_model = LinearRegression()
sklearn_model.fit(X, y)

# Presentamos los resultados en una tabla
results_df = pd.DataFrame({
    'Parámetro': ['Peso 1 (sepal_width)', 'Peso 2 (petal_width)', 'Peso 3 (sepal_length)', 'Sesgo (bias)'],
    'Implementación Propia': [custom_weights[0], custom_weights[1], custom_weights[2], custom_bias],
    'scikit-learn': [sklearn_model.coef_[0], sklearn_model.coef_[1], sklearn_model.coef_[2], sklearn_model.intercept_]
})
print(results_df.to_string(index=False))