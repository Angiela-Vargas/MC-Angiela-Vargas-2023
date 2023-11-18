import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def pedir_coordenadas():
    coords = []
    for i in range(5):
        try:
            x = float(input(f"Ingrese el valor en x para la coordenada {i + 1}: "))
            y = float(input(f"Ingrese el valor en y para la coordenada {i + 1}: "))
            coords.append([x, y])
        except ValueError:
            print("Entrada inválida. Por favor, ingrese números válidos.")
    return np.array(coords)

def pedir_grado():
    while True:
        try:
            grado = int(input("Ingrese el grado de la regresión polinómica (debe ser mayor que 2): "))
            if grado > 2:
                break
            else:
                print("Entrada inválida. Por favor, ingrese un grado mayor que 2.")
        except ValueError:
            print("Entrada inválida. Por favor, ingrese un número entero.")
    return grado

def main():
    coordenadas = pedir_coordenadas()
    grado = pedir_grado()

    X = coordenadas[:, 0].reshape(-1, 1)
    y = coordenadas[:, 1]

    características_polinómicas = PolynomialFeatures(degree=grado)
    X_polinomio = características_polinómicas.fit_transform(X)

    regresion_polinomica = LinearRegression()
    regresion_polinomica.fit(X_polinomio, y)

    plt.scatter(coordenadas[:, 0], coordenadas[:, 1], color='red')
    plt.plot(X, regresion_polinomica.predict(X_polinomio), color='blue')
    plt.title(f"Regresión Polinómica (grado {grado})")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
