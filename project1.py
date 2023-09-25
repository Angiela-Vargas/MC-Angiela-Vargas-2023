

import sympy as sp

def factorial(num):
    if num == 0:
        return 1
    else:
        return num * factorial(num-1)

def seriemclaurinn(funcion, valor, n):
    x = sp.symbols('x')
    valor = valor % (2 * sp.pi)  
    resultado = 0.0

    for k in range(n):
        termino = ((-1) ** k) * (valor ** (2 * k)) / factorial(2 * k)
        resultado += termino

    return resultado

def sinmac(x, n):
    resultado = seriemclaurinn(sp.sin(x), x, n)
    print(f"sin({valorx}): {resultado:.8f}")

def cosmac(x, n):
    resultado = seriemclaurinn(sp.cos(x), x, n)
    print(f"cos({valorx}): {resultado:.8f}")

def tanmac(x, n):
    sin_x = seriemclaurinn(sp.sin(x), x, n)
    cos_x = seriemclaurinn(sp.cos(x), x, n)
    resultado = sin_x / cos_x
    print(f"tan({valorx}): {resultado:.8f}")

def secmac(x, n):
    cos_x = seriemclaurinn(sp.cos(x), x, n)
    resultado = 1 / cos_x
    print(f"sec({valorx}): {resultado:.8f}")

def cosecmac(x, n):
    sin_x = seriemclaurinn(sp.sin(x), x, n)
    resultado = 1 / sin_x
    print(f"csc({valorx}): {resultado:.8f}")

def cotmac(x, n):
    sin_x = seriemclaurinn(sp.sin(x), x, n)
    cos_x = seriemclaurinn(sp.cos(x), x, n)
    resultado = cos_x / sin_x
    print(f"cot({valorx}): {resultado:.8f}")

print("1. Seno")
print("2. Coseno")
print("3. Tangente")
print("4. Secante")
print("5. Cosecante")
print("6. Cotangente")

seleccion = int(input("Ingrese el número de la operación que deseas realizar "))
valorx = float(input("Ingresa el valor de x en radianes "))
n = int(input("Ingresa el valor de n para la serie de Maclaurin: "))

if seleccion == 1:
    sinmac(valorx, n)

elif seleccion == 2:
    cosmac(valorx, n)

elif seleccion == 3:
    tanmac(valorx, n)

elif seleccion == 4:
    secmac(valorx, n)

elif seleccion == 5:
    cosecmac(valorx, n)

elif seleccion == 6:
    cotmac(valorx, n)
