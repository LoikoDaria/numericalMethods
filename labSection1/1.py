# Задача 1: Решение системы линейных уравнений с трехдиагональной матрицей
import numpy as np

def lu3(A):
    d = np.array([A[i, i] for i in range(len(A))], float)
    eu = np.array([A[i, i + 1] for i in range(len(A) - 1)], float)
    el = np.array([A[i + 1, i] for i in range(len(A) - 1)], float)
    return d, eu, el


def solve_lu3(A, b):
    d, eu, el = lu3(A)
    alpha = np.zeros(len(A), float)
    alpha[1] = -eu[0] / d[0]
    for i in range(2, len(A)):
        alpha[i] = -eu[i - 1] / (d[i - 1] + el[i - 1] * alpha[i - 1])
    betta = np.zeros(len(A), float)
    betta[1] = b[0] / d[0]
    for i in range(2, len(A)):
        betta[i] = (-el[i - 1] * betta[i - 1] + b[i - 1]) / (d[i - 1] + el[i - 1] * alpha[i - 1])
    x = np.zeros(len(A), float)
    x[-1] = (-el[-1] * betta[-1] + b[-1]) / (d[-1] + el[-1] * alpha[-1])
    for i in range(len(A) - 2, -1, -1):
        x[i] = alpha[i + 1] * x[i + 1] + betta[i + 1]
    return x


A = np.array([[160, 2, 0, 0], [6, 185, 5, 0], [0, 3, 193, 11], [0, 0, 8, 134]])
b = np.array([10, 22, 42, 72])

x = solve_lu3(A, b)
print('result =', np.dot(A, x))
