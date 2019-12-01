# Задача 3: Разложение Холецкого
import numpy as np
from math import sqrt


def cholesky_decomposition(A):
    L = np.zeros((len(A), len(A)), float)
    for i in range(len(A)):
        for j in range(0, i, 1):
            temp1 = 0
            for k in range(0, j, 1):
                temp1 += L[i, k] * L[j, k]
            L[i, j] = (A[i, j] - temp1) / L[j, j]
        temp2 = 0
        for t in range(0, i, 1):
            temp2 += L[i, t] * L[i, t]
        L[i, i] = sqrt(A[i, i] - temp2)
    return L


def solve_lower_triangular_matrix(GGt, b):
    y = np.zeros(len(GGt))
    for i in range(len(GGt)):
        temp = 0
        for j in range(i):
            temp += GGt[i, j] * y[j]
        y[i] = (b[i] - temp) / GGt[i, i]
    return y


def solve_transposed_lower_triangular_matrix(GGt, y):
    x = np.zeros(len(GGt), float)
    for i in range(len(GGt), 0, -1):
        temp = 0
        for j in range(i, len(GGt), 1):
            temp += GGt[i - 1, j] * x[j]
        x[i - 1] = (y[i - 1] - temp) / GGt[i - 1, i - 1]
    return x


def test_cholesky_decomposition():
    A = np.array([[17, 3, 10], [3, 17, -2], [10, -2, 12]], float)
    expected = np.array([1, 3, 4])
    b = np.dot(A, expected)
    GGt = cholesky_decomposition(A)
    print('cholesky decomposition:', GGt,sep='\n')
    y = solve_lower_triangular_matrix(GGt, b)
    computed = solve_transposed_lower_triangular_matrix(GGt.transpose(), y)
    tol = 1e-14
    success = np.linalg.norm(computed - expected) < tol
    assert success
    msg = 'x_exact = ' + str(expected) + '; x_computed = ' + str(computed)
    return msg


result_message = test_cholesky_decomposition()
print(result_message)
