# Задача 2: Метод Гаусса с частичным выбором ведущего элемента
import numpy as np


def gauss(x):
    x = np.array(x, float)
    return x[1:] / x[0]


def gauss_app(C, t):
    C = np.array(C, float)
    t = np.array([[t[i]] for i in range(len(t))], float)
    C[1:, :] = C[1:, :] - t * C[0, :]
    return C


def lu_partial(A):
    A = np.array(A, float)
    piv = np.zeros(len(A) - 1, float)
    for k in range(len(A) - 1):
        piv[k] = list(A[k:, k]).index(max(A[k:, k]))
        ch = np.array(A[k, k:len(A)], float)
        A[k, k:len(A)] = A[int(piv[k]), k:len(A)]
        A[int(piv[k]), k:len(A)] = ch
        if A[k, k] != 0:
            t = gauss(A[k:, k])
        A[k + 1:, k] = t
        A[k:, k + 1:] = gauss_app(A[k:, k + 1:], t)
    return A, piv


def solve_lu(A, b):
    LU, piv = lu_partial(A)
    print('piv =', piv)
    ch1 = np.zeros(len(A), float)
    M = []
    E = []
    for i in range(len(A) + 1):
        M.append(np.identity(len(A), float))
        E.append(np.identity(len(A), float))
    for j in range(0, len(A) - 1, 1):
        ch1[:] = E[j][j, :]
        E[j][j, :] = E[j][int(piv[j]), :]
        E[j][int(piv[j]), :] = ch1[:]
    for k in range(1, len(A)):
        M[k - 1][k:, k - 1] = -np.dot(E[k], LU)[k:, k - 1]
    U = np.array(A, float)
    y = np.array(b, float)
    for t in range(0, len(A) - 1):
        U = np.dot(M[t], np.dot(E[t], U))
        y = np.dot(M[t], np.dot(E[t], y))
    result = np.zeros(len(A), float)
    for i in range(len(A), 0, -1):
        temp = 0
        for j in range(i, len(A), 1):
            temp = temp + U[i - 1, j] * result[j]
        result[i - 1] = (y[i - 1] - temp) / U[i - 1, i - 1]
    return result


def test_solve_lu():
    A = np.array([[3, 17, 10], [2, 4, -2], [6, 18, -12]], float)
    expected = np.array([1, 3, 4])
    b = np.dot(A, expected)
    computed = solve_lu(A, b)
    tol = 1e-14
    success = np.linalg.norm(computed - expected) < tol
    msg = 'x_exact = ' + str(expected) + '; x_computed = ' + str(computed)
    assert success
    return msg


print(test_solve_lu())
