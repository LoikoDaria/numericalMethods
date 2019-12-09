# -*- coding: utf-8 -*-
import numpy as np
import time
from math import sqrt


def check_sor_applicability(A, w):
    """
    Функция для проверки применимости метода Последовательной Верхней Релаксации
    @param A: матрица
    @param w: параметр релаксации
    @return:
    """
    t = 0
    for i in range(1, len(A)):
        for j in range(i):
            if A[i, j] == A[j, i]:
                t += 1
    return t == (len(A) ** 2 - len(A)) / 2 and 1 < w < 2


def sor(A, b, x0, w, accuracy, max_iteration_count):
    """
    Метод Последовательной Верхней Релаксации
    @param A: матрица
    @param b: свободный столбец
    @param x0: начальное приблежение
    @param w: параметр релаксации
    @param accuracy: точность
    @param max_iteration_count: максимально число итераций
    @return:
    """
    if check_sor_applicability(A, w):
        start_time = time.time()
        matrix_size = len(A)
        x = np.zeros((2, matrix_size), float)
        x[1, :] = x0
        k = 0
        while sqrt(sum((x[1] - x[0]) ** 2)) > accuracy and k < max_iteration_count:
            x[0, :] = x[1, :]
            for i in range(matrix_size):
                x[1, i] = w * (b[i] - (sum(A[i, :i] * x[1, :i]) + sum(A[i, i + 1:] * x[0, i + 1:]))) / A[i, i] + (
                        1 - w) * x[0, i]
            k += 1
        execution_time = time.time() - start_time
        return x[1], k, execution_time
    else:
        return x0, 0, 0


def sor_vectorized(A, b, x0, w, accuracy, max_iteration_count):
    """
    Метод Последовательной Верхней Релаксации
    @param A: матрица
    @param b: свободный столбец
    @param x0: начальное приблежение
    @param w: параметр релаксации
    @param accuracy: точность
    @param max_iteration_count: максимально число итераций
    @return:
    """
    if check_sor_applicability(A, w):
        start_time = time.time()
        matrix_size = len(A)
        L = np.zeros((matrix_size, matrix_size), float)
        U = np.zeros((matrix_size, matrix_size), float)
        D = np.zeros((matrix_size, matrix_size), float)
        for i in range(matrix_size):
            L[i, :i] = A[i, :i]
            D[i, i] = A[i, i]
            U[i, i + 1:] = A[i, i + 1:]
        x = np.zeros((2, matrix_size), float)
        x[1, :] = x0
        k = 0
        while sqrt(sum((x[1] - x[0]) ** 2)) > accuracy and k < max_iteration_count:
            x[0, :] = x[1, :]
            x[1] = np.dot(np.linalg.inv(D + w * L), (np.dot((1 - w) * D - w * U, x[0]) + w * b))
            k += 1
        execution_time = time.time() - start_time
        return x[1], k, execution_time
    else:
        return x0, 0, 0
