# -*- coding: utf-8 -*-
import numpy as np
import time
from math import sqrt


def check_seidel_applicability(A):
    """
    Функция для проверки применимости метода зейделя
    @param A:
    @return:
    """
    t = 0
    for i in range(1, len(A)):
        for j in range(0, i):
            if A[i, j] == A[j, i]:
                t += 1
    return t == (len(A) ** 2 - len(A)) / 2


def seidel(A, b, x0, accuracy, max_iteration_count):
    """
    Метод Зейделя
    @param A: матрица
    @param b: свободный столбец
    @param x0: начальное приблежение
    @param accuracy: точность
    @param max_iteration_count: максимально число итераций
    @return:
    """
    if check_seidel_applicability(A):
        start_time = time.time()
        matrix_size = len(A)
        x = np.zeros((2, matrix_size), float)
        x[1, :] = x0
        k = 0
        while sqrt(sum((x[1] - x[0]) ** 2)) > accuracy and k < max_iteration_count:
            x[0, :] = x[1, :]
            for i in range(matrix_size):
                x[1, i] = (b[i] - (sum(A[i, :i] * x[1, :i]) + sum(A[i, i + 1:] * x[0, i + 1:]))) / A[i, i]
            k += 1
        execution_time = time.time() - start_time
        return x[1], k, execution_time
    else:
        return x0, 0, 0


def seidel_vectorized(A, b, x0, accuracy, max_iteration_count):
    """
    Метод Зейделя
    @param A: матрица
    @param b: свободный столбец
    @param x0: начальное приблежение
    @param accuracy: точность
    @param max_iteration_count: максимально число итераций
    @return:
    """
    if check_seidel_applicability(A):
        matrix_size = len(A)
        start_time = time.time()
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
            x[1] = np.dot(np.linalg.inv(D + L), (np.dot(-U, x[0]) + b))
            k += 1
        execution_time = time.time() - start_time
        return x[1], k, execution_time
    else:
        return x0, 0, 0
