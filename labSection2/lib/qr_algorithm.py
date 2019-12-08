# -*- coding: utf-8 -*-
import numpy as np


def one_iteration_decomposition(A):
    """
    Функция для нахождения разложения для одной итерации
    @param A: Матрица
    @return:
    """
    A1 = np.array(A, float)
    shape = len(A)
    W = np.zeros((shape - 1, shape), float)
    for i in range(shape - 1):
        A0 = A1
        s = (sum(A0[i:, i] ** 2)) ** 0.5
        if (s - A0[i, i]) == 0:
            return np.zeros(shape), np.zeros(shape)
        mu = (2 * s * (s - A0[i, i])) ** (-0.5)
        W[i, :] = mu * np.array([row[i] for row in A0])
        W[i, i] = mu * (A0[i, i] - s)
        W[i, :i] = 0
        A1 = np.dot((np.identity(shape, float) - 2 * np.outer(np.transpose(W[i]), W[i])), A0)
    Q0 = (np.identity(shape, float) - 2 * np.outer(np.transpose(W[0]), W[0]))
    for i in range(1, shape - 1):
        Q0 = np.dot(Q0, (np.identity(shape, float) - 2 * np.outer(np.transpose(W[i]), W[i])))
    return A1, Q0


def qr(A, accuracy, max_iteration_count):
    """
    Функция реализующая QR-алгоритм
    @param A: матрица
    @param accuracy: точность
    @param max_iteration_count: максимально число итераций
    @return:
    """
    iteration_count = 0
    R0, Q0 = one_iteration_decomposition(A)
    R1, Q1 = one_iteration_decomposition(np.dot(R0, Q0))

    while not ((np.diag(R1) - np.diag(R0)) > accuracy).all() and iteration_count < max_iteration_count:
        R0, Q0 = one_iteration_decomposition(np.dot(R1, Q1))
        if (R0.all()) == np.zeros(len(A)).all():
            return R1, iteration_count
        R1, Q1 = one_iteration_decomposition(np.dot(R0, Q0))
        if R1.all() == np.zeros(len(A)).all():
            return R0, iteration_count
        iteration_count += 1
    return R1, iteration_count
