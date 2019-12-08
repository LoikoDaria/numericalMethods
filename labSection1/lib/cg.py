# -*- coding: utf-8 -*-
import numpy as np
import time


def cg(A, b, accuracy, max_iteration_count):
    """
    Метод собряженных градиентов
    @param A: матрица
    @param b: свободный столбец
    @param accuracy: точность
    @param max_iteration_count: максимальное число итераций
    @return:
    """
    start_time = time.time()
    k = 0
    x = 0
    r = np.copy(b)
    r_prev = np.copy(b)
    rho = np.dot(r, r)
    p = np.copy(r)
    while np.sqrt(rho) > accuracy * np.sqrt(np.dot(b, b)) and k < max_iteration_count:
        k += 1
        if k == 1:
            p[:] = r[:]
        else:
            beta = np.dot(r, r) / np.dot(r_prev, r_prev)
            p = r + beta * p
            w = np.dot(A, p)
            alpha = np.dot(r, r) / np.dot(p, w)
            x = x + alpha * p
            r_prev[:] = r[:]
            r = r - alpha * w
            rho = np.dot(r, r)
    execution_time = time.time() - start_time
    return x, k, execution_time
