# -*- coding: utf-8 -*-
import numpy as np
import time


def power_method(A, q0, accuracy, max_iteration_count):
    """
    Степенной метод
    @param A: Матрица
    @param q0: вектор
    @param accuracy: точность
    @param max_iteration_count: максимально число итераций
    @return:
    """
    start_execution_time = time.time()
    q = np.array(q0, float)
    iteration_count = 0
    _lambda = np.zeros(2, float)
    _lambda[0] = 1
    while iteration_count < max_iteration_count and abs(_lambda[1] - _lambda[0]) > accuracy:
        _lambda[0] = _lambda[1]
        z = np.dot(A, q)
        q = z / np.linalg.norm(z)
        _lambda[1] = np.dot(np.dot(A, q), q)
    execution_time = time.time() - start_execution_time
    return _lambda[1], execution_time
