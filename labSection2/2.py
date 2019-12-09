# -*- coding: utf-8 -*-
# Задача 2: Решение полной задачи на собственные значения
import numpy as np
from labSection2.lib.qr_algorithm import qr


def evaluate_and_print_results():
    accuracy = 0.00001
    max_iteration_count = 11
    A = np.array([[5, 6, 3], [-1, 0, 1], [1, 2, -1]])
    result, iterations_count = qr(A, accuracy, max_iteration_count)
    print("Iterations count =", iterations_count, "\n")
    print("Result:", np.diag(result), sep="\n")


evaluate_and_print_results()