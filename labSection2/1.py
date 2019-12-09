# -*- coding: utf-8 -*-
# Задача 1: Нахождение максимального собственного значения степенным методом
import numpy as np
from labSection2.lib.power_method import power_method


def evaluate_and_print_results(shape):
    """
    Функция для расчета Степенным методом и вывода результатов на экран
    @param shape: размерность
    @return:
    """
    A = np.zeros((shape, shape), float)
    q0 = np.zeros(shape, float)
    q0[0] = 1
    for i in range(shape):
        for j in range(shape):
            A[i][j] = 1 / (i + j + 1)
    maximal_lambda, execution_time = power_method(A, q0, 0.001, 1000)
    lambda_string="lambda = %s" % maximal_lambda
    execution_time_string="Execution time: %f" % execution_time
    print(lambda_string, execution_time_string, "\n", sep="\n")


for shape in range(2, 11):
    evaluate_and_print_results(shape)
