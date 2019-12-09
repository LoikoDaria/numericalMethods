# -*- coding: utf-8 -*-
# Задача 4: Метод Якоби
# Задача 5: Метод Зейделя
# Задача 6: Сравнение методов Якоби и Зейделя
# Задача 7: Метод верхней релаксации
# Задача 8: Метод сопряженных градиентов

import numpy as np
from labSection1.lib.jacobi import jacobi, jacobi_vectorized
from labSection1.lib.seidel import seidel, seidel_vectorized
from labSection1.lib.sor import sor, sor_vectorized
from labSection1.lib.cg import cg
import matplotlib.pyplot as plt


def evaluate(shape, alpha, omega):
    A = np.zeros((shape, shape), float)
    b = np.zeros(shape)
    x0 = np.zeros(shape)
    A[0, 0] = 2
    A[0, 1] = alpha - 1
    A[shape - 1, shape - 1] = 2
    A[shape - 1, shape - 2] = alpha - 1
    b[0] = 1 - alpha
    b[shape - 1] = 0.8
    x0[0] = 0.7
    x0[shape - 1] = 0.8
    for i in range(1, shape - 1):
        A[i, i] = 2
        A[i, i + 1] = alpha - 1
        A[i, i - 1] = alpha - 1
        b[i] = 0
        x0[i] = i * 0.1
    jacobi_result = jacobi(A, b, x0, 0.00001, 1000)
    jacobi_vectorized_result = jacobi_vectorized(A, b, x0, 0.00001, 1000)
    seidel_result = seidel(A, b, x0, 0.00001, 1000)
    seidel_vectorized_result = seidel_vectorized(A, b, x0, 0.00001, 1000)
    sor_result = sor(A, b, x0, omega, 0.00001, 1000)
    sor_vectorized_result = sor_vectorized(A, b, x0, omega, 0.00001, 1000)
    cg_result = cg(A, b, 0.00001, 1000)
    return jacobi_result[1], jacobi_vectorized_result[1], seidel_result[1], seidel_vectorized_result[1], sor_result[1], \
           sor_vectorized_result[1], cg_result[1]


max_integer_shape = 5
alpha = 0.3
omega = 1.3
shapes_list = []
jacobi_iterations_count_list = []
jacobi_vectorized_iterations_count_list = []
seidel_iterations_count_list = []
seidel_vectorized_iterations_count_list = []
cg_iterations_count_list = []

for i in range(2, max_integer_shape):
    current_shape = i + 1
    shapes_list.append(current_shape)
    evaluations_results = evaluate(current_shape, alpha, omega)
    jacobi_iterations_count_list.append(evaluations_results[0])
    jacobi_vectorized_iterations_count_list.append(evaluations_results[1])
    seidel_iterations_count_list.append(evaluations_results[2])
    seidel_vectorized_iterations_count_list.append(evaluations_results[3])
    cg_iterations_count_list.append(evaluations_results[6])

various_accuracy_list = np.arange(0.1, 1, 0.01)
various_accuracy_jacobi_iterations_count_list = []
various_accuracy_jacobi_vectorized_iterations_count_list = []
various_accuracy_seidel_iterations_count_list = []
various_accuracy_seidel_vectorized_iterations_count_list = []
for i in range(0, len(various_accuracy_list)):
    evaluations_results = evaluate(10, various_accuracy_list[i], omega)
    various_accuracy_jacobi_iterations_count_list.append(evaluations_results[0])
    various_accuracy_jacobi_vectorized_iterations_count_list.append(evaluations_results[1])
    various_accuracy_seidel_iterations_count_list.append(evaluations_results[2])
    various_accuracy_seidel_vectorized_iterations_count_list.append(evaluations_results[3])

various_relaxation_param_list = np.arange(1.1, 2, 0.1)
NO = 5
sor_iterations_count = np.zeros((NO, len(various_relaxation_param_list)))
sor_vectorized_iterations_count = np.zeros((NO, len(various_relaxation_param_list)))
for i in range(3, NO + 3):
    for j in range(len(various_relaxation_param_list)):
        evaluations_results = evaluate(i, alpha, various_relaxation_param_list[j])
        sor_iterations_count[i - 3, j] = evaluations_results[4]
        sor_vectorized_iterations_count[i - 3, j] = evaluations_results[5]

plt.plot(shapes_list, jacobi_iterations_count_list, label="jacobi")
plt.plot(shapes_list, seidel_iterations_count_list, label="seidel")
plt.plot(shapes_list, cg_iterations_count_list, label="cg")
plt.legend()
plt.ylabel("Iterations")
plt.xlabel("Shapes")
plt.show()

plt.plot(shapes_list, jacobi_vectorized_iterations_count_list, label="jacobi_vec")
plt.plot(shapes_list, seidel_vectorized_iterations_count_list, label="seidel_vec")
plt.legend()
plt.ylabel("Iterations")
plt.xlabel("Shapes")
plt.show()

plt.plot(various_accuracy_list, various_accuracy_jacobi_iterations_count_list, label="jacobi")
plt.plot(various_accuracy_list, various_accuracy_seidel_iterations_count_list, label="seidel")
plt.legend()
plt.ylabel("Iterations")
plt.xlabel("$\\alpha$")
plt.show()

plt.plot(various_accuracy_list, various_accuracy_jacobi_vectorized_iterations_count_list, label="jacobi_vec")
plt.plot(various_accuracy_list, various_accuracy_seidel_vectorized_iterations_count_list, label="seidel_vec")
plt.legend()
plt.ylabel("Iterations")
plt.xlabel("$\\alpha$")
plt.show()

plt.plot(various_relaxation_param_list, sor_iterations_count[0], label="sor_shape=3")
plt.plot(various_relaxation_param_list, sor_iterations_count[1], label="sor_shape=4")
plt.plot(various_relaxation_param_list, sor_iterations_count[2], label="sor_shape=5")
plt.plot(various_relaxation_param_list, sor_iterations_count[3], label="sor_shape=6")
plt.plot(various_relaxation_param_list, sor_iterations_count[4], label="sor_shape=7")
plt.legend()
plt.ylabel("Iterations")
plt.xlabel("$\\omega$")
plt.show()

plt.plot(various_relaxation_param_list, sor_vectorized_iterations_count[0], label="sorvec_shape=3")
plt.plot(various_relaxation_param_list, sor_vectorized_iterations_count[1], label="sorvec_shape=4")
plt.plot(various_relaxation_param_list, sor_vectorized_iterations_count[2], label="sorvec_shape=5")
plt.plot(various_relaxation_param_list, sor_vectorized_iterations_count[3], label="sorvec_shape=6")
plt.plot(various_relaxation_param_list, sor_vectorized_iterations_count[4], label="sorvec_shape=7")
plt.legend()
plt.ylabel("Iterations")
plt.xlabel("$\\omega$")
plt.show()