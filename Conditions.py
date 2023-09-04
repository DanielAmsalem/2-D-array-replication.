import numpy as np
import math
import random
from Functions import return_neighbours

# parameters
row_num = 4
array_size = row_num * row_num
islands = list(range(array_size))

# define tunneling parameters
R = 10
C = 1e-5
R_t = [math.exp(random.random()) * R for i in range(array_size)]

Cij = np.zeros((array_size, array_size))

# define relaxation time parameters
Cg = 20 * C
Rg = 1000 * R

for i in range(array_size):
    x_value = i % row_num  # y position of ith SC
    y_value = (i - i % row_num) / row_num  # x position of ith SC
    for j in range(array_size):
        # coordinates of jth SC
        neighbour_x = j % row_num
        neighbour_y = (j - j % row_num) / row_num

        neighbour = (neighbour_x, neighbour_y)
        neighbours_ij = return_neighbours(row_num, x_value, y_value)  # return coordinates of ith's neighbours

        if neighbour in neighbours_ij:
            Cij[i][j] = -abs(np.random.normal(C, C / 10))  # [C]ij = -Cij = -Cji
            Cij[j][i] = Cij[i][j]

for i in range(array_size):
    y_value = i % row_num  # y position of ith SC
    x_value = (i - i % row_num) / row_num  # x position of ith SC

    neighbours_ij = return_neighbours(row_num, x_value, y_value)  # return coordinates of ith's neighbour

    for neighbour in neighbours_ij:
        Cij[i][i] += -(Cij[i][neighbour[0] * row_num + neighbour[1]])

    Cij[i][i] += Cg

C_inverse = np.linalg.inv(Cij)  # define inverse

near_right = islands[(row_num - 1)::row_num]
near_left = islands[0::row_num]

# define capacitance Cix
Cix = np.zeros(array_size)
for i in near_left:
    Cix[i] = abs(np.random.normal(C, C / 10))
for i in near_right:
    Cix[i] = abs(np.random.normal(C, C / 10))
print("done")


def VxCix(Vl, Vr):
    _VxCix = np.zeros(array_size)
    for u in near_left:
        _VxCix[u] = Cix[u] * Vl
    for u in near_right:
        _VxCix[u] = Cix[u] * Vr
    return _VxCix


# define tau matrix
tau_inv_matrix = np.zeros((array_size, array_size))
for i in range(array_size):
    for K in range(array_size):
        if i == K:
            tau_inv_matrix[i][K] = (C_inverse[i][K] + (1 / Cg)) / Rg
        else:
            tau_inv_matrix[i][K] = C_inverse[i][K] / Rg

Tau = tuple(tau_inv_matrix)  # for conditions

eig_val_tau, eig_vec_tau = np.linalg.eig(tau_inv_matrix)
eig_val_tau = eig_val_tau.reshape(1, eig_val_tau.size)  # turns array to column vector
default_dt = -0.1 / np.min(eig_val_tau)
