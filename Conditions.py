import numpy as np
import math
import random

# parameters
row_num = 4
array_size = row_num * row_num

# define tunneling parameters
R_t = [math.exp(random.random()) * 10 for i in range(array_size)]
R = sum(R_t) / len(R_t)
C = 1
Cij = np.zeros((array_size, array_size))

# define relaxation time parameters
Cg = 20 * C
Rg = 1000 * R
Steady_charge_std = 1


def return_neighbours(n, I, J):  # Return positions of neighbours of (i,j) in nxn matrix
    I, J = int(I), int(J)
    neighbours = []
    if J + 1 < n:  # right neighbour
        neighbours += [(I, J + 1)]
    if J > 0:  # left neighbour
        neighbours += [(I, J - 1)]
    if I + 1 < n:  # down neighbour
        neighbours += [(I + 1, J)]
    if I > 0:  # up neighbour
        neighbours += [(I - 1, J)]

    return neighbours


for i in range(array_size):
    for j in range(array_size):
        y_value = i % row_num  # y position of ith SC
        x_value = (i - i % row_num) / row_num  # x position of ith SC

        neighbour = (int((j - j % row_num) / row_num), int(j % row_num))  # coordinates of jth SC

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

C_inverse = tuple(np.linalg.inv(Cij))  # define inverse

# define capacitance Cix
Cix = np.zeros(array_size)
for i in range(array_size):
    if i < math.sqrt(array_size):
        Cix[i] = abs(np.random.normal(C, C / 10))
    elif i > array_size - math.sqrt(array_size) - 1:
        Cix[i] = abs(np.random.normal(C, C / 10))


def VxCix(Vl, Vr):
    _VxCix = np.zeros(array_size)
    for u in range(array_size):
        if u < math.sqrt(array_size):
            _VxCix[u] = Cix[u] * Vl
        elif u > array_size - math.sqrt(array_size) - 1:
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
default_dt = -0.1/np.min(eig_val_tau)

