import numpy as np
import math
import random
from Functions import return_neighbours
import Functions

# parameters
row_num = 4
array_size = row_num * row_num
islands = list(range(array_size))
distribute_R = False

# define tunneling parameters
e = Functions.e
kB = Functions.kB
R = 10
C = 1

if distribute_R:
    R_t = [math.exp(random.random()) * R for i in range(array_size)]
else:
    R_t = [R for i in range(array_size)]

Cij = np.zeros((array_size, array_size))

# define relaxation time parameters
Cg = 20 * C
Rg = 1000 * R

#Capacitance Cond
Ch = np.random.normal(5*C,C/10, size=(row_num, row_num+1))
Cv = np.random.normal(C,C/10, size=(row_num + 1, row_num))

diagonal = Ch[:, :-1] + Ch[:, 1:] + Cv[:-1, :] + Cv[1:, :]
second_diagonal = np.copy(Ch[:, 1:])
second_diagonal[:, -1] = 0
second_diagonal = second_diagonal.flatten()
second_diagonal = second_diagonal[:-1]
n_diagonal = np.copy(Cv[1:-1, :])
C_mat = np.diagflat(diagonal) - np.diagflat(second_diagonal, k=1) - np.diagflat(second_diagonal, k=-1) - \
        np.diagflat(n_diagonal, k=row_num) - np.diagflat(n_diagonal, k=-row_num)
invC = np.linalg.inv(C_mat)
C_inverse = np.linalg.inv(C_mat + np.diagflat([Cg]*array_size)) # define inverse

near_right = islands[(row_num - 1)::row_num]
near_left = islands[0::row_num]

# define capacitance Cix
Cix = np.zeros(array_size)
for i in near_left:
    Cix[i] = abs(np.random.normal(C, C / 10))
for i in near_right:
    Cix[i] = abs(np.random.normal(C, C / 10))
print("done")


def VxCix(Vl, Vr, V):
    _VxCix = np.zeros(array_size)
    for u in near_left:
        _VxCix[u] = Cix[u] * (V[u])
    for u in near_right:
        _VxCix[u] = Cix[u] * (V[u])
    return np.array(_VxCix)


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
