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
    R_t = np.array([math.exp(random.random()) * R for i in range(array_size)])
else:
    R_t = np.array([R for i in range(array_size)])

Cij = np.zeros((array_size, array_size))

# define relaxation time parameters
Cg = 20 * C
Rg = 1000 * R

# Capacitance Cond
Ch = np.random.normal(C, 0, size=(row_num, row_num + 1))
Cv = np.random.normal(C, 0, size=(row_num + 1, row_num))

diagonal = Ch[:, :-1] + Ch[:, 1:] + Cv[:-1, :] + Cv[1:, :]
second_diagonal = np.copy(Ch[:, 1:])
second_diagonal[:, -1] = 0
second_diagonal = second_diagonal.flatten()
second_diagonal = second_diagonal[:-1]
n_diagonal = np.copy(Cv[1:-1, :])
C_mat = np.diagflat(diagonal) - np.diagflat(second_diagonal, k=1) - np.diagflat(second_diagonal, k=-1) - \
        np.diagflat(n_diagonal, k=row_num) - np.diagflat(n_diagonal, k=-row_num)
invC = np.linalg.inv(C_mat)
C_inverse = np.linalg.inv(C_mat + np.diagflat([Cg] * array_size))  # define inverse

near_right = islands[(row_num - 1)::row_num]
near_left = islands[0::row_num]

# define capacitance Cix
if array_size == 1:
    Cix = np.array([2 * C, C])
else:
    Cix = np.zeros(array_size)
    for i in near_left:
        Cix[i] = abs(np.random.normal(C, 0))
    for i in near_right:
        Cix[i] = abs(np.random.normal(C / 2, 0))

# define tau matrix
res = C_inverse + np.diagflat([1 / Cg] * array_size)
Tau_inv = -res / np.repeat(Functions.flattenToColumn(np.array([Rg] * array_size)), res.shape[1], axis=1)

# tau matrix properties, eigenvalues and default time step
InvTauEigenValues, InvTauEigenVectors = np.linalg.eig(Tau_inv)
InvTauEigenVectorsInv = np.linalg.inv(InvTauEigenVectors)
default_dt = -0.1 / np.min(InvTauEigenValues)

# Cg matrix, and Qn calculation
Tau = np.linalg.inv(Tau_inv)
CGMat = np.full((array_size, array_size), 1 / Cg)
matrixQnPart = Tau * CGMat - np.eye(Tau.shape[0])
print("done")


def VxCix(Vl, Vr):
    _VxCix = np.zeros(array_size)
    for u in near_left:
        _VxCix[u] = Cix[u] * Vl
    for u in near_right:
        _VxCix[u] = Cix[u] * Vr
    return np.array(_VxCix)
