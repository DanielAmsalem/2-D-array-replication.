import numpy as np
import Functions

# parameters
row_num = 2
array_size = row_num * row_num
islands = list(range(array_size))
distribute_R = True

# define tunneling parameters
e = Functions.e
kB = Functions.kB
R = 10
C = 1

# define relaxation time parameters
Cg = 10 * C
Rg = 100 * R

near_right = islands[(row_num - 1)::row_num]
near_left = islands[0::row_num]

if distribute_R:
    R_t_ij = np.random.exponential(1, (array_size, array_size)) * R
    R_i = np.random.exponential(1, array_size) * R
    R_t_i = [val if idx in set(near_left + near_right) else 0 for idx, val in enumerate(R_i)]
else:
    R_t_ij = np.full((array_size, array_size), R)
    R_i = np.full(array_size, R)
    R_t_i = [val if idx in set(near_left + near_right) else 0 for idx, val in enumerate(R_i)]

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


Cix = np.zeros(array_size)
for i in near_left:
    Cix[i] = abs(np.random.normal(C, 0))
for i in near_right:
    Cix[i] = abs(np.random.normal(C / 2, 0))


def VxCix(Vl, Vr):
    _VxCix = np.zeros(array_size)
    for u in near_left:
        _VxCix[u] = Cix[u] * Vl
    for u in near_right:
        _VxCix[u] = Cix[u] * Vr
    return np.array(_VxCix)


# define tau matrix
res = C_inverse + np.diagflat([1 / Cg] * array_size)
Tau_inv = -res / np.repeat(Functions.flattenToColumn(np.array([Rg] * array_size)), res.shape[1], axis=1)

# tau matrix properties, eigenvalues and default time step
InvTauEigenValues, InvTauEigenVectors = np.linalg.eig(Tau_inv)
InvTauEigenVectorsInv = np.linalg.inv(InvTauEigenVectors)
default_dt = -0.1 / np.min(InvTauEigenValues)

# Cg matrix, and Qn calculation
Tau = np.linalg.inv(Tau_inv)
matrixQnPart = Tau / (Cg * Rg) - np.eye(Tau.shape[0])
print("done")
print(default_dt)


