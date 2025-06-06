import numpy as np
import Functions
import datetime

# parameters
row_num = 10
array_size = row_num * row_num
islands = list(range(array_size))
distribute_R = True
distribute_C = False

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
    lambd = 2
    R_t_ij = np.random.exponential(lambd, (array_size, array_size)) * R
    R_i = np.random.exponential(lambd, array_size) * R
    R_t_i = [val if idx in set(near_left + near_right) else 0 for idx, val in enumerate(R_i)]
else:
    R_t_ij = np.full((array_size, array_size), R)
    R_i = np.full(array_size, R)
    R_t_i = [val if idx in set(near_left + near_right) else 0 for idx, val in enumerate(R_i)]

# Capacitance Cond
if distribute_C:
    sig = 1 / 10
    Ch = np.random.normal(C, C * sig, size=(row_num, row_num + 1))
    Cv = np.random.normal(C, C * sig, size=(row_num + 1, row_num))
else:
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

date_ = datetime.datetime.now()
strin = "parameters_" + date_.strftime("%Y%m%d") + ".txt"

with open(strin, "w") as f:
    f.write("loop parameters" + "\n")
    f.write("---------------------------------------------" + "\n")
    f.write("row_num : " + str(row_num) + "\n")
    f.write("distribute_R : " + str(distribute_R) + "\n")
    if distribute_R:
        f.write("lambda (exponent) : " + str(lambd) + "\n")
    f.write("distribute_C : " + str(distribute_C) + "\n")
    if distribute_C:
        f.write("sig (normal) : " + str(sig) + "\n")
    f.write("e : " + str(e) + "\n")
    f.write("C : " + str(C) + "\n")
    f.write("Cg : " + str(Cg) + "\n")
    f.write("R : " + str(R) + "\n")
    f.write("Rg : " + str(Rg) + "\n")

print("done")
print(default_dt)
