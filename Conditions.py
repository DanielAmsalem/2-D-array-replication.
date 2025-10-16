import numpy as np
import datetime

# parameters
row_num = 10
array_size = row_num * row_num
islands = list(range(array_size))
distribute_R = True
distribute_C = True

# define tunneling parameters
e = 1
kB = 1
R = 10
C = 1
T0 = 0.001 * e * e / (C * kB)

# define relaxation time parameters
Cg = 10 * C
Rg = 100 * R
Cg_list = [Cg] * array_size
Rg_list = [Rg] * array_size
tg = Cg * Rg

near_right = islands[(row_num - 1)::row_num]
near_left = islands[0::row_num]

if distribute_R:
    stdR = 0.9 * R
    R_t_ij = 2 ** np.random.uniform(low=np.log2(max(R - stdR, 0.01)),
                                    high=np.log2(R + stdR), size=(array_size, array_size))
    R_i = 2 ** np.random.uniform(low=np.log2(max(R - stdR, 0.01)),
                                 high=np.log2(R + stdR), size=array_size)
    R_t_i = [val if idx in set(near_left + near_right) else 0 for idx, val in enumerate(R_i)]
else:
    R_t_ij = np.full((array_size, array_size), R)
    R_i = np.full(array_size, R)
    R_t_i = [val if idx in set(near_left + near_right) else 0 for idx, val in enumerate(R_i)]

# Capacitance Cond
Cix = np.zeros(array_size)
if distribute_C:
    sig = 0.5
    Ch = np.random.normal(0, C * sig, size=(row_num, row_num + 1))
    Cv = np.random.normal(0, C * sig, size=(row_num + 1, row_num))

    all_Cs = np.concatenate([Ch.ravel(), Cv.ravel()])
    if np.all(all_Cs >= 0):
        pass
    else:
        min_val = -np.min(all_Cs) + 0.1

    # Ch, Cv = Ch + max(min_val, C), Cv + max(min_val, C)
    Ch, Cv = Ch + min_val + C, Cv + min_val + C
    all_Cs = np.concatenate([Ch.ravel(), Cv.ravel()])

    Cl = np.random.normal(0, C * sig/3, size=(1, array_size))
    Cr = np.random.normal(0, C * sig/3, size=(1, array_size))

    side_Cs = np.concatenate([Cl.ravel(), Cr.ravel()])
    if np.all(side_Cs >= 0):
        pass
    else:
        min_val = -np.min(side_Cs) + 0.1
    # Cl, Cr = Cl + max(min_val, C), Cr + max(min_val, C/2)
    Cl, Cr = Cl + min_val + C, Cr + min_val + C/2
    side_Cs = np.concatenate([Cl.ravel(), Cr.ravel()])

    Cix = np.zeros(array_size)
    for i in near_left:
        Cix[i] = Cl[0][i]
    for i in near_right:
        Cix[i] = Cr[0][i]

    for c in np.concatenate([side_Cs.ravel(), all_Cs.ravel()]):
        if c < 0:
            print(c)
            raise ValueError("Negative")

else:
    Ch = np.random.normal(C, 0, size=(row_num, row_num + 1))
    Cv = np.random.normal(C, 0, size=(row_num + 1, row_num))

    Cix = np.zeros(array_size)
    for i in near_left:
        Cix[i] = np.random.normal(C, 0)
    for i in near_right:
        Cix[i] = np.random.normal(C/2, 0)

diagonal = Ch[:, :-1] + Ch[:, 1:] + Cv[:-1, :] + Cv[1:, :]
second_diagonal = np.copy(Ch[:, 1:])
second_diagonal[:, -1] = 0
second_diagonal = second_diagonal.flatten()
second_diagonal = second_diagonal[:-1]
n_diagonal = np.copy(Cv[1:-1, :])
C_mat = np.diagflat(diagonal) - np.diagflat(second_diagonal, k=1) - np.diagflat(second_diagonal, k=-1) - \
        np.diagflat(n_diagonal, k=row_num) - np.diagflat(n_diagonal, k=-row_num)
C_inverse = np.linalg.inv(C_mat)  # define inverse


def VxCix(Vl, Vr):
    _VxCix = np.zeros(array_size)
    for u in near_left:
        _VxCix[u] = Cix[u] * Vl
    for u in near_right:
        _VxCix[u] = Cix[u] * Vr
    return np.array(_VxCix)


# define tau matrix
res = C_inverse + np.diagflat([1 / Cg] * array_size)
a = np.array([Rg] * array_size)  # flattening to coloumn
reshaped = a.reshape((a.size, 1))
Tau_inv = -res / np.repeat(reshaped, res.shape[1], axis=1)

# tau matrix properties, eigenvalues and default time step
InvTauEigenValues, InvTauEigenVectors = np.linalg.eig(Tau_inv)
InvTauEigenVectorsInv = np.linalg.inv(InvTauEigenVectors)
default_dt = -0.1 / np.min(InvTauEigenValues)  # time in which Qg don't change much
timeStep = -2 / np.max(InvTauEigenValues)  # time for steady state

# Cg matrix, and Qn calculation
Tau = np.linalg.inv(Tau_inv)
matrixQnPart = Tau / (Cg * Rg) - np.eye(Tau.shape[0])

date_ = datetime.datetime.now()
strin = "parameters_" + date_.strftime("%Y%m%d, %Hh%Mm%Ss") + ".txt"  #

with open(strin, "w") as f:
    f.write("loop parameters" + "\n")
    f.write("---------------------------------------------" + "\n")
    f.write("row_num : " + str(row_num) + "\n")
    f.write("distribute_R : " + str(distribute_R) + "\n")
    if distribute_R:
        f.write("stdR (exponent) : " + str(stdR) + "\n")
    f.write("distribute_C : " + str(distribute_C) + "\n")
    if distribute_C:
        f.write("sig (normal) : " + str(sig) + "\n")
    f.write("e : " + str(e) + "\n")
    f.write("C : " + str(C) + "\n")
    f.write("Cg : " + str(Cg) + "\n")
    f.write("R : " + str(R) + "\n")
    f.write("Rg : " + str(Rg) + "\n")
    f.write("default_dt : " + str(default_dt) + "\n")
    f.write("time step : " + str(timeStep) + "\n")
    f.write("---------------------------------------------" + "\n")
    f.write("\n")
    f.write("\n")

print("done")
print(default_dt)
print(timeStep)
