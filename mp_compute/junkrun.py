import Functions as F
import numpy as np
import matplotlib
from curve_plotter import plot_capacitance_map
from gamma_functions import execute_transition
from preparation import compute_distributed_C_matrices

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

mean_Cg = 10
mean_Rg = 1000
C = 1
sig = 0.5
row_num = 3
array_size = row_num ** 2
islands = list(range(array_size))
near_right = islands[(row_num - 1):: row_num]
near_left = islands[0::row_num]

np.random.seed(0)
Ch = np.random.normal(0, sig, size=(row_num, row_num + 1))
Cv = np.random.normal(0, sig, size=(row_num + 1, row_num))

all_Cs = np.concatenate([Ch.ravel(), Cv.ravel()])
if np.all(all_Cs >= 0):
    pass
else:
    min_val = -np.min(all_Cs) + 0.1

# Ch, Cv = Ch + max(min_val, C), Cv + max(min_val, C)
Ch, Cv = Ch + min_val + C, Cv + min_val + C

# update Cix to be a factor Cix_ration smaller:
Ch[:, 0] /= 1
Ch[:, -1] /= 1

print("#################################################")

# get Cix in sparse form.
Cl = Ch[:, :-1].copy()
Cl[:, 1:] = 0
Cr = Ch[:, 1:].copy()
Cr[:, :-1] = 0

all_Cs = np.concatenate([Ch.ravel(), Cv.ravel()])
side_Cs = np.concatenate([Cl.ravel(), Cr.ravel()])

Cix = np.zeros(array_size)
for i in near_left:
    Cix[i] = Cl[i // row_num][0]
for i in near_right:
    Cix[i] = Cr[i // row_num][0]

diagonal = Ch[:, :-1] + Ch[:, 1:] + Cv[:-1, :] + Cv[1:, :]
second_diagonal = np.copy(Ch[:, 1:])
second_diagonal[:, -1] = 0
second_diagonal = second_diagonal.flatten()
second_diagonal = second_diagonal[:-1]
n_diagonal = np.copy(Cv[1:-1, :])
C_mat = (
        np.diagflat(diagonal)
        - np.diagflat(second_diagonal, k=1)
        - np.diagflat(second_diagonal, k=-1)
        - np.diagflat(n_diagonal, k=row_num)
        - np.diagflat(n_diagonal, k=-row_num)
)

offset = (row_num - 1) * row_num
wrap_vals = Cv[0, :]
wrap_flat = wrap_vals.flatten()

C_mat -= np.diagflat(wrap_flat, k=offset)
C_mat -= np.diagflat(wrap_flat, k=-offset)

#plot_capacitance_map(C_mat, row_num, periodic_y=True, show=True, results_path=int(4))
from preparation import define_tau_inverse_matrix

C_inverse = np.linalg.inv(C_mat)

Tau_inv = define_tau_inverse_matrix(C_inverse, mean_Cg, mean_Rg, array_size=array_size)
InvTauEigenValues, InvTauEigenVectors = np.linalg.eig(Tau_inv)
InvTauEigenVectorsInv = np.linalg.inv(InvTauEigenVectors)
default_dt = -0.1 / np.min(InvTauEigenValues)  # time in which Qg don't change much
timeStep = -2 / np.max(InvTauEigenValues)
Tau = np.linalg.inv(Tau_inv)

Qg = np.zeros(array_size)
n = np.zeros(array_size)

A = C_mat
plt.imshow(A, cmap='viridis', origin='lower')
plt.colorbar(label="Value")
plt.title("Heatmap of A")
plt.xlabel("j")
plt.ylabel("i")
plt.gca().invert_yaxis()
plt.show()
exit()

def developQ(Q, dt, Qnn):
    b = -Tau_inv.dot(Qnn)

    # exponent for time step
    exponent = np.exp(InvTauEigenValues * dt)

    # basis change
    Q_in_eigenbasis, b = InvTauEigenVectorsInv.dot(Q), InvTauEigenVectorsInv.dot(b)

    # solution in time
    Q_new_in_eigenbasis = (exponent * Q_in_eigenbasis) + (b / InvTauEigenValues) * (exponent - 1)

    # revert to old basis
    return InvTauEigenVectors.dot(Q_new_in_eigenbasis)


def developQpaper(Q, dt):
    res = -C_inverse.dot(n + VxCix) / mean_Rg
    b = res

    # exponent for time step
    exponent = np.exp(InvTauEigenValues * dt)
    # basis change
    Q_in_eigenbasis, b = InvTauEigenVectorsInv.dot(Q), InvTauEigenVectorsInv.dot(b)

    # solution in time
    Q_new_in_eigenbasis = (exponent * Q_in_eigenbasis) + (b / InvTauEigenValues) * (exponent - 1)

    # revert to old basis
    return InvTauEigenVectors.dot(Q_new_in_eigenbasis)


VxCix = F.get_VxCix(0, 0, array_size, near_left, near_right, Cix)
print(f"V : {F.getVoltage(n,Qg,C_inverse,VxCix,1)}")
matrixQnPart = -Tau / (mean_Cg * mean_Rg) - np.eye(Tau.shape[0])
n = np.random.randint(0, 5, size=n.shape[0])
n = np.array([4,0,0,
              3,0,0,
              1,0,0])
n_prime = n
Qn = matrixQnPart.dot(n_prime)


Qg_me = developQ(Qg, default_dt, Qn)
Qg_paper = developQpaper(Qg, default_dt)
print("########################DOWN IS MY Qg##########################")
print(n)
print(f"V : {F.getVoltage(n,Qg,C_inverse,VxCix,1)}")
print("########################DOWN IS PAPER QG##########################")
print(Qg_paper)
#print(f"V : {F.getVoltage(n,Qg_paper,C_inverse,VxCix,1)}")
print("########################DOWN IS DIFF##########################")
for i in np.abs(Qg_me - Qg_paper):
    if i > 1e-10:
        print(i)



