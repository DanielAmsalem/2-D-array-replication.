import Functions as F
import numpy as np
import matplotlib
from curve_plotter import plot_capacitance_map
from preparation import compute_distributed_C_matrices

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

C_to_Cix_ratio = 100

C=1
sig=0.5
row_num = 10
array_size = row_num ** 2
islands = list(range(array_size))
near_right = islands[(row_num - 1):: row_num]
near_left = islands[0::row_num]

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
Ch[:, 0] /= C_to_Cix_ratio
Ch[:, -1] /= C_to_Cix_ratio

print(Ch)
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

plot_capacitance_map(C_mat, row_num, periodic_y=True, show=True, results_path=int(4))
