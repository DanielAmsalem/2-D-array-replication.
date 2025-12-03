import Functions as F
import numpy as np
import matplotlib
from curve_plotter import plot_capacitance_map
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


###############HEAT MAP STUUFF
# n = 10
# islands = list(range(n ** 2))
# near_right = islands[(n - 1):: n]
# near_left = islands[0::n]
# i = 9
#
# gamma_list = []
# rec_idx = []
#
# gamma_list_full = [(8, 9, 1),
#                    (18,19,2),
#                    (2, 3, 2),
#                    (4, 3, 3),
#                    (50, 60, 2),
#                    (16, 6, 1),
#                    (98,99, 1)]
# for listing in gamma_list_full:
#     l, m, g = listing
#     gamma_list.append(g)
#     rec_idx.append((l, m))
#
# print(F.neighbour_list(n, i))
# Jx, Jy = F.Get_current_map(gamma_list, rec_idx, near_right, near_left, n)
#
# # create grid
# Y, X = np.mgrid[0:n, 0:(n + 1)]
# plt.figure(figsize=(6, 6))
# # create x current vecs at each point
# plt.quiver(X + 0.5, Y + 0.5, Jx, np.zeros((n, n + 1)),
#            np.sqrt(Jx ** 2 + Jy ** 2),  # color by magnitude
#            scale=np.abs(Jx).max(), scale_units='xy', angles='xy',
#            cmap='coolwarm')
# # create y current vecs at each point
# plt.quiver(X + 0.5, Y + 0.5, np.zeros((n, n + 1)), Jy,
#            np.sqrt(Jx ** 2 + Jy ** 2),  # color by magnitude
#            scale=np.abs(Jy).max(), scale_units='xy', angles='xy',
#            cmap='coolwarm')
# plt.grid(True, color="lightgray", alpha=0.5)
# plt.xticks(list(range(n + 2)), ["Vleft"] + [str(i) for i in range(n)] + ["Vright"])
# plt.yticks(range(n + 1))
# print(Jx)
# Jx_form = []
# k = 0
# for i in range(Jx.shape[0]):
#     Jx_pos = []
#     for j in range(Jx.shape[1]):
#         Jx_pos += [(k, int(Jx[i, j]))]
#         k += 1
#     Jx_form.append(Jx_pos)
# print(Jx_form)
# plt.show()

######################################### C INVERSE CHECKER
row_num = 10
sig = 0
C = 1
array_size = row_num**2
Ch = np.random.normal(0, sig, size=(row_num, row_num + 1))
Cv = np.random.normal(0, sig, size=(row_num + 1, row_num))

all_Cs = np.concatenate([Ch.ravel(), Cv.ravel()])
if np.all(all_Cs >= 0):
    min_val = 0
else:
    min_val = -np.min(all_Cs) + 0.1

# Ch, Cv = Ch + max(min_val, C), Cv + max(min_val, C)
Ch, Cv = Ch + min_val + C, Cv + min_val + C
all_Cs = np.concatenate([Ch.ravel(), Cv.ravel()])

Cl = np.random.normal(0, sig / 3, size=(1, array_size))
Cr = np.random.normal(0, sig / 3, size=(1, array_size))

side_Cs = np.concatenate([Cl.ravel(), Cr.ravel()])

if np.all(side_Cs >= 0):
    min_val = 0
else:
    min_val = -np.min(side_Cs) + 0.1

# Cl, Cr = Cl + max(min_val, C), Cr + max(min_val, C/2)
Cl, Cr = Cl + min_val + C, Cr + min_val + C
side_Cs = np.concatenate([Cl.ravel(), Cr.ravel()])

for c in np.concatenate([side_Cs.ravel(), all_Cs.ravel()]):
    if c < 0:
        print(c)
        raise ValueError("Negative")

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
        - np.diagflat(n_diagonal, k=-row_num))

plot_capacitance_map(C_mat, row_num, True)

C_inv = np.linalg.inv(C_mat)

plot_capacitance_map(C_inv, row_num, True)
