import copy
import numpy as np
import matplotlib.pyplot as plt
import Functions
import Functions as F
import Conditions as Cond

# Conditions
row_num = Cond.row_num
array_size = Cond.array_size
islands = list(range(array_size))
R_t = Cond.R_t
Rg = [Cond.Rg] * array_size
Cg = [Cond.Cg] * array_size
dt = Cond.default_dt
Tau_inv_matrix = Cond.Tau

# parameters
e = -1.6e-19
kB = 1.38e-23
Volts = e * e / (kB * Cond.C)  # normalized voltage unit
Amp = abs(e) / (Cond.C * Cond.R)  # normalized current unit
Vr = 0
Vplot = np.linspace((Vr+4) * Volts,  (Vr) * Volts, num=1000)
T = 0.01 * e * e / (Cond.C * kB)

# a vector counting charge flow
cycles = len(Vplot)
I_vec = np.zeros(cycles)
Time_vec = []

# general Charge distribution vectors
Qg = np.zeros(array_size)
n = np.zeros(array_size)

for cycle in range(cycles):
    k = 0
    cycle_voltage = Vplot[cycle]
    print("start " + str(cycle_voltage / Volts))

    # starting conditions
    dq_dt_is_not_0 = True

    Time = []
    dQ_dt = []

    while dq_dt_is_not_0:
        k += 1
        # define overall reaction rate R, rate vector for both kinds of particles, and a useful index
        V = tuple(F.V_t(n, Qg, cycle_voltage, Vr))  # find V_i at t =0 for ith island
        R = 0
        rates = []
        reaction_index = []
        Gamma = []

        # dE values for i->j transition
        dEij = np.zeros((array_size, array_size))

        # island to island transition
        for i in islands:
            a = Functions.neighbour_list(Cond.row_num, i)
            for j in a:
                n_tag = np.array(list(n))
                # for isle to isle transition
                if n_tag[i] != 0:
                    n_tag[i] -= e
                    n_tag[j] += e
                V_new = F.V_t(n_tag, Qg, cycle_voltage, Vr)
                dEij[i][j] = -e * (V[j] + V_new[j] - V[i] - V_new[i]) / 2

                # rate for i->j
                Gamma += [F.gamma(dEij[i][j], T, R_t[i])]
                R += Gamma[-1]
                reaction_index += [(i, int(j))]

        # left electrode to island transition:
        dE_left = 0
        near_left = islands[0::row_num]
        for i in list(range(row_num)):
            n_tag = np.array(list(n))

            # for ith transition from electrode
            n_tag[i] += e
            V_new = F.V_t(n_tag, Qg, cycle_voltage, Vr)
            dE_left = (V[near_left[i]] + V_new[near_left[i]] - 2 * cycle_voltage) * e / 2

            # rate for V_left->i
            Gamma += [F.gamma(dE_left, T, R_t[i])]
            R += Gamma[-1]
            reaction_index += [(near_left[i], "from")]

            # for ith transition to electrode
            n_tag[i] -= 2 * e
            V_new = F.V_t(n_tag, Qg, cycle_voltage, Vr)
            dE_left = (V[near_left[i]] + V_new[near_left[i]] - 2 * cycle_voltage) * e / 2

            # rate for V_left->i
            Gamma += [F.gamma(dE_left, T, R_t[i])]
            R += Gamma[-1]
            reaction_index += [(near_left[i], "to")]

        # similarly, for right side
        dE_right = 0
        near_right = islands[(row_num - 1)::row_num]
        for i in list(range(row_num)):
            n_tag = np.array(list(n))

            # for ith transition from electrode
            n_tag[i] += e
            V_new = F.V_t(n_tag, Qg, cycle_voltage, Vr)
            dE_right = (V[near_right[i]] + V_new[near_right[i]] - 2 * Vr) * e / 2

            # rate for V_left->i
            Gamma += [F.gamma(dE_right, T, R_t[i])]
            R += Gamma[-1]
            reaction_index += [(near_right[i], "from")]

            # for ith transition to electrode
            n_tag[i] -= 2 * e
            V_new = F.V_t(n_tag, Qg, cycle_voltage, Vr)
            dE_right = (V[near_right[i]] + V_new[near_right[i]] - 2 * Vr) * e / 2

            # rate for V_left->i
            Gamma += [F.gamma(dE_right, T, R_t[i])]
            R += Gamma[-1]
            reaction_index += [(near_right[i], "to")]

        # time for reaction
        if R > 1e-10:  # transition occurred
            dt = np.log(1 / np.random.random()) / R
            Time += [dt]

            # pick transition
            r = 0
            x = np.random.random() * R
            chosen_rate = 0
            # print(Gamma)
            for i in list(range(len(Gamma))):
                if r < x:
                    r += Gamma[i]
                    chosen_rate = Gamma[i]
                else:
                    # register transition
                    l, m = reaction_index[i]
                    # print((l, m))
                    if isinstance(m, int):  # island to island transition
                        # print("i")
                        n[l] -= 1
                        n[m] += 1
                        break
                    elif isinstance(m, str):  # side - island transition
                        if m == "from":  # side to island
                            # print(m)
                            n[l] += 1
                            break
                        elif m == "to":  # island to side
                            # print(m)
                            n[l] -= 1
                            break
                    else:
                        raise NameError

        else:
            Time += [dt]
            chosen_rate = 0
            print("transition did not occur")

        Qn = []
        for i in islands:
            summ = 0
            for j in islands:
                summ += Tau_inv_matrix[i][j] * e * n[j] + Cond.VxCix(cycle_voltage, Vr)[j] / Cg[j]
            Qn += [(summ / Rg[i]) - e * n[i] + Cond.VxCix(cycle_voltage, Vr)[i]]

        # calculate dQ/dt = T-1(Qg-Qn)
        Qdot = np.dot(np.linalg.inv(Tau_inv_matrix), Qg - Qn)

        # update Qg, V
        Qg = (Qg + Qdot * dt)
        dQ_dt += [chosen_rate]

        if k > 100:
            print(k)
            dq_dt_is_not_0 = False

    I_vec[cycle] = (sum(dQ_dt) / len(dQ_dt))

print("bruh")
I_V = plt.plot(Vplot / Volts, I_vec / Amp)
plt.xlabel("Voltage")
plt.ylabel("Current")
plt.title("decreasing voltage C=<c>*10^14")
plt.show()
