import numpy as np
import matplotlib.pyplot as plt
import Functions
import Functions as F
import Conditions as Cond

# Conditions
loops = 1000
row_num = Cond.row_num
array_size = Cond.array_size
islands = list(range(array_size))
R_t = Cond.R_t
Rg = [Cond.Rg] * array_size
Cg = [Cond.Cg] * array_size
default_dt = Cond.default_dt
Tau_inv_matrix = Cond.Tau
increase = False  # true if increasing else decreasing voltage during run

# parameters
e = 1
kB = 1
Volts = abs(e) / Cond.C  # normalized voltage unit
Amp = abs(e) / (Cond.C * Cond.R)  # normalized current unit
Vright = 0
T = 0.01 * e * e / (Cond.C * kB)

# implements increasing\decreasing choice
if increase:
    Vleft = np.linspace(Vright * Volts, (Vright + 4) * Volts, num=100)
else:
    Vleft = np.linspace((Vright + 4) * Volts, Vright * Volts, num=100)

# results matrix, ith column has the ith loop, jth row is the jth step of voltage
cycles = len(Vleft)
I_matrix = np.zeros((loops, cycles))

for loop in range(loops):
    # general Charge distribution vectors
    Qg = np.zeros(array_size)
    n = np.zeros(array_size)

    # vector counting charge flow
    I_vec = np.zeros(cycles)

    for cycle in range(cycles):
        k = 0
        cycle_voltage = Vleft[cycle]
        print("start " + str(cycle_voltage / Volts) + " loop:" + str(loop))

        # starting conditions
        not_in_steady_state = True

        dQ_dt = []

        while not_in_steady_state:
            k += 1

            # define overall reaction rate R, rate vector for both kinds of particles, and a useful index
            V = tuple(F.V_t(n, Qg, cycle_voltage, Vright, Cond.C_inverse, Cond.VxCix))  # find V_i at t =0 for ith island
            R = 0
            rates = []
            reaction_index = []
            Gamma = []

            # dE values for i->j transition
            dEij = np.zeros((array_size, array_size))

            # island to island transition
            for i in islands:
                neighbour_list = Functions.neighbour_list(Cond.row_num, i)

                for j in neighbour_list:
                    n_tag = np.array(list(n))
                    # for isle i to isle j transition
                    if n_tag[i] != 0:
                        n_tag[i] -= e
                        n_tag[j] += e
                    # calculate energy difference due to transition
                    V_new = F.V_t(n_tag, Qg, cycle_voltage, Vright, Cond.C_inverse, Cond.VxCix)
                    dEij[i][j] = e * (V[j] + V_new[j] - V[i] - V_new[i]) / 2

                    # rate for i->j
                    Gamma += [F.gamma(dEij[i][j], T, R_t[i])]
                    R += Gamma[-1]
                    reaction_index += [(i, int(j))]

            # left electrode to island transition:
            dE_left = 0
            near_left = islands[0::row_num]
            for isle in near_left:
                n_tag = np.array(list(n))

                # for ith transition from electrode
                n_tag[isle] += e
                V_new = F.V_t(n_tag, Qg, cycle_voltage, Vright, Cond.C_inverse, Cond.VxCix)
                dE_left = (V[isle] + V_new[isle] - 2 * cycle_voltage) * e / 2

                # rate for V_left->i
                Gamma += [F.gamma(dE_left, T, R_t[isle])]
                R += Gamma[-1]
                reaction_index += [(isle, "from")]

                # for ith transition to electrode
                n_tag[isle] -= 2 * e
                V_new = F.V_t(n_tag, Qg, cycle_voltage, Vright, Cond.C_inverse, Cond.VxCix)
                dE_left = (V[isle] + V_new[isle] - 2 * cycle_voltage) * e / 2

                # rate for V_left->i
                Gamma += [F.gamma(dE_left, T, R_t[isle])]
                R += Gamma[-1]
                reaction_index += [(isle, "to")]

            # similarly, for right side
            dE_right = 0
            near_right = islands[(row_num - 1)::row_num]
            for isle in islands:
                n_tag = np.array(list(n))

                # for ith transition from electrode
                n_tag[isle] += e
                V_new = F.V_t(n_tag, Qg, cycle_voltage, Vright, Cond.C_inverse, Cond.VxCix)
                dE_right = (V[isle] + V_new[isle] - 2 * Vright) * e / 2

                # rate for V_left->i
                Gamma += [F.gamma(dE_right, T, R_t[isle])]
                R += Gamma[-1]
                reaction_index += [(isle, "from")]

                # for ith transition to electrode
                n_tag[isle] -= 2 * e
                V_new = F.V_t(n_tag, Qg, cycle_voltage, Vright, Cond.C_inverse, Cond.VxCix)
                dE_right = (V[isle] + V_new[isle] - 2 * Vright) * e / 2

                # rate for V_left->i
                Gamma += [F.gamma(dE_right, T, R_t[isle])]
                R += Gamma[-1]
                reaction_index += [(isle, "to")]

            # time for reaction
            if R > 1e-10:  # transition occurred
                dt = np.log(1 / np.random.random()) / R

                # pick transition
                r = 0
                x = np.random.random() * R
                chosen_rate = 0
                for i in range(len(Gamma)):
                    if r < x:
                        r += Gamma[i]
                    else:
                        # register transition
                        chosen_rate = Gamma[i]
                        l, m = reaction_index[i]
                        if isinstance(m, int):  # island to island transition
                            n[l] -= 1
                            n[m] += 1
                            break
                        elif isinstance(m, str):  # side - island transition
                            if m == "from":  # side to island
                                n[l] += 1
                                break
                            elif m == "to":  # island to side
                                n[l] -= 1
                                break
                        else:
                            raise NameError

            else:  # no transition occurred
                dt = default_dt
                chosen_rate = 0
                print("transition did not occur")

            Qn = []
            Tau_matrix = np.linalg.inv(Tau_inv_matrix)
            for i in islands:
                summ = 0
                for j in islands:
                    summ += Tau_matrix[i][j] * (e * n[j] + Cond.VxCix(cycle_voltage, Vright)[j]) / Cg[j]
                Qn += [(summ / Rg[i]) - e * n[i] - Cond.VxCix(cycle_voltage, Vright)[i]]

            # calculate dQ/dt = (T^-1)(Qg-Qn)
            Qdot_steady = np.dot(Tau_inv_matrix, Qg - Qn)

            # update Qg, I
            Qg = (Qg + Qdot_steady * dt)
            dQ_dt += [chosen_rate]

            # calculate distance from steady state:
            dist = np.max(np.abs(Qg - Qn))

            if cycle > 1 and k > 50 and dist < Cond.Steady_charge_std:
                print("steady state")
                not_in_steady_state = False

            elif k > 100:
                print(dist)
                not_in_steady_state = False

        I_vec[cycle] = (sum(dQ_dt) / len(dQ_dt))

    I_matrix[loop] = I_vec

I_vec_avg = np.zeros(cycles)  # results vector
for run in I_matrix:
    I_vec_avg += run

I_V = plt.plot(Vleft / Volts, I_vec_avg)
plt.xlabel("Voltage")
plt.ylabel("Current")
if increase:
    plt.title("increasing voltage")
else:
    plt.title("decreasing voltage")
plt.show()
