import numpy as np
import matplotlib.pyplot as plt
import Functions as F
import Conditions as Cond
import copy

# Conditions
Enable, Print = False, False
loops = 1000
steps = 1000
row_num = Cond.row_num
array_size = Cond.array_size
islands = list(range(array_size))
near_left = Cond.near_left
near_right = Cond.near_right
R_t_ij = Cond.R_t_ij
R_t_i = Cond.R_t_i
Rg = np.array([Cond.Rg] * array_size)
Cg = np.array([Cond.Cg] * array_size)
default_dt = Cond.default_dt
Tau = Cond.Tau
C_inv = copy.copy(Cond.C_inverse)
increase = True  # true if increasing else decreasing voltage during run

# parameters
e = Cond.e
kB = Cond.kB
Volts = abs(e) / Cond.C  # normalized voltage unit
Amp = abs(e) / (Cond.C * Cond.R)  # normalized current unit
Vright = 0
T = 0.001 * e * e / (Cond.C * kB)

# Gillespie parameter, KS statistic value for significance
KS_boundary = e * 1e-2
Steady_state_rep = 100

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
    Qg, Q_avg, Q_var = np.zeros(array_size), np.zeros(array_size), np.zeros(array_size)
    n, n_avg, n_var = np.zeros(array_size), np.zeros(array_size), np.zeros(array_size)
    I_avg, I_var = 0, 0

    # vector counting charge flow
    I_vec = np.zeros(cycles)

    for cycle in range(cycles):
        k = 0
        zero_curr_steady_state_counter = 0
        not_decreasing = 0
        decreasing = 0

        cycle_voltage = float(Vleft[cycle])
        print("start " + str(cycle_voltage / Volts) + " loop:" + str(loop))

        # starting conditions
        not_in_steady_state = True
        t = 0

        while not_in_steady_state:
            # update number of reactions and voltage from last loop
            k += 1
            l, m = None, None

            VxCix = Cond.VxCix(cycle_voltage, Vright)
            V = F.getVoltage(n, Qg, C_inv, VxCix)  # find V_i for ith island

            # define overall reaction rate R, rate vector, and a useful index
            R = 0
            reaction_index = []
            Gamma = []

            # dE values for i->j transition
            dEij = np.zeros((array_size, array_size))

            # island i to island j transition
            for i in islands:
                # if island i is empty pass over
                if n[i] == 0 or array_size == 1:
                    continue

                # else calculate transition rate to jth island
                neighbour_list = F.neighbour_list(Cond.row_num, i)
                for j in neighbour_list:
                    n_tag = copy.copy(n)

                    # for isle i to isle j transition
                    n_tag[i] -= e
                    n_tag[j] += e

                    # calculate energy difference due to transition
                    V_new = F.getVoltage(n_tag, Qg, C_inv, VxCix)
                    dEij[i][j] = e * (V[j] + V_new[j] - V[i] - V_new[i]) / 2

                    # dEij must be negative for transition i->j
                    if dEij[i][j] < 0:
                        Gamma += [F.gamma(dEij[i][j], T, R_t_ij[i][j])]
                        R += Gamma[-1]
                        reaction_index += [(i, j)]

            # left electrode to island transition:
            for isle in near_left:
                n_tag = copy.copy(n)

                # for ith transition from electrode
                n_tag[isle] += e

                V_new = F.getVoltage(n_tag, Qg, C_inv, VxCix)
                dE_left = (V[isle] + V_new[isle] - 2 * cycle_voltage) * e / 2

                # rate for V_left->i
                if dE_left < 0:
                    Gamma += [F.gamma(dE_left, T, R_t_i[isle])]
                    R += Gamma[-1]
                    reaction_index += [(isle, "from")]

                # return "borrowed electron" used for calculation
                n_tag[isle] -= e

                # for ith transition to electrode there must be at least one electron at isle i
                if n_tag[isle] / e >= 1:

                    # for ith transition from electrode
                    n_tag[isle] -= e

                    V_new = F.getVoltage(n_tag, Qg, C_inv, VxCix)
                    dE_left = (2 * cycle_voltage - V[isle] - V_new[isle]) * e / 2

                    # rate for i->V_left
                    if dE_left < 0:
                        Gamma += [F.gamma(dE_left, T, R_t_i[isle])]
                        R += Gamma[-1]
                        reaction_index += [(isle, "to")]

            # similarly, for right side
            for isle in near_right:
                n_tag = copy.copy(n)

                # for ith transition from electrode
                n_tag[isle] += e

                V_new = F.getVoltage(n_tag, Qg, C_inv, VxCix)
                dE_right = (V[isle] + V_new[isle] - 2 * Vright) * e / 2

                # rate for V_right->i
                if dE_right < 0:
                    Gamma += [F.gamma(dE_right, T, R_t_i[isle])]
                    R += Gamma[-1]
                    reaction_index += [(isle, "from")]

                # return "borrowed electron" used for calculation
                n_tag[isle] -= e

                # for ith transition to electrode
                if n_tag[isle] / e >= 1:
                    # for ith transition to electrode
                    n_tag[isle] -= e

                    V_new = F.getVoltage(n_tag, Qg, C_inv, VxCix)
                    dE_right = (2 * Vright - V[isle] - V_new[isle]) * e / 2

                    # rate for i->V_right
                    if dE_right < 0:
                        Gamma += [F.gamma(dE_right, T, R_t_i[isle])]
                        R += Gamma[-1]
                        reaction_index += [(isle, "to")]

            # transition occurred, limit for R is the typical ground drain current
            if R > abs(cycle_voltage / (e * Cond.Rg)):
                zero_curr_steady_state_counter = 0

                # typical interaction time
                dt = np.log(1 / np.random.random()) / R

                # picking a specific transition
                r = 0
                x = np.random.random() * R
                for i in range(len(Gamma)):
                    r += Gamma[i]
                    if r < x:
                        pass
                    else:
                        # register transition
                        l, m = reaction_index[i]
                        chosen_rate = Gamma[i]
                        if isinstance(m, int):  # island to island transition
                            n[l] -= e
                            n[m] += e
                            break
                        elif isinstance(m, str):  # side - island transition
                            if m == "from":  # electrode side to island
                                n[l] += e
                                break
                            elif m == "to":  # island to side electrode
                                n[l] -= e
                                break
                        else:
                            raise NameError

            else:  # rates too low, Tau leap instead
                dt = default_dt
                zero_curr_steady_state_counter += 1
                chosen_rate = 0
                if zero_curr_steady_state_counter % Steady_state_rep == 1 and zero_curr_steady_state_counter > 2:
                    print("counter is " + str(zero_curr_steady_state_counter))
                    not_in_steady_state = False

            # calculate I
            I_right, I_down = F.Get_current_from_gamma(Gamma, reaction_index, near_right, near_left)

            # solve ODE to update Qg, dQg/dt = (T^-1)(Qg-Qn)
            Qg = F.developQ(Qg, dt, Cond.InvTauEigenVectors, Cond.InvTauEigenValues,
                            n, Cond.InvTauEigenVectorsInv,
                            Cond.Tau_inv, C_inv, VxCix,
                            Rg, Tau, Cg, Cond.matrixQnPart)

            # update statistics
            I_avg, I_var = F.update_statistics(I_right, I_avg, I_var, t, dt)
            Q_avg, Q_var = F.update_statistics(Qg, Q_avg, Q_var, t, dt)
            n_avg, n_var = F.update_statistics(n, n_avg, n_var, t, dt)

            # calculate distance from steady state:
            steady_Q = F.return_Qn_for_n(n_avg, VxCix, Cg, Rg, Tau, Cond.matrixQnPart)
            dist_new = np.max(np.abs(steady_Q - Q_avg))
            max_diff_index = np.argmax(dist_new)
            std = np.sqrt(Q_var[max_diff_index])

            # check if distance from steady state is larger than the last by more than the allowed error
            if k > 1:
                # steady state condition
                if abs(dist_new / std) < KS_boundary:
                    print("dist is " + str(dist_new) + " there have been: " + str(not_decreasing) + " errors, k is "
                          + str(k))
                    print("counter is " + str(zero_curr_steady_state_counter))
                    not_in_steady_state = False

                # convergence
                if dist_new - dist > KS_boundary:
                    if Print:
                        print("err " + str(dist_new) + " std is " + str(std))
                        print(Qg)
                        print(steady_Q)
                        print(V)
                        print(n)
                        print(I_avg, abs(cycle_voltage / (e * Cond.Rg)))
                        print(R, chosen_rate)
                        print(cycle_voltage)
                    not_decreasing += 1

                elif k % 1000 == 0:
                    print("dist is " + str(dist_new) + " error num is " + str(not_decreasing) + " std is " + str(std))

            # update time
            dist = dist_new
            t += dt

        I_vec[cycle] = I_avg

    I_matrix[loop] = I_vec

I_vec_avg = np.zeros(cycles)  # results vector
for run in I_matrix:
    I_vec_avg += run / len(I_matrix)

I_V = plt.plot(Vleft / Volts, I_vec_avg / Amp)
plt.xlabel("Voltage")
plt.ylabel("Current")
if increase:
    plt.title("increasing voltage")
else:
    plt.title("decreasing voltage")
plt.show()
