import numpy as np
import matplotlib.pyplot as plt
import Functions
import Functions as F
import Conditions as Cond
import copy

# Conditions
loops = 1000
steps = 1000
row_num = Cond.row_num
array_size = Cond.array_size
islands = list(range(array_size))
R_t = Cond.R_t
Rg = [Cond.Rg] * array_size
Cg = [Cond.Cg] * array_size
default_dt = Cond.default_dt
Tau_inv_matrix = Cond.Tau
Tau_matrix = np.linalg.inv(Tau_inv_matrix)
increase = True  # true if increasing else decreasing voltage during run

# parameters
e = Cond.e
kB = Cond.kB
Volts = abs(e) / Cond.C  # normalized voltage unit
Amp = abs(e) / (Cond.C * Cond.R)  # normalized current unit
Vright = 0
T = 0.001 * e * e / (Cond.C * kB)

# Gillespie parameter, KS statistic value for significance
KS_boundary = e*1e-2
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
        lastV = np.zeros(array_size)

        cycle_voltage = Vleft[cycle]
        print("start " + str(cycle_voltage / Volts) + " loop:" + str(loop))

        # starting conditions
        not_in_steady_state = True
        t = 0

        while not_in_steady_state:
            # update number of reactions and voltage from last loop
            k += 1
            V = F.V_t(n, Qg, cycle_voltage, Vright, Cond.C_inverse, Cond.VxCix, lastV)  # find V_i for ith island

            # define overall reaction rate R, rate vector, and a useful index
            R = 0
            reaction_index = []
            Gamma = []

            # dE values for i->j transition
            dEij = np.zeros((array_size, array_size))

            # island i to island j transition
            for i in islands:
                # if island i is empty pass over
                if n[i] == 0:
                    continue

                # else calculate transition rate to jth island
                neighbour_list = Functions.neighbour_list(Cond.row_num, i)
                for j in neighbour_list:
                    V_tag = copy.copy(V)
                    n_tag = np.array(list(n))

                    # for isle i to isle j transition
                    n_tag[i] -= e
                    n_tag[j] += e
                    V_tag[i] -= Cond.Cg * e
                    V_tag[j] += Cond.Cg * e

                    # calculate energy difference due to transition
                    V_new = F.V_t(n_tag, Qg, cycle_voltage, Vright, Cond.C_inverse, Cond.VxCix, V_tag)
                    dEij[i][j] = e * (V[j] + V_new[j] - V[i] - V_new[i]) / 2

                    # dEij must be negative fo transition i->j
                    if dEij[i][j] < 0:
                        Gamma += [F.gamma(dEij[i][j], T, R_t[i])]
                        R += Gamma[-1]
                        reaction_index += [(i, j)]

            # left electrode to island transition:
            dE_left = 0
            near_left = islands[0::row_num]
            for isle in near_left:
                V_tag = copy.copy(V)
                n_tag = np.array(list(n))

                # for ith transition from electrode
                n_tag[isle] += e
                V_tag[isle] += Cond.Cg * e

                V_new = F.V_t(n_tag, Qg, cycle_voltage, Vright, Cond.C_inverse, Cond.VxCix, V_tag)
                dE_left = (V[isle] + V_new[isle] - 2 * cycle_voltage) * e / 2

                # rate for V_left->i
                if dE_left < 0:
                    Gamma += [F.gamma(dE_left, T, R_t[isle])]
                    R += Gamma[-1]
                    reaction_index += [(isle, "from")]

                # return "borrowed electron" used for calculation
                n_tag[isle] -= e
                V_tag[isle] -= Cond.Cg * e

                # for ith transition to electrode there must be at least one electron at isle i
                if n_tag[isle] >= e:

                    # for ith transition from electrode
                    n_tag[isle] -= e
                    V_tag[isle] -= Cond.Cg * e

                    V_new = F.V_t(n_tag, Qg, cycle_voltage, Vright, Cond.C_inverse, Cond.VxCix, V)
                    dE_left = (V[isle] + V_new[isle] - 2 * cycle_voltage) * e / 2

                    # rate for i->V_left
                    if dE_left < 0:
                        Gamma += [F.gamma(dE_left, T, R_t[isle])]
                        R += Gamma[-1]
                        reaction_index += [(isle, "to")]

            # similarly, for right side
            dE_right = 0
            near_right = islands[(row_num - 1)::row_num]
            for isle in near_right:
                V_tag = copy.copy(V)
                n_tag = np.array(list(n))

                # for ith transition from electrode
                n_tag[isle] += e
                V_tag[isle] += Cond.Cg * e

                V_new = F.V_t(n_tag, Qg, cycle_voltage, Vright, Cond.C_inverse, Cond.VxCix, V)
                dE_right = (V[isle] + V_new[isle] - 2 * Vright) * e / 2

                # rate for V_right->i
                if dE_right < 0:
                    Gamma += [F.gamma(dE_right, T, R_t[isle])]
                    R += Gamma[-1]
                    reaction_index += [(isle, "from")]

                # return "borrowed electron" used for calculation
                n_tag[isle] -= e
                V_tag[isle] -= Cond.Cg * e

                # for ith transition to electrode
                if n_tag[isle] >= e:
                    # for ith transition from electrode
                    n_tag[isle] -= e
                    V_tag[isle] -= Cond.Cg * e

                    V_new = F.V_t(n_tag, Qg, cycle_voltage, Vright, Cond.C_inverse, Cond.VxCix, V)
                    dE_right = (V[isle] + V_new[isle] - 2 * Vright) * e / 2

                    # rate for i->V_right
                    if dE_right < 0:
                        Gamma += [F.gamma(dE_right, T, R_t[isle])]
                        R += Gamma[-1]
                        reaction_index += [(isle, "to")]

            # transition occurred, limit for R is the typical ground drain current
            if R > cycle_voltage / (e * Cond.Rg):
                zero_curr_steady_state_counter = 0

                # typical interaction time
                dt = np.log(1 / np.random.random()) / R

                # picking a specific transition
                r = 0
                x = np.random.random() * R
                chosen_rate = 0
                for i in range(len(Gamma)):
                    r += Gamma[i]
                    if r < x:
                        pass
                    else:
                        # register transition
                        chosen_rate = Gamma[i]
                        l, m = reaction_index[i]
                        if isinstance(m, int):  # island to island transition
                            n[l] -= e
                            n[m] += e
                            V[l] -= Cond.Cg*e
                            V[m] += Cond.Cg*e
                            break
                        elif isinstance(m, str):  # side - island transition
                            if m == "from":  # electrode side to island
                                n[l] += e
                                V[l] += Cond.Cg * e
                                break
                            elif m == "to":  # island to side electrode
                                n[l] -= e
                                V[l] -= Cond.Cg * e
                                break
                        else:
                            raise NameError

            else:  # rates too low, Tau leap instead
                dt = default_dt
                chosen_rate = None
                zero_curr_steady_state_counter += 1
                if zero_curr_steady_state_counter > Steady_state_rep:
                    print("counter is " + str(zero_curr_steady_state_counter))
                    not_in_steady_state = False


            # calculate Qn, solve ODE to update Qg --> Q, dQ/dt = (T^-1)(Q-Qn)
            Qn = F.return_Qn_for_n(n, Cond.VxCix(cycle_voltage, Vright, V), Rg, Cg, islands, Tau_matrix)
            Qg = F.developQ(Qg, dt, Cond.tau_inv_matrix, Qn)

            # update I
            I_right, I_down = F.Get_current_from_gamma(Gamma, reaction_index, near_right, near_left)

            # update statistics
            I_avg, I_var = F.update_statistics(I_right, I_avg, I_var, t, dt)
            Q_avg, Q_var = F.update_statistics(Qg, Q_avg, Q_var, t, dt)
            n_avg, n_var = F.update_statistics(n, n_avg, n_var, t, dt)

            # calculate distance from steady state:
            steady_Q = F.return_Qn_for_n(n_avg, Cond.VxCix(cycle_voltage, Vright, V), Rg, Cg, islands, Tau_matrix)
            dist_new = np.max(np.abs(steady_Q - Q_avg))

            # check if distance from steady state is larger than the last by more than the allowed error
            if k > 1:
                if dist_new > dist:
                    if dist_new - dist > 0.1 and k > 10:
                        print("err " + str(dist_new))
                    not_decreasing += 1
                    if not_decreasing == 50000:
                        print(l, m)
                        raise NameError
                    # if not_decreasing % Steady_state_rep == 1:

                # steady state condition
                elif dist_new < KS_boundary:
                    print("dist is " + str(dist_new) + " there have been: " + str(not_decreasing) + " errors, k is "
                          + str(k))
                    print("counter is " + str(zero_curr_steady_state_counter))
                    not_in_steady_state = False

                elif k % 1000 == 0:
                    print("dist is " + str(dist_new) + " error num is " + str(not_decreasing))

            dist = dist_new
            t += dt
            lastV = V

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
