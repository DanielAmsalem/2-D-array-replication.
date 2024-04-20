import numpy as np
import matplotlib.pyplot as plt
import Functions as F
import Conditions as Cond
import copy
from line_profiler_pycharm import profile
import time

# Conditions
Enable, Print = False, False
loops = 2
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
C_inv = Cond.C_inverse
increase = True  # true if increasing else decreasing voltage during run
taylor_limit = 0.0001

# parameters
e = Cond.e
kB = Cond.kB
Volts = abs(e) / Cond.C  # normalized voltage unit
Amp = abs(e) / (Cond.C * Cond.R)  # normalized current unit
Vright = 0
T = 0.001 * e * e / (Cond.C * kB)

# Gillespie parameter, KS statistic value for significance
Steady_state_rep = 100


@profile
def gamma(dE, Temp, Rt):
    """
    dE is a strictly negative real number; dE<0
    """
    try:
        beta = 1 / (Temp * kB)
        a = dE * beta
    except OverflowError:  # T may be too small
        return NameError

    return float(-dE / (e * e * Rt * (1 - np.exp(a))))


@profile
def execute_transition(Gamma_list, n_list, RR, reaction_index_):
    r = 0
    x = np.random.random() * RR
    for item in range(len(Gamma_list)):
        r += Gamma_list[item]
        if r < x:
            pass
        else:
            # register transition
            ll, mm = reaction_index_[item]
            rate = Gamma_list[item]
            if isinstance(mm, int):  # island to island transition
                n_list[ll] -= e
                n_list[mm] += e
                break
            elif isinstance(mm, str):  # side - island transition
                if mm == "from":  # electrode side to island
                    n_list[ll] += e
                    break
                elif mm == "to":  # island to side electrode
                    n_list[ll] -= e
                    break
            else:
                raise NameError
    return n_list, ll, mm, rate


@profile
def Get_Gamma(Gamma_, RR, reaction_index_, n_list, Q, VxCix_, curr_V, cycle_voltage_):
    # dE values for i->j transition
    dEij = np.zeros((array_size, array_size))

    # island i to island j transition
    for i in islands:
        # if island i is empty pass over
        if n_list[i] == 0:
            continue

        # else calculate transition rate to jth island
        neighbour_list = F.neighbour_list(Cond.row_num, i)
        for j in neighbour_list:
            # calculate energy difference due to transition
            dEij[i][j] = e * (2 * curr_V[j] - e * C_inv[j][i] + e * C_inv[j][j] -
                              (2 * curr_V[i] - e * C_inv[i][i] + e * C_inv[i][j])) / 2

            # dEij must be negative for transition i->j
            if dEij[i][j] < 0:
                Gamma_ += [gamma(dEij[i][j], T, R_t_ij[i][j])]
                RR += Gamma_[-1]
                reaction_index_ += [(i, j)]

    # left electrode to island transition:
    for isle in near_left:
        # for ith transition from electrode
        dE_left = (2 * curr_V[isle] - e * C_inv[isle][isle] - 2 * cycle_voltage_) * e / 2

        # rate for V_left->i
        if dE_left < 0:
            Gamma_ += [gamma(dE_left, T, R_t_i[isle])]
            RR += Gamma_[-1]
            reaction_index_ += [(isle, "from")]

        # for ith transition to electrode there must be at least one electron at isle i
        if n_list[isle] / e >= 1:
            dE_left = (2 * cycle_voltage_ - 2 * curr_V[isle] + e * C_inv[isle][isle]) * e / 2

            # rate for i->V_left
            if dE_left < 0:
                Gamma_ += [gamma(dE_left, T, R_t_i[isle])]
                RR += Gamma_[-1]
                reaction_index_ += [(isle, "to")]

    # similarly, for right side
    for isle in near_right:
        # for ith transition from electrode
        dE_right = (2 * curr_V[isle] - e * C_inv[isle][isle] - 2 * Vright) * e / 2

        # rate for V_right->i
        if dE_right < 0:
            Gamma_ += [gamma(dE_right, T, R_t_i[isle])]
            RR += Gamma_[-1]
            reaction_index_ += [(isle, "from")]

        # for ith transition to electrode
        if n_list[isle] / e >= 1:
            # for ith transition to electrode
            dE_right = (2 * Vright - 2 * curr_V[isle] + e * C_inv[isle][isle]) * e / 2

            # rate for i->V_right
            if dE_right < 0:
                Gamma_ += [gamma(dE_right, T, R_t_i[isle])]
                RR += Gamma_[-1]
                reaction_index_ += [(isle, "to")]

    return Gamma_, RR, reaction_index_


@profile
def Get_Steady_State():
    # general Charge distribution vectors
    Qg, Q_avg, Q_var = np.zeros(array_size), np.zeros(array_size), np.zeros(array_size)
    n, n_avg, n_var = np.zeros(array_size), np.zeros(array_size), np.zeros(array_size)
    I_avg, I_var = 0, 0

    # vector counting charge flow
    I_vec = np.zeros(cycles)

    for cycle in range(cycles):

        cycle_voltage = float(Vleft[cycle])
        print("start " + str(cycle_voltage / Volts) + " loop:" + str(loop))

        k = 0
        zero_curr_steady_state_counter = 0
        not_decreasing = 0

        # starting conditions
        not_in_steady_state = True
        t = 0
        t0 = time.time()

        while not_in_steady_state:
            # update number of reactions and voltage from last loop
            k += 1

            VxCix = copy.copy(Cond.VxCix(cycle_voltage, Vright))
            V = F.getVoltage(n, Qg, C_inv, VxCix)  # find V_i for ith island

            # define overall reaction rate R, rate vector, and a useful index
            R = 0
            reaction_index = []
            Gamma = []

            Gamma, R, reaction_index = Get_Gamma(Gamma, R, reaction_index, n, Qg, VxCix, V, cycle_voltage)

            # transition occurred, limit for R is the typical ground drain current
            if R > abs(cycle_voltage / (e * Cond.Rg)):
                zero_curr_steady_state_counter = 0
                # typical interaction time
                dt = np.log(1 / np.random.random()) / R

                # picking a specific transition
                n, l, m, chosen_rate = execute_transition(Gamma, n, R, reaction_index)

            else:  # rates too low, Tau leap instead
                dt = default_dt
                zero_curr_steady_state_counter += 1
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

            # check if distance from steady state is larger than the last by more than the allowed error
            if k > 100:
                std = np.sqrt(Q_var[max_diff_index] * (k + 1) / (k * t))
                # steady state condition
                if abs(dist_new) < std < 0.1:
                    print("dist is " + str(dist_new) + " there have been: " + str(not_decreasing) + " errors, k is "
                          + str(k) + " std is " + str(std) + " n " + str(np.sum(n)))
                    # print("counter is " + str(zero_curr_steady_state_counter))
                    print("timer is " + str(time.time() - t0))
                    not_in_steady_state = False

                # convergence
                if dist_new - dist > 0:
                    not_decreasing += 1
                    #print(dist_new)
                    if not_decreasing > 10000:
                        print(k, not_decreasing, std)
                        print(Qg)
                        print(V)
                        raise NameError

                elif k % 1000 == 0:
                    print("dist is " + str(dist_new) + " error num is " + str(not_decreasing) + " std is " + str(std))

            # update time
            dist = dist_new
            t += dt

        I_vec[cycle] = I_avg
    return I_vec


# implements increasing\decreasing choice
if increase:
    Vleft = np.linspace(Vright * Volts, (Vright + 4) * Volts, num=100)
else:
    Vleft = np.linspace((Vright + 4) * Volts, Vright * Volts, num=100)

# results matrix, ith column has the ith loop, jth row is the jth step of voltage
cycles = len(Vleft)
I_matrix = np.zeros((loops, cycles))

for loop in range(loops):
    I_vec = Get_Steady_State()
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
