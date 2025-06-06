import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import Functions as F
import Conditions as Cond
import copy
import time
from mpmath import quad, mp, exp, sqrt
import csv
import os
import sys
import bisect

# NORMAL ARRAY
# os.system("nohup bash -c '" + sys.executable + " train.py --size 192 >result.txt" + "' &")

# Conditions
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
taylor_limit = 0.0001
pos_energy_bound = -0.02  # -0.02 for T=0.001; 0.08 for T=0.01; 1.3 for T=0.1
neg_energy_bound = -0.07  # -0.07 for T=0.001; -0.19 for T=0.01; -1.4 for T=0.1

# parameters
e = Cond.e
kB = Cond.kB
Volts = abs(e) / Cond.C  # normalized voltage unit
Amp = abs(e) / (Cond.C * Cond.R)  # normalized current unit
Vright = 0
T0 = 0.001 * e * e / (Cond.C * kB)
T = 1 * T0
Ec = e ** 2 / (2 * np.mean(Cg))

# Gillespie parameter, KS statistic value for significance
Steady_state_rep = 100
expected_error = 0.01 * (Cond.row_num - 1) * np.sqrt(T / T0)
error_count = 0

# tabulating integral values
resolution = 0.000001
num_of_calc = (pos_energy_bound - neg_energy_bound) / resolution
vals_to_calc = np.linspace(pos_energy_bound, neg_energy_bound, num=round(num_of_calc))
table_val = []  # list of dE values
table_prob = []  # list of gamma(dE) values
s_eff = np.sqrt(2 * Ec * T)


def high_impedance_p(x, mu, Temp):
    """
    P- function for high impedance.
    :param x: function input (energy) == E+dE.
    :param mu: Electrostatic energy of environment == Ec.
    :param Temp: Temperature.
    :return: P(x)
    """
    sigma_squared = 2 * mu * Temp
    mu = -mu
    return exp(-(x - mu) ** 2 / (2 * sigma_squared)) / sqrt(2 * np.pi * sigma_squared)


wrote = False

if wrote:
    with open('table.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                table_val.append(float(row[0]))
                table_prob.append(float(row[1]))
            except IndexError:
                continue

else:
    rr = 0
    with open("table.csv", "w+") as f:
        file = csv.writer(f)
        for val in vals_to_calc:
            def integrand_gauss(x):
                if x == 0:
                    return T
                result = high_impedance_p(x + val, Ec, T) * x / (1 - exp(-x / T))
                return float(result)


            mp.dps = 30
            probability = quad(integrand_gauss, [-val - 0.1, -val + 0.1])
            mp.dps = 15
            print(probability)

            table_val += [val]
            table_prob += [probability]

            file.writerow([float(val), float(probability)])

            rr += 1
            print("done " + str(rr) + "out of" + str(len(vals_to_calc)))

t0 = time.time()


def approximate_gamma_integral(dE):
    global table_val
    global table_prob

    idx = bisect.bisect_left(table_val, dE)

    # Compare the two closest values: sorted_list[idx - 1] and sorted_list[idx]
    if idx == 0:
        if abs(table_val[0] - dE) <= abs(table_val[1] - dE):
            return table_prob[0]
        else:
            return table_prob[1]
    elif idx == len(table_val):
        if abs(table_val[idx - 2] - dE) <= abs(table_val[idx - 1] - dE):
            return table_prob[idx - 2]
        else:
            return table_prob[idx - 1]
    elif abs(table_val[idx - 1] - dE) <= abs(table_val[idx] - dE):
        return table_prob[idx - 1]
    else:
        return table_prob[idx]


def Gamma_approx(dE, Temp, Rt):
    global Ec
    global e
    s = np.sqrt(2 * Ec * Temp)

    # low bound starts at dE=-0.07 from thence is analytical
    # high bound starts at dE=-0.02 from thence is 0

    if dE < neg_energy_bound:
        return (-np.sqrt(np.pi / 2) * (Ec + dE) * s * e ** 2) / Rt
    elif pos_energy_bound > dE > neg_energy_bound:
        return approximate_gamma_integral(dE) * e ** 2 / Rt
    else:
        raise ValueError


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


def Get_Gamma(Gamma_, RR, reaction_index_, n_list, curr_V, cycle_voltage_):
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
            if dEij[i][j] < pos_energy_bound:
                Gamma_ += [Gamma_approx(dEij[i][j], T, R_t_ij[i][j])]
                RR += Gamma_[-1]
                reaction_index_ += [(i, j)]

    # left electrode to island transition:
    for isle in near_left:
        # for ith transition from electrode
        dE_left = (2 * curr_V[isle] - e * C_inv[isle][isle] - 2 * cycle_voltage_) * e / 2

        # rate for V_left->i
        if dE_left < pos_energy_bound:
            Gamma_ += [Gamma_approx(dE_left, T, R_t_i[isle])]
            RR += Gamma_[-1]
            reaction_index_ += [(isle, "from")]

        # for ith transition to electrode there must be at least one electron at isle i
        if n_list[isle] / e >= 1:
            dE_left = (2 * cycle_voltage_ - 2 * curr_V[isle] + e * C_inv[isle][isle]) * e / 2

            # rate for i->V_left
            if dE_left < pos_energy_bound:
                Gamma_ += [Gamma_approx(dE_left, T, R_t_i[isle])]
                RR += Gamma_[-1]
                reaction_index_ += [(isle, "to")]

    # similarly, for right side
    for isle in near_right:
        # for ith transition from electrode
        dE_right = (2 * curr_V[isle] - e * C_inv[isle][isle] - 2 * Vright) * e / 2

        # rate for V_right->i
        if dE_right < pos_energy_bound:
            Gamma_ += [Gamma_approx(dE_right, T, R_t_i[isle])]
            RR += Gamma_[-1]
            reaction_index_ += [(isle, "from")]

        # for ith transition to electrode
        if n_list[isle] / e >= 1:
            # for ith transition to electrode
            dE_right = (2 * Vright - 2 * curr_V[isle] + e * C_inv[isle][isle]) * e / 2

            # rate for i->V_right
            if dE_right < pos_energy_bound:
                Gamma_ += [Gamma_approx(dE_right, T, R_t_i[isle])]
                RR += Gamma_[-1]
                reaction_index_ += [(isle, "to")]

    return Gamma_, RR, reaction_index_


def Get_Steady_State(V_cycle):
    global error_count

    # general Charge distribution vectors
    Qg, Q_avg, Q_var = np.zeros(array_size), np.zeros(array_size), np.zeros(array_size)
    n, n_avg, n_var = np.zeros(array_size), np.zeros(array_size), np.zeros(array_size)
    I_avg, I_var = 0, 0

    # vector counting charge flow
    I_vec = np.zeros(cycles)

    for cycle in range(cycles):

        cycle_voltage = float(V_cycle[cycle])
        print("start " + str(cycle_voltage / Volts) + " loop:" + str(loop))

        k = 0
        zero_curr_steady_state_counter = 0
        not_decreasing = 0

        # starting conditions
        not_in_steady_state = True
        t = 0
        steady_state_timer = 5 * Cond.default_dt  #steady state fixed time

        while not_in_steady_state:
            # update number of reactions and voltage from last loop
            k += 1

            VxCix = copy.copy(Cond.VxCix(cycle_voltage, Vright))
            V = F.getVoltage(n, Qg, C_inv, VxCix)  # find V_i for ith island

            # define overall reaction rate R, rate vector, and a useful index
            R = 0
            reaction_index = []
            Gamma = []

            Gamma, R, reaction_index = Get_Gamma(Gamma, R, reaction_index, n, V, cycle_voltage)

            # transition occurred, limit for R is the typical ground drain current
            if R > abs(cycle_voltage / (e * Cond.Rg)):
                zero_curr_steady_state_counter = 0
                # typical interaction time
                dt = float(np.log(1 / np.random.random()) / R)
                if dt < 0:
                    raise ValueError

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
            steady_Q = F.return_Qn_for_n(n_avg, VxCix, Cg, Rg, Tau)
            dist_new = np.max(np.abs(steady_Q - Q_avg))
            max_diff_index = np.argmax(dist_new)

            # check if distance from steady state is larger than the last by more than the allowed error
            dist_info = False
            if k > 5:
                std = np.sqrt(Q_var[max_diff_index] * (k + 1) / (k * t))
                # steady state condition
                # print(k, dist_new, std)
                if abs(dist_new) < expected_error:
                    if dist_info:
                        print("dist is " + str(dist_new) + " there have been: " + str(not_decreasing) + " errors, k is "
                              + str(k) + " std is " + str(std) + " n " + str(np.sum(n)))
                        # print("counter is " + str(zero_curr_steady_state_counter))
                        print("timer is " + str(time.time() - t0))
                        print(steady_state_timer, dt)

                    steady_state_timer -= dt
                    if steady_state_timer <= 0:
                        not_in_steady_state = False

                # convergence failsafe
                if dist_new - dist > 0:
                    not_decreasing += 1
                    if not not_decreasing % 100000:
                        if abs(dist_new) > 0:
                            print("error")
                            print("dist is " + str(dist_new) + " there have been: " + str(
                                not_decreasing) + " errors, k is "
                                  + str(k) + " std is " + str(std) + " n " + str(np.sum(n)))
                            #print("counter is " + str(zero_curr_steady_state_counter))
                            #print("timer is " + str(time.time() - t0))
                            error_count += 1
                            not_in_steady_state = False

                # update on non-convergence
                elif k % 1000 == 0:
                    print("dist is " + str(dist_new) + " error num is " + str(not_decreasing) + " std is " + str(std))

            # update time
            dist = dist_new
            t += dt

        I_vec[cycle] = I_avg
    return I_vec


# implements increasing\decreasing choice
steps = 100
V_diff = 3
Vleft = np.linspace(Vright * Volts, (Vright + V_diff) * Volts, num=steps)
V_doubled = np.concatenate([Vleft, Vleft[-2::-1]])

# results matrix, ith column has the ith loop, jth row is the jth step of voltage
cycles = len(V_doubled)
I_matrix = np.zeros((loops, cycles))

for loop in range(loops):
    I_vec = Get_Steady_State(V_doubled)
    I_matrix[loop] = I_vec

I_vec_avg = np.zeros(cycles)  # results vector
for run in I_matrix:
    I_vec_avg += run / len(I_matrix)

I_vec_var = I_vec_avg = np.zeros(cycles)  # errors vector
for run_num in range(len(I_matrix)):
    I_vec_var += np.abs(I_matrix[run_num] - I_vec_avg) ** 2 / len(I_matrix)

I_vec_std = np.sqrt(I_vec_var)
# w+ truncates file
with open("book.csv", "w+") as f:
    file = csv.writer(f)
    for row in range(len(V_doubled)):
        to_write = [float(V_doubled[row] / Volts), float(I_vec_avg[row] / Amp), float(I_vec_std[row] / Amp)]
        file.writerow(to_write)

plot = True
if not plot:
    pass
else:
    I_V_increase = plt.plot(Vleft / Volts, I_vec_avg[:steps] / Amp, label="increasing", color="blue")
    I_V_decrease = plt.plot(V_doubled[steps:] / Volts, I_vec_avg[steps:] / Amp, label="decreasing", color="red")
    plt.xlabel("Voltage")
    plt.ylabel("Current")
    plt.legend()
    plt.show()

end_time = time.time()
print(int(end_time - t0) / 60)
