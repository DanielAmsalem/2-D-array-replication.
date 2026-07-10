import bisect
import sys

import numpy as np
import numpy.typing as npt

import Functions as F
from define_objects import ExperimentInitialState, SteadyStateResult
import matplotlib

matplotlib.use("Agg")  # for clustrer, TkAgg for Pc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def approximate_gamma_integral(dE, T_at_junction, table_val, table_prob, T_table, flip):
    tol = 1e-6

    matches = np.where(np.abs(np.array(T_table) - T_at_junction) < tol)[0]
    if len(matches) != 1:
        raise ValueError("No single matching T found within tolerance\n T_table is " + str(np.array(T_table)) +
                         "\n T_at_junction is " + str(np.array(T_at_junction)) + " flip =" + str(flip))
    temp_idx = matches[0]

    sorted_vals = table_val[temp_idx::len(T_table)]
    probs = table_prob[temp_idx::len(T_table)]

    if len(sorted_vals) > 1 and sorted_vals[0] > sorted_vals[-1]:
        sorted_vals = sorted_vals[::-1]
        probs = probs[::-1]

    idx = bisect.bisect_left(sorted_vals, dE)

    # Compare the two closest values: sorted_list[idx - 1] and sorted_list[idx]
    if idx == 0:
        if abs(dE - sorted_vals[0]) <= abs(dE - sorted_vals[1]):
            return probs[0]
        else:
            return probs[1]
    elif idx == len(sorted_vals):
        if abs(dE - sorted_vals[-1]) <= abs(dE - sorted_vals[-2]):
            return probs[-1]
        else:
            return probs[-2]
    else:
        if abs(sorted_vals[idx - 1] - dE) <= abs(sorted_vals[idx] - dE):
            return probs[idx - 1]
        else:
            return probs[idx]


def Gamma_approx(dE, T_at_junction, Rt, Ec, e, neg_energy_bound, pos_energy_bound, table_val, table_prob, T_table,
                 flip):
    # low bound starts at dE=-0.09 and thence is analytically the mean of a gaussian
    # high bound starts at dE=-0.01 and thence is 0
    if Rt < 0:
        raise ValueError

    val = (-dE - Ec) / Rt

    if T_at_junction == 0:
        if dE < -Ec:
            return val
        else:
            return 0

    if dE < neg_energy_bound:
        if val < 0:
            print(val)
            raise ValueError
        return val

    elif pos_energy_bound > dE > neg_energy_bound:
        approx = approximate_gamma_integral(
            dE, T_at_junction, table_val, table_prob, T_table, flip
        )
        val = approx / Rt
        return val
    else:
        raise ValueError


def execute_transition(Gamma_list, n_list, reaction_index_, e):
    r = 0
    x = np.random.random() * np.sum(Gamma_list)

    for item in range(len(Gamma_list)):
        r += Gamma_list[item]
        if r < x:
            continue
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


def execute_gapped_transition(Gamma_list, n_list, reaction_index_, e):
    r = 0
    x = np.random.random() * np.sum(Gamma_list)

    for item in range(len(Gamma_list)):
        r += Gamma_list[item]
        if r < x:
            continue
        else:
            # register transition l->m
            ll, mm, number_of_particles_moved = reaction_index_[item]
            rate = Gamma_list[item]
            if isinstance(mm, int):  # island to island transition
                n_list[ll] -= e * number_of_particles_moved
                n_list[mm] += e * number_of_particles_moved
                break
            elif isinstance(mm, str):  # side - island transition
                if mm == "from":  # electrode side to island
                    n_list[ll] += e * number_of_particles_moved
                    break
                elif mm == "to":  # island to side electrode
                    n_list[ll] -= e * number_of_particles_moved
                    break
            else:
                raise NameError
    return n_list, ll, mm, rate


def Get_Gamma(Gamma_, e, reaction_index_, n_list, curr_V, cycle_voltage_, array_size, islands, row_num, C_inv,
              pos_energy_bound, neg_energy_bound, T_gradient, R_t_ij, R_t_i, near_left, near_right, Vright, Ec,
              table_val, table_prob, T_table, flip, periodic_y):
    # dE values for i->j transition
    dEij = np.zeros((array_size, array_size))

    # island i to island j transition
    for i in islands:
        # if island i is empty pass over
        if n_list[i] == 0:
            continue

        # else calculate transition rate to jth island
        neighbour_list = F.neighbour_list(row_num, i, periodic_y=periodic_y)
        for j in neighbour_list:
            # calculate energy difference due to transition dE = e*[(Vj-Vi)+(V'j-V'í)]/2
            # V'j = Vj +[C^-1 * (ej-ei)]j = Vj + C^-1(self) - C^-1[i][j]
            # V'i = Vi + [C^-1 * (ej-ei)]i = Vi - C^-1(self) + C^-1[i][j]
            dEij[i][j] = e * (curr_V[j] - curr_V[i]) + e * e * (C_inv[j][j] + C_inv[i][i] - 2 * C_inv[i][j]) / 2

            # dEij must be negative enough for transition i->j
            if dEij[i][j] < pos_energy_bound:
                Gamma_ += [Gamma_approx(dEij[i][j], T_gradient[i % row_num], R_t_ij[i][j], Ec, e, neg_energy_bound,
                                        pos_energy_bound, table_val, table_prob, T_table, flip)]
                reaction_index_ += [(i, j)]

    # left electrode to island transition:
    for isle in near_left:
        # for ith transition from electrode:
        # V'i = V'i [C^-1(ei)]i = V'i + e*C^-1(self)
        dE_left = (2 * curr_V[isle] + e * C_inv[isle][isle] - 2 * cycle_voltage_) * e / 2

        # rate for V_left->i
        if dE_left < pos_energy_bound:
            Gamma_ += [Gamma_approx(dE_left, T_gradient[isle % row_num], R_t_i[isle], Ec, e, neg_energy_bound,
                                    pos_energy_bound, table_val, table_prob, T_table, flip)]
            reaction_index_ += [(isle, "from")]

        # for ith transition to electrode there must be at least one electron at isle i
        if n_list[isle] / e >= 1:
            dE_left = (2 * cycle_voltage_ - 2 * curr_V[isle] + e * C_inv[isle][isle]) * e / 2

            # rate for i->V_left
            if dE_left < pos_energy_bound:
                Gamma_ += [Gamma_approx(dE_left, T_gradient[isle % row_num], R_t_i[isle], Ec, e, neg_energy_bound,
                                        pos_energy_bound, table_val, table_prob, T_table, flip)]
                reaction_index_ += [(isle, "to")]

    # similarly, for right side
    for isle in near_right:
        # for ith transition from electrode
        dE_right = (2 * curr_V[isle] + e * C_inv[isle][isle] - 2 * Vright) * e / 2

        # rate for V_right->i
        if dE_right < pos_energy_bound:
            Gamma_ += [Gamma_approx(dE_right, T_gradient[isle % row_num], R_t_i[isle], Ec, e, neg_energy_bound,
                                    pos_energy_bound, table_val, table_prob, T_table, flip)]
            reaction_index_ += [(isle, "from")]

        # for ith transition to electrode
        if n_list[isle] / e >= 1:
            # for ith transition to electrode
            dE_right = (2 * Vright - 2 * curr_V[isle] + e * C_inv[isle][isle]) * e / 2

            # rate for i->V_right
            if dE_right < pos_energy_bound:
                Gamma_ += [Gamma_approx(dE_right, T_gradient[isle % row_num], R_t_i[isle], Ec, e, neg_energy_bound,
                                        pos_energy_bound, table_val, table_prob, T_table, flip)]
                reaction_index_ += [(isle, "to")]

    return Gamma_, reaction_index_


def Get_Gamma_gapped(Gamma_, e, reaction_index_, n_list, curr_V, cycle_voltage_, array_size, islands, row_num, C_inv,
                     pos_energy_bound, neg_energy_bound, T_gradient, R_t_ij, R_t_i, near_left, near_right, Vright, Ec,
                     table_val, table_prob, T_table, flip, periodic_y, gap):
    '''
    opposed to regular get gamma here reaction index is of the form
    [(i,j,n)] where n={1,2} for qp/cp transition
    '''

    # dE values for i->j transition
    dEij = np.zeros((array_size, array_size))

    # island i to island j transition
    for i in islands:
        # if island i is empty pass over
        if n_list[i] == 0:
            continue

        # else calculate transition rate to jth island
        neighbour_list = F.neighbour_list(row_num, i, periodic_y=periodic_y)
        for j in neighbour_list:
            # calculate energy difference due to transition dE = e*[(Vj-Vi)+(V'j-V'í)]/2
            # V'j = Vj +[C^-1 * (ej-ei)]j = Vj + C^-1(self) - C^-1[i][j]
            # V'i = Vi + [C^-1 * (ej-ei)]i = Vi - C^-1(self) + C^-1[i][j]
            dEij[i][j] = e * (curr_V[j] - curr_V[i]) + e * e * (C_inv[j][j] + C_inv[i][i] - 2 * C_inv[i][j]) / 2

            # for cp just e -> 2*e
            if n_list[i] / e >= 2:
                cp_dE = (2 * e) * (curr_V[j] - curr_V[i]) + (2 * e) * (2 * e) * (
                        C_inv[j][j] + C_inv[i][i] - 2 * C_inv[i][j]) / 2

                Gamma_ += [F.Gamma_cp(cp_dE, T_gradient[i % row_num], Ec, Rt=R_t_ij[i][j], gap=gap)]
                reaction_index_ += [(i, j, 2)]

            # dEij must be negative enough for qp transition i->j
            # for qp the integrals are store beforehand and divided the same way by Rt*e^2
            if dEij[i][j] < pos_energy_bound:
                Gamma_ += [Gamma_approx(dEij[i][j], T_gradient[i % row_num], R_t_ij[i][j], Ec, e, neg_energy_bound,
                                        pos_energy_bound, table_val, table_prob, T_table, flip)]
                reaction_index_ += [(i, j, 1)]

    # left electrode to island transition:
    for isle in near_left:
        # for ith transition from electrode:
        # V'i = V'i [C^-1(ei)]i = V'i + e*C^-1(self)
        dE_left = (2 * curr_V[isle] + e * C_inv[isle][isle] - 2 * cycle_voltage_) * e / 2
        cp_dE_left = (2 * curr_V[isle] + (2 * e) * C_inv[isle][isle] - 2 * cycle_voltage_) * e

        Gamma_ += [F.Gamma_cp(cp_dE_left, T_gradient[isle % row_num], Ec, Rt=R_t_i[isle], gap=gap)]
        reaction_index_ += [(isle, "from", 2)]

        # rate for V_left->i
        if dE_left < pos_energy_bound:
            Gamma_ += [Gamma_approx(dE_left, T_gradient[isle % row_num], R_t_i[isle], Ec, e, neg_energy_bound,
                                    pos_energy_bound, table_val, table_prob, T_table, flip)]
            reaction_index_ += [(isle, "from", 1)]

        # for ith transition to electrode there must be at least one electron at isle i
        if n_list[isle] / e >= 1:
            dE_left = (2 * cycle_voltage_ - 2 * curr_V[isle] + e * C_inv[isle][isle]) * e / 2

            if n_list[isle] / e >= 2:
                cp_dE_left = (2 * cycle_voltage_ - 2 * curr_V[isle] + 2 * e * C_inv[isle][isle]) * e

                Gamma_ += [F.Gamma_cp(cp_dE_left, T_gradient[isle % row_num], Ec, Rt=R_t_i[isle], gap=gap)]
                reaction_index_ += [(isle, "to", 2)]

            # rate for i->V_left
            if dE_left < pos_energy_bound:
                Gamma_ += [Gamma_approx(dE_left, T_gradient[isle % row_num], R_t_i[isle], Ec, e, neg_energy_bound,
                                        pos_energy_bound, table_val, table_prob, T_table, flip)]
                reaction_index_ += [(isle, "to", 1)]

    # similarly, for right side
    for isle in near_right:
        # for ith transition from electrode
        dE_right = (2 * curr_V[isle] + e * C_inv[isle][isle] - 2 * Vright) * e / 2
        cp_dE_right = (2 * curr_V[isle] + 2 * e * C_inv[isle][isle] - 2 * Vright) * e

        Gamma_ += [F.Gamma_cp(cp_dE_right, T_gradient[isle % row_num], Ec, Rt=R_t_i[isle], gap=gap)]
        reaction_index_ += [(isle, "from", 2)]

        # rate for V_right->i
        if dE_right < pos_energy_bound:
            Gamma_ += [Gamma_approx(dE_right, T_gradient[isle % row_num], R_t_i[isle], Ec, e, neg_energy_bound,
                                    pos_energy_bound, table_val, table_prob, T_table, flip)]
            reaction_index_ += [(isle, "from", 1)]

        # for ith transition to electrode
        if n_list[isle] / e >= 1:
            # for ith transition to electrode
            dE_right = (2 * Vright - 2 * curr_V[isle] + e * C_inv[isle][isle]) * e / 2

            if n_list[isle] / e >= 2:
                cp_dE_right = (2 * Vright - 2 * curr_V[isle] + 2 * e * C_inv[isle][isle]) * e

                Gamma_ += [F.Gamma_cp(cp_dE_right, T_gradient[isle % row_num], Ec, Rt=R_t_i[isle], gap=gap)]
                reaction_index_ += [(isle, "to", 2)]

            # rate for i->V_right
            if dE_right < pos_energy_bound:
                Gamma_ += [Gamma_approx(dE_right, T_gradient[isle % row_num], R_t_i[isle], Ec, e, neg_energy_bound,
                                        pos_energy_bound, table_val, table_prob, T_table, flip)]
                reaction_index_ += [(isle, "to", 1)]

    return Gamma_, reaction_index_


def apply_multiple_transitions(n_list, reaction_index_, firings, e):
    """
    Applies multiple transitions based on an array of firing counts.
    Works for both standard and gapped reaction indices.
    """
    for item, count in enumerate(firings):
        if count == 0:
            continue

        reaction = reaction_index_[item]
        # Check if it's a gapped reaction (length 3) or normal (length 2)
        if len(reaction) == 3:
            ll, mm, particles_moved = reaction
            charge_transfer = e * particles_moved * count
        else:
            ll, mm = reaction
            charge_transfer = e * count

        # Apply the transitions
        if isinstance(mm, int):  # island to island
            n_list[ll] -= charge_transfer
            n_list[mm] += charge_transfer
        elif isinstance(mm, str):  # side to island
            if mm == "from":
                n_list[ll] += charge_transfer
            elif mm == "to":
                n_list[ll] -= charge_transfer

    return n_list


def Get_Steady_State(
        loop_index: int,
        init: ExperimentInitialState,
        V_cycle: npt.NDArray,
        cycles: int,
        table_val,
        table_prob,
        flip,
        table_T,
        expected_error: float,
        T: npt.NDArray,
        pos_energy_bound: float,
        neg_energy_bound: float,
        repetition: int,
        capture_heatmap_at_idx: float,
        periodic_y: bool,
        plot_ongoing_voltage_map: bool,
        gap_ratio: float
):
    error_count = 0
    # general Charge distribution vectors
    Qg, Q_avg, Q_var = np.zeros(init.array_size), np.zeros(init.array_size), np.zeros(init.array_size)
    n, n_avg, n_var = np.zeros(init.array_size), np.zeros(init.array_size), np.zeros(init.array_size)

    # vectors counting charge flow
    I_vec = np.zeros(cycles)
    Jx, Jy = np.zeros((init.row_num, init.row_num + 1)), np.zeros((init.row_num, init.row_num + 1))

    if plot_ongoing_voltage_map:
        plt.ion()
        fig, ax = plt.subplots()

        ### plot V on each island
        im = ax.imshow(n.reshape(init.row_num, init.row_num), cmap='viridis', origin='lower')
        im.set_clim(vmin=0, vmax=4)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Voltage (V)")

        ### add population on each vertex
        x, y = np.arange(init.row_num), np.arange(init.row_num)
        X, Y = np.meshgrid(x, y)

        # log norm to see particle movement rather than stationary chrage
        norm = LogNorm(vmin=0.1, vmax=15)
        scatter = ax.scatter(X.ravel(), Y.ravel(), c=n.reshape(-1), cmap='plasma', norm=norm, s=20, edgecolors='none')

        plt.colorbar(scatter, ax=ax, label="Population")

    for cycle in range(cycles):
        cycle_voltage = float(V_cycle[cycle])
        k = 0
        q = 0
        zero_curr_steady_state_counter = 0
        not_decreasing = 0

        # starting conditions
        not_in_steady_state = True
        t = 0
        t_ss = 0  # steady state timer for recording the current
        I_avg, I_var = 0, 0

        steady_state_timer = init.timeStep  # steady state fixed time
        steady_state_reps = init.Steady_state_rep * 5  # "5*RC"

        while not_in_steady_state:
            # update number of reactions and voltage from last loop
            k += 1
            q += 1

            VxCix = F.get_VxCix(cycle_voltage, init.Vright, init.array_size, init.near_left, init.near_right, init.Cix)

            V = F.getVoltage(n, Qg, init.C_inv, VxCix, init.e)  # find V_i for ith island
            if 0 < q % 1000 < 10 and plot_ongoing_voltage_map:
                # print(n.reshape(init.row_num, init.row_num))
                im.set_data(V.reshape(init.row_num, init.row_num))
                ax.set_title(r" $\Delta$" + f" V : {round(cycle_voltage, 3)}")
                scatter.set_array(n.reshape(-1))
                plt.pause(0.001)

            if k == 1 and not loop_index % 5:
                print(f"T_std={repetition}/20,{loop_index=}: current voltage is: {cycle}", flush=True)

            # define overall    rate vector, and a useful index
            reaction_index = []
            Gamma = []

            if abs(gap_ratio) < 1e-3:
                # normal metalic island case
                Gamma, reaction_index = Get_Gamma(
                    Gamma_=Gamma,
                    e=init.e,
                    reaction_index_=reaction_index,
                    n_list=n,
                    curr_V=V,
                    cycle_voltage_=cycle_voltage,
                    array_size=init.array_size,
                    islands=init.islands,
                    row_num=init.row_num,
                    C_inv=init.C_inv,
                    pos_energy_bound=pos_energy_bound,
                    neg_energy_bound=neg_energy_bound,
                    T_gradient=T,
                    R_t_ij=init.R_t_ij,
                    R_t_i=init.R_t_i,
                    near_left=init.near_left,
                    near_right=init.near_right,
                    Vright=init.Vright,
                    Ec=init.Ec,
                    table_val=table_val,
                    table_prob=table_prob,
                    T_table=table_T,
                    flip=flip,
                    periodic_y=periodic_y)

                # transition occurred, limit for R=sum(Gamma) is the typical ground drain current
                R = np.sum(Gamma)
                if R > cycle_voltage / init.CondRg:
                    # reset zero curr steady state detection
                    zero_curr_steady_state_counter = 0

                    # typical interaction time
                    dt = float(np.log(1 / np.random.random()) / R)
                    if dt <= 0:
                        raise ValueError

                    # picking a specific transition
                    n, l, m, chosen_rate = execute_transition(Gamma, n, reaction_index, init.e)

                else:  # rates too low, Poisson Tau-Leaping instead
                    dt = init.default_dt
                    # expected occurrences = Rate * Time
                    firings = np.random.poisson(np.array(Gamma) * dt)
                    if np.any(firings > 0):
                        # A transition occurred! Apply them and reset freeze counter.
                        zero_curr_steady_state_counter = 0
                        n = apply_multiple_transitions(n, reaction_index, firings, init.e)
                    else:
                        # Truly no transitions occurred.
                        zero_curr_steady_state_counter += 1
                        if (
                                zero_curr_steady_state_counter % init.Steady_state_rep == 1
                                and zero_curr_steady_state_counter > 2
                        ):
                            not_in_steady_state = False

            else:
                # gapped case
                Gamma, reaction_index = Get_Gamma_gapped(
                    Gamma_=Gamma,
                    e=init.e,
                    reaction_index_=reaction_index,
                    n_list=n,
                    curr_V=V,
                    cycle_voltage_=cycle_voltage,
                    array_size=init.array_size,
                    islands=init.islands,
                    row_num=init.row_num,
                    C_inv=init.C_inv,
                    pos_energy_bound=pos_energy_bound,
                    neg_energy_bound=neg_energy_bound,
                    T_gradient=T,
                    R_t_ij=init.R_t_ij,
                    R_t_i=init.R_t_i,
                    near_left=init.near_left,
                    near_right=init.near_right,
                    Vright=init.Vright,
                    Ec=init.Ec,
                    table_val=table_val,
                    table_prob=table_prob,
                    T_table=table_T,
                    flip=flip,
                    periodic_y=periodic_y,
                    gap=gap_ratio * init.Ec)

                R = np.sum(Gamma)
                if R > cycle_voltage / init.CondRg:
                    # reset zero curr steady state detection
                    zero_curr_steady_state_counter = 0

                    # typical interaction time
                    dt = float(np.log(1 / np.random.random()) / R)
                    if dt <= 0:
                        raise ValueError

                    # picking a specific transition
                    n, l, m, chosen_rate = execute_gapped_transition(Gamma, n, reaction_index, init.e)

                else:  # rates too low, Poisson Tau-Leaping instead
                    dt = init.default_dt
                    # expected occurrences = Rate * Time
                    firings = np.random.poisson(np.array(Gamma) * dt)
                    if np.any(firings > 0):
                        # A transition occurred! Apply them and reset freeze counter.
                        zero_curr_steady_state_counter = 0
                        n = apply_multiple_transitions(n, reaction_index, firings, init.e)
                    else:
                        # Truly no transitions occurred.
                        zero_curr_steady_state_counter += 1
                        if (
                                zero_curr_steady_state_counter % init.Steady_state_rep == 1
                                and zero_curr_steady_state_counter > 2
                        ):
                            not_in_steady_state = False

            # solve ODE to update Qg, dQg/dt = (T^-1)(Qg-Qn)
            Qg = F.developQ(Qg, dt, n, VxCix, init)

            # update statistics
            if steady_state_reps <= 0:
                I_right, I_down = F.Get_current_from_gamma(Gamma, reaction_index, init.near_right, init.near_left,
                                                           init.row_num, periodic_y=periodic_y)
                # Use t_ss for current statistics only
                I_avg, I_var = F.update_statistics(I_right, I_avg, I_var, t_ss, dt)
                t_ss += dt  # Increment steady state time

            # Charge statistics continue to use standard time 't'
            Q_avg, Q_var = F.update_statistics(Qg, Q_avg, Q_var, t, dt)
            n_avg, n_var = F.update_statistics(n, n_avg, n_var, t, dt)

            # calculate distance from steady state:
            steady_Q = F.return_Qn_for_n(n_avg, VxCix, init)
            dist_new = np.max(np.abs(steady_Q - Q_avg))
            max_diff_index = np.argmax(dist_new)

            # check if distance from steady state is larger than the last by more than the allowed error
            if k > 100:
                std = (np.sqrt(Q_var[max_diff_index] * (k + 1) / (k * t))) / np.sqrt(len(Q_avg))

                if dist_new - dist > min(std, expected_error):
                    not_decreasing += 1
                    steady_state_reps = init.Steady_state_rep * 5
                    # reset current stats/timer when steady state is lost
                    t_ss = 0
                    I_avg, I_var = 0, 0

                    if not not_decreasing % init.max_count:
                        error_count += 1
                        not_in_steady_state = False

                # steady state conditions
                elif abs(dist_new) < expected_error:
                    steady_state_reps -= 1
                    if steady_state_reps <= 0:
                        steady_state_timer -= dt
                        if cycle_voltage == V_cycle[capture_heatmap_at_idx]:
                            Jx_, Jy_ = F.Get_current_map(Gamma, reaction_index,
                                                         init.near_right, init.near_left, init.row_num, n,
                                                         periodic_y=periodic_y)
                            Jx += Jx_
                            Jy += Jy_

                        if steady_state_timer <= 0:
                            # for this V capture the current map
                            not_in_steady_state = False

                # reset steady_state_timer
                else:
                    steady_state_timer = init.timeStep
                    steady_state_reps = init.Steady_state_rep * 5
                    # reset current stats/timer when steady state is lost
                    t_ss = 0
                    I_avg, I_var = 0, 0

            # update time
            dist = dist_new
            t += dt

        I_vec[cycle] = I_avg
    return SteadyStateResult(loop_index, error_count, I_vec, Jx, Jy)
