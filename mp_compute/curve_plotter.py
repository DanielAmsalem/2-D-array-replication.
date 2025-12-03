import csv
from pathlib import Path
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import time
from define_objects import ExperimentInitialState


def iv_curve_compute_and_save_csv(
        init: ExperimentInitialState,
        filename: str,
        Vleft,
        results,
        repetition: int,
        results_path: Path,
        get_heatmap: bool,
        heatmap_at_V: float,
):
    V_doubled = np.concatenate([Vleft, Vleft[-2::-1]])

    # results matrix, ith column has the ith loop, jth row is the jth step of voltage
    cycles = len(V_doubled)
    I_vec_avg = np.zeros(cycles)

    for run in results:
        I_vec_avg += run.I_vec / len(results)

    I_vec_var = np.zeros(cycles)  # variance of current vector
    for run in results:
        I_vec_var += np.abs(run.I_vec - I_vec_avg) ** 2 / len(results)
    I_vec_std = np.sqrt(I_vec_var)

    # w+ truncates file
    with open(results_path / f"book_{filename}_rep{repetition}.csv", "w+") as f:
        file = csv.writer(f)
        for row in range(len(V_doubled)):
            to_write = [
                float(V_doubled[row] / init.Volts),
                float(I_vec_avg[row] / init.Amp),
                float(I_vec_std[row] / init.Amp),
            ]
            file.writerow(to_write)

    if get_heatmap:
        return plot_heat_map(results=results,
                             row_num=init.row_num,
                             filename=filename,
                             repetition=repetition,
                             results_path=results_path,
                             I_avg=I_vec_avg,
                             heatmap_at_V=heatmap_at_V)
    else:
        return 0


def report_param(init: ExperimentInitialState,
                 filename: str,
                 repetition: int,
                 T: list,
                 expected_error: float,
                 loop_count: int,
                 T_std: float,
                 t0,
                 results_path: Path,
                 tot_error_count: int):
    with open(results_path / f"parameters_{filename}_rep{repetition}.txt", "w") as f:
        f.write(f"repetition {repetition}\n")
        f.write("loop parameters" + "\n")
        f.write("---------------------------------------------" + "\n")
        f.write("row_num : " + str(init.row_num) + "\n")
        f.write("distribute_R : " + str(init.distribute_R) + "\n")
        if init.distribute_R:
            f.write("stdR (exponent) : " + str(init.stdR) + "\n")
        f.write("distribute_C : " + str(init.distribute_C) + "\n")
        if init.distribute_C:
            f.write("sig (normal) : " + str(init.sig) + "\n")
        f.write("e : " + str(init.e) + "\n")
        f.write("C : " + str(init.C_avg) + "\n")
        f.write("Cg : " + str(np.mean(init.Cg)) + "\n")
        f.write("R : " + str(init.R_avg) + "\n")
        f.write("Rg : " + str(init.CondRg) + "\n")
        f.write("default_dt : " + str(init.default_dt) + "\n")
        f.write("time step : " + str(init.timeStep) + "\n")
        f.write("---------------------------------------------" + "\n")
        f.write("\n")
        f.write("\n")
        f.write("loop variables : " + str(loop_count) + "\n")
        f.write("---------------------------------------------" + "\n")
        f.write("these are the raw variances and means\n")
        f.write("<R> : " + str(init.R_avg) + ", std(Rt_ij) : " + str(np.std(init.R_t_ij)) + "\n")
        f.write("<Rt_i> : " + str(np.mean(init.R_t_i)) + ", std(Rt_i) : " + str(np.std(np.array(init.R_t_i))) + "\n")
        f.write("<C> : " + str(init.mean_allCs) + ", std(C) : " + str(init.std_allCs) + "\n")
        f.write("<Cix> : " + str(init.mean_sideCs) + ", std(Cix) : " + str(init.std_sideCs) + "\n")
        f.write("\n")
        f.write("T0 : " + str(init.T0) + "\n")
        f.write("T_std : " + str(T_std) + "\n")
        f.write("T : " + str(T) + "\n")
        f.write("---------------------------------------------" + "\n")
        f.write(f"flip : {init.flip}\n")
        f.write("steady_state_rep : " + str(init.Steady_state_rep) + "\n")
        f.write("expected_error : " + str(expected_error) + "\n")
        f.write("resolution : " + str(init.resolution) + "\n")
        end_time = time.time()
        mins = int((end_time - t0) / 60)
        f.write("runtime : " + str(mins) + "m\n")
        f.write(f"error count: {tot_error_count}")
    return True


def plot_heat_map(results,
                  row_num,
                  repetition: int,
                  filename: str,
                  results_path: Path,
                  I_avg,
                  heatmap_at_V: float):
    n = row_num
    Jx, Jy = np.zeros((n, n + 1)), np.zeros((n, n + 1))
    # store average currents
    for run in results:
        Jx += run.Jx / len(results)
        Jy += run.Jy / len(results)
    print(Jy)

    # create grid
    Y, X = np.mgrid[0:n, 0:(n + 2)]
    plt.figure(figsize=(6, 6))
    # create x current vecs at each point
    Y, X = np.mgrid[0:n, 0:(n + 1)]
    plt.figure(figsize=(6, 6))
    # create x current vecs at each point
    plt.quiver(X + 0.5, Y + 0.5, Jx, np.zeros((n, n + 1)),
               np.sqrt(Jx ** 2 + Jy ** 2),  # color by magnitude
               scale=np.abs(Jx).max(), scale_units='xy', angles='xy',
               cmap='coolwarm')
    # create y current vecs at each point
    plt.quiver(X + 0.5, Y + 0.5, np.zeros((n, n + 1)), Jy,
               np.sqrt(Jx ** 2 + Jy ** 2),  # color by magnitude
               scale=np.abs(Jy).max(), scale_units='xy', angles='xy',
               cmap='coolwarm')
    plt.grid(True, color="lightgray", alpha=0.5)
    plt.xticks(list(range(n + 2)), ["Vleft"] + [str(i) for i in range(n)] + ["Vright"])
    plt.yticks(range(n + 1))
    pic_name = filename + f"_rep{repetition}.png"
    plt.title(f"at V = {heatmap_at_V}")
    plt.savefig(fname=results_path / pic_name, dpi=2100, bbox_inches="tight")
    # plt.show()
    return 0


def extract_nn_resistances(R, n, R_sides, near_left, near_right):
    Rx = np.zeros((n, n + 1))  # horizontal edges
    Ry = np.zeros((n - 1, n))  # vertical edges

    for y in range(n):
        for x in range(n):
            i = x + n * y  # index of (x,y)

            # horizontal neighbour
            if x < n - 1:
                j = (x + 1) + n * y
                Rx[y, x + 1] = R[i, j]

            # vertical neighbour
            if y < n - 1:
                j = x + n * (y + 1)
                Ry[y, x] = R[i, j]

    ## add electordes:
    for y in range(n):
        for isle in near_left:
            if isle // n == y:
                Rx[y][0] = R_sides[isle]

        for isle in near_right:
            if isle // n == y:
                Rx[y][n] = R_sides[isle]

    return Rx, Ry


def plot_resistance_maps(Rx, Ry, n, results_path, show: bool):
    # Horizontal edges
    plt.figure(figsize=(6, 4))
    plt.title("Horizontal resistances $R_x$")
    plt.imshow(Rx, cmap='inferno', origin='lower', extent=[0, Rx.shape[1], 0, Rx.shape[0]])
    plt.colorbar(label="Resistance")
    plt.xlabel("x (edge start)")
    plt.ylabel("y")
    plt.xticks(list(range(n + 2)), ["Vleft"] + [str(i) for i in range(n)] + ["Vright"])
    plt.yticks(range(n))
    plt.tight_layout()
    plt.savefig(fname=results_path / "_Rx", dpi=2100, bbox_inches="tight")
    if show:
        plt.show()
    print("saved Rx plot")

    # Vertical edges
    plt.figure(figsize=(6, 4))
    plt.title("Vertical resistances $R_y$")
    plt.imshow(Ry, cmap='inferno', origin='lower')
    plt.colorbar(label="Resistance")
    plt.xlabel("x")
    plt.ylabel("y (edge start)")
    plt.tight_layout()
    plt.savefig(fname=results_path / "_Ry", dpi=2100, bbox_inches="tight")
    if show:
        plt.show()

def plot_capacitance_map(C_inv,n, results_path, show: bool):
    """
        C_inv is an (n*n, n*n) inverse capacitance matrix.
        n is grid dimension.
    """
    # Extract blocks
    C_self = np.zeros((n, n))
    C_horiz = np.zeros((n, n - 1))
    C_vert = np.zeros((n - 1, n))

    for y in range(n):
        for x in range(n):
            i = y * n + x
            C_self[y, x] = C_inv[i, i]

            if x < n - 1:
                C_horiz[y, x] = C_inv[i, i + 1]

            if y < n - 1:
                C_vert[y, x] = C_inv[i, (y + 1) * n + x]

    # =====================================================
    #      GLOBAL COLOR NORMALIZATION
    # =====================================================
    # Self-capacitance normalization
    norm_self = Normalize(
        vmin=np.min(C_self),
        vmax=np.max(C_self)
    )

    # Mutual capacitance normalization (horizontal+vertical)
    mutual_vals = np.concatenate([C_horiz.flatten(), C_vert.flatten()])
    norm_mut = Normalize(
        vmin=np.min(mutual_vals),
        vmax=np.max(mutual_vals)
    )

    # =====================================================
    #                PLOTTING
    # =====================================================
    fig, ax = plt.subplots(figsize=(8, 8))

    # -------------------------
    # Self-cap blocks
    # -------------------------
    for y in range(n):
        for x in range(n):
            xL = x - 0.25
            xR = x + 0.25
            yB = y - 0.25
            yT = y + 0.25

            ax.pcolormesh(
                [xL, xR],
                [yB, yT],
                np.array([[C_self[y, x]]]),
                cmap="inferno",
                shading="auto",
                norm=norm_self
            )

    # -------------------------
    # Horizontal couplings
    # -------------------------
    for y in range(n):
        for x in range(n - 1):
            xL = x + 0.25
            xR = x + 0.75
            yB = y - 0.25
            yT = y + 0.25

            ax.pcolormesh(
                [xL, xR],
                [yB, yT],
                np.array([[C_horiz[y, x]]]),
                cmap="viridis",
                shading="auto",
                norm=norm_mut
            )

    # -------------------------
    # Vertical couplings
    # -------------------------
    for y in range(n - 1):
        for x in range(n):
            xL = x - 0.25
            xR = x + 0.25
            yB = y + 0.25
            yT = y + 0.75

            ax.pcolormesh(
                [xL, xR],
                [yB, yT],
                np.array([[C_vert[y, x]]]),
                cmap="viridis",
                shading="auto",
                norm=norm_mut
            )

    # =====================================================
    # Formatting
    # =====================================================
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(alpha=0.2)
    ax.scatter(
        np.arange(n).repeat(n),  # x coords: 0,0,0...,1,1,1..., ...
        np.tile(np.arange(n), n),  # y coords: 0,1,2,...,0,1,2,...
        c="black", s=8, zorder=10
    )

    ax.set_title("$C^{-1}$ matrix, black dots are the sites/islands")

    # Colorbars (one for each norm)
    sm_self = plt.cm.ScalarMappable(norm=norm_self, cmap="inferno")
    sm_mut = plt.cm.ScalarMappable(norm=norm_mut, cmap="viridis")

    cbar1 = fig.colorbar(sm_self, ax=ax, fraction=0.046, pad=0.16)
    cbar1.set_label("Self capacitance $C^{-1}_{ii}$")

    cbar2 = fig.colorbar(sm_mut, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label("Mutual capacitance $C^{-1}_{ij}$")

    plt.tight_layout()
    if show:
        plt.show()