import csv
import warnings
from pathlib import Path
import matplotlib

matplotlib.use("Agg") # for clustrer, TkAgg for Pc
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import time
from define_objects import ExperimentInitialState
import pandas as pd
from pathlib import Path
from matplotlib import rcParams

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
                 tot_error_count: int,
                 gap_ratio: float,
                 ):
    with open(results_path / f"parameters_{filename}_rep{repetition}.txt", "w") as f:
        f.write(f"repetition {repetition}\n")
        f.write(f"gap ratio : {gap_ratio}" + "\n")
        f.write("---------------------------------------------" + "\n")
        f.write("\n")
        f.write("\n")
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
        f.write("<R> : " + str(np.mean(init.R_t_ij)) + ", std(Rt_ij) : " + str(np.std(init.R_t_ij)) + "\n")
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
    plt.title(f"at V = {heatmap_at_V}\n Ix scale = {round(np.abs(Jx).max(),1)} ;"
              f" Iy scale = {round(np.abs(Jy).max(),1)}")
    plt.savefig(fname=results_path / pic_name, dpi=2100, bbox_inches="tight")
    # plt.show()
    return 0


def extract_nn_resistances(R, n, R_sides, near_left, near_right, periodic_y):
    """
    R: full R_ij (n*n by n*n)
    n: grid dimension
    R_sides: resistances to electrodes
    near_left, near_right: electrode island lists
    periodic_y: if True, bottom row is connected to top row
    """

    Rx = np.zeros((n, n + 1))      # horizontal edges
    Ry = np.zeros((n if periodic_y else (n - 1), n))  # vertical edges

    for y in range(n):
        for x in range(n):
            i = x + n * y  # index (x,y)

            if x < n - 1:
                j = (x + 1) + n * y
                Rx[y, x + 1] = R[i, j]

            if y < n - 1:
                j = x + n * (y + 1)
                Ry[y, x] = R[i, j]

            # periodic vertical neighbour (y = n-1 → y = 0)
            if periodic_y and y == n - 1:
                j = x  # top row index is y=0 → i = x
                Ry[n - 1, x] = R[i, j]

    for y in range(n):
        for isle in near_left:
            if isle // n == y:
                Rx[y, 0] = R_sides[isle]

        for isle in near_right:
            if isle // n == y:
                Rx[y, n] = R_sides[isle]

    return Rx, Ry



def plot_resistance_maps(Rx, Ry, n, results_path, show: bool):
    # Horizontal edges
    plt.figure(figsize=(6, 4))
    plt.title("Horizontal resistances $R_x$")
    plt.imshow(Rx, cmap='inferno', origin='lower', extent=[0, Rx.shape[1], -0.5, Rx.shape[0]-0.5])
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
    plt.imshow(Ry, cmap='inferno', origin='lower', extent=[-0.5, Ry.shape[1]-0.5, 0, Ry.shape[0]])
    plt.colorbar(label="Resistance")
    plt.xlabel("x")
    plt.ylabel("y (edge start)")
    plt.xticks(range(n))
    plt.yticks(list(range(n + 1)), [str(i) for i in range(n)] + ["0"])
    plt.tight_layout()
    plt.savefig(fname=results_path / "_Ry", dpi=2100, bbox_inches="tight")
    if show:
        plt.show()
    print("saved Ry plot")


def plot_capacitance_map(C_inv, n, periodic_y: bool, show: bool, results_path: Path):
    """
        C_inv is an (n*n, n*n) inverse capacitance matrix.
        n is row_num.
    """
    # Extract blocks
    C_self = np.zeros((n, n))
    C_horiz = np.zeros((n, n - 1))
    C_vert = np.zeros((n, n))  # include periodic edge in last row

    for y in range(n):
        for x in range(n):
            i = y * n + x
            C_self[y, x] = C_inv[i, i]

            # Horizontal C(x,y) => (x+1,y)
            if x < n - 1:
                C_horiz[y, x] = C_inv[i, i + 1]

            # Vertical C(x,y) => (x,y+1)
            if y < n - 1:
                C_vert[y, x] = C_inv[i, (y + 1) * n + x]

    # If periodic_y enabled, fill wrap-around coupling
    if periodic_y:
        for x in range(n):
            i_top = (n - 1) * n + x
            i_bot = x
            C_vert[n - 1, x] = C_inv[i_top, i_bot]  # last row contains periodic coupling

    # colours need norm
    norm_self = Normalize(vmin=np.min(C_self), vmax=np.max(C_self))
    mutual_vals = np.concatenate([C_horiz.flatten(), C_vert.flatten()])
    norm_mut = Normalize(vmin=np.min(mutual_vals), vmax=np.max(mutual_vals))

    fig, ax = plt.subplots(figsize=(8, 8))
    for y in range(n):
        for x in range(n):
            xL, xR = x - 0.25, x + 0.25
            yB, yT = y - 0.25, y + 0.25
            ax.pcolormesh(
                [xL, xR], [yB, yT],
                np.array([[C_self[y, x]]]),
                cmap="inferno", norm=norm_self, shading="auto"
            )
    # horizontal
    for y in range(n):
        for x in range(n - 1):
            xL, xR = x + 0.25, x + 0.75
            yB, yT = y - 0.25, y + 0.25
            ax.pcolormesh(
                [xL, xR], [yB, yT],
                np.array([[C_horiz[y, x]]]),
                cmap="viridis", norm=norm_mut, shading="auto"
            )
    # vertical coupling
    for y in range(n - 1):
        for x in range(n):
            xL, xR = x - 0.25, x + 0.25
            yB, yT = y + 0.25, y + 0.75
            ax.pcolormesh(
                [xL, xR], [yB, yT],
                np.array([[C_vert[y, x]]]),
                cmap="viridis", norm=norm_mut, shading="auto"
            )
    # periodic extra rows
    if periodic_y:
        for x in range(n):
            val = C_vert[n - 1, x]

            xL, xR = x - 0.25, x + 0.25
            yB, yT = -0.75, -0.25
            ax.pcolormesh(
                [xL, xR], [yB, yT],
                np.array([[val]]),
                cmap="viridis", norm=norm_mut, shading="auto"
            )

            yB2, yT2 = n + 0.25, n + 0.75
            ax.pcolormesh(
                [xL, xR], [yB2, yT2],
                np.array([[val]]),
                cmap="viridis", norm=norm_mut, shading="auto"
            )

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-1.0, n) if periodic_y else ax.set_ylim(-0.5, n - 0.5)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(alpha=0.3)

    # Black dots marking node positions
    xs = np.arange(n).repeat(n)
    ys = np.tile(np.arange(n), n)
    ax.scatter(xs, ys, c="black", s=10, zorder=10)

    ax.set_title("Inverse Capacitance Map $C^{-1}_{ij}$")

    sm_self = plt.cm.ScalarMappable(norm=norm_self, cmap="inferno")
    sm_mut = plt.cm.ScalarMappable(norm=norm_mut, cmap="viridis")

    fig.colorbar(sm_self, ax=ax, fraction=0.046, pad=0.16).set_label("Self capacitance $C^{-1}_{ii}$")
    fig.colorbar(sm_mut, ax=ax, fraction=0.046, pad=0.04).set_label("Mutual capacitance $C^{-1}_{ij}$")

    plt.tight_layout()
    if isinstance(results_path, Path):
        plt.savefig(fname=results_path / "_Cinv", dpi=2100, bbox_inches="tight")
    else:
        warnings.warn("NO RESULTS PATH FOR CAPACITANCE MAP")
    if show:
        plt.show()

# ==========================================
# 1. KC's Strict Formatting Guidelines
# ==========================================
# Use Myriad Pro (Ensure it is installed on your OS, otherwise falls back to Arial)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Myriad Pro', 'Arial']

# Ensure true vector font rendering in PDF
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# Global Axis and Tick Formatting (0.5 pt for boxes/axes so they don't dominate)
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5
rcParams['xtick.minor.width'] = 0.5
rcParams['ytick.minor.width'] = 0.5

# Font Sizes (KC's ~20-25% larger scaling ratio)
rcParams['axes.titlesize'] = 25
rcParams['axes.labelsize'] = 22
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 18


# ==========================================
# 2. Compute and Plot Function
# ==========================================

def thermopower_curve_compute_and_save_csv(init, filename: str, results: list, V_sweep: np.ndarray, repetition: int,
                                           results_path: Path):
    """
    Parses SteadyStateVaryVResult objects, calculates S(V), saves to CSV,
    and generates a strict, publication-ready PDF plot.
    """
    DeltaV_matrix = []
    I_baseline_matrix = []

    successful_loops = 0
    failed_loops = 0

    # Unpack the results
    for res in results:
        if res is None:
            continue
        # Drop instances that failed to converge in the feedback loop
        if res.error_count > 10:
            failed_loops += 1
            continue

        DeltaV_matrix.append(res.DeltaV_vec)
        I_baseline_matrix.append(res.I_baseline_vec)
        successful_loops += 1

    print(f"Aggregation complete: {successful_loops} successful loops, {failed_loops} dropped due to instability.")

    if not DeltaV_matrix:
        print("CRITICAL ERROR: No valid data to save.")
        return

    # Convert to 2D numpy arrays
    DeltaV_matrix = np.array(DeltaV_matrix)
    I_baseline_matrix = np.array(I_baseline_matrix)

    # Statistics
    DeltaV_avg = np.mean(DeltaV_matrix, axis=0)
    DeltaV_err = np.std(DeltaV_matrix, axis=0) / np.sqrt(successful_loops)  # SEM

    I_baseline_avg = np.mean(I_baseline_matrix, axis=0)

    # Calculate Thermopower S(V) = - DeltaV / dT_total
    T_std = repetition * init.T0 / 20
    dT_total = (init.row_num - 1) * T_std
    Seebeck_coeff = -DeltaV_avg / dT_total
    Seebeck_err = DeltaV_err / dT_total

    # Package into a DataFrame
    df = pd.DataFrame({
        "V_baseline_(V)": V_sweep,
        "DeltaV_avg_(V)": DeltaV_avg,
        "DeltaV_err_(V)": DeltaV_err,
        "I_baseline_avg_(e/s)": I_baseline_avg,
        "Thermopower_S(V)": Seebeck_coeff,
        "Thermopower_err": Seebeck_err
    })

    # Save to CSV
    csv_filename = results_path / f"VaryV_rep{repetition}_data.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Successfully saved Vary-V data to {csv_filename}")

    # Generate the Publication Plot
    _plot_publication_sv(V_sweep, Seebeck_coeff, Seebeck_err, I_baseline_avg, init.Cg[0], dT_total, repetition,
                         results_path)


def _plot_publication_sv(V, S, S_err, I_base, Cg, dT_total, rep, path):
    """Generates a strict, publication-ready PDF using KC's aesthetic rules."""

    # Setup Figure without excess whitespace
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # --- Primary Axis: Thermopower S(V) ---
    color_s = 'tab:red'
    # LaTeX formatting for labels using your specific unit style
    ax1.set_xlabel(r'Baseline Voltage Bias $V$ $\left[ \frac{e}{\langle C \rangle} \right]$', labelpad=15)
    ax1.set_ylabel(r'Thermopower $S(V)$ $\left[ \frac{k_B}{e} \right]$', color=color_s, labelpad=15)

    # Using 1.5pt "Goldilocks" linewidth
    line1 = ax1.errorbar(V, S, yerr=S_err, fmt='-o', color=color_s,
                         linewidth=1.5, elinewidth=1.5, markersize=6, capsize=3,
                         label=r'$S(V)$')
    ax1.tick_params(axis='y', labelcolor=color_s, length=6, width=0.5)
    ax1.tick_params(axis='x', length=6, width=0.5)

    # --- Secondary Axis: Current I(V) ---
    ax2 = ax1.twinx()
    color_i = 'tab:blue'
    ax2.set_ylabel(r'Current $I(V)$ $\left[ \frac{e}{\langle R \rangle \langle C \rangle} \right]$', color=color_i, labelpad=15)

    # Using 1.5pt "Goldilocks" linewidth
    line2, = ax2.plot(V, I_base, '--', color=color_i, linewidth=1.5, alpha=0.8, label=r'$I(V)$')
    ax2.tick_params(axis='y', labelcolor=color_i, length=6, width=0.5)

    # Since twinx adds a new set of spines, we must enforce KC's 0.5pt rule here too
    for spine in ax2.spines.values():
        spine.set_linewidth(0.5)

    # --- Title and Legend ---
    # Formatting title to include relevant parameters cleanly (No "Rep" in title per your request)
    ax1.set_title(f'Thermopower & Current vs. Voltage Bias\n$C_g={Cg}, \\Delta T={dT_total:.4f}$', pad=20)

    # Combine legends from both axes
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels, loc='best', frameon=True, edgecolor='black')

    # Enforce KC's 0.5 pt legend border rule
    legend.get_frame().set_linewidth(0.5)

    # Clean layout to remove whitespace
    fig.tight_layout()

    # Save as strictly vector PDF
    pdf_path = path / f"Publication_VaryV_rep{rep}.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"Publication vector PDF saved to: {pdf_path.name}")
