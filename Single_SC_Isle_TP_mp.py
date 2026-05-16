import os

ratio = 10/9
os.environ["OPENBLAS_NUM_THREADS"] = str(ratio)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
num_workers = int(total_cpus / ratio) - 1
print(f"worker number set to {num_workers} ; for {total_cpus} cpus", flush=True)

import csv
import numpy as np
from mpmath import quad, mp, exp, sqrt
import matplotlib
import datetime
import mpmath

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eig

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# --- Global Parameters ---
e = 1
Vr = 0
Cl = 2
Cr = 0.01
Rl = 10
Rr = 1  # R2
Rg = 1000 * (Cl + Cr)
Cg = 10 * (Cl + Cr)
Cs = Cg + Cl + Cr

# --- Added for Cooper Pairs ---
Ec = (e ** 2) / (2 * Cg)
Ej = Ec*0.2  # Adjust this to match your system's Josephson energy


def Gamma_cp(dE, T, Ec, Ej):
    # P(-dE)
    gauss = mp.exp(-((dE + Ec) ** 2) / (4 * Ec * T))
    gauss = gauss / mp.sqrt(mp.pi * 4 * Ec * T)
    # Casting to float to ensure numpy matrix compatibility later
    return float(gauss * Ej * Ej * mp.pi)


def qs_integrand(T, dE, Ec, D):
    def conv(E, Etag):
        n_E = dos(E, D)
        if n_E == 0: return 0

        n_Etag = dos(Etag - dE, D)
        if n_Etag == 0: return 0

        gauss = exp(-((E - Etag - Ec) ** 2) / (4 * Ec * T))
        gauss = gauss / sqrt(np.pi * 4 * Ec * T)

        return n_E * n_Etag * f(E, T) * (1 - f(Etag - dE, T)) * gauss

    return conv


def f(x, t):
    if x / t > 1e10:
        return exp(-x / t)
    if x / t < -1e10:
        return 1
    expon = exp(x / t)
    return 1 / (1 + expon)


def dos(E, D):
    if mpmath.fabs(E) <= D:
        return 0
    val = E * E - D * D
    if val <= 0:
        return 0
    return mpmath.fabs(E) / sqrt(val)


def Gamma(w, T, Rt, mu=Ec):
    D = 0.2 * mu
    mp.dps = 50
    func = qs_integrand(T, w, mu, D)
    absval = abs(D)

    # Calculate the exact physical width of the Gaussian spike
    sigma = mp.sqrt(2 * mu * T)
    bracket_width = 5 * sigma

    limits_E = [-mp.inf, -absval, 0, absval, mp.inf]

    def get_mapping(a, b):
        if a == -mp.inf:
            return lambda t: (b - t / (1 - t), 1 / ((1 - t) ** 2))
        elif b == mp.inf:
            return lambda t: (a + t / (1 - t), 1 / ((1 - t) ** 2))
        else:
            width = b - a
            if width < 1e-8:
                return None
            return lambda t: (a + t * width, width)

    mappings_E = [get_mapping(limits_E[i], limits_E[i + 1]) for i in range(len(limits_E) - 1)]

    probability = 0
    print(f"  [T={T}] Calculating for w = {w:.3f}", flush=True)

    for m_E in mappings_E:
        if m_E is None: continue

        # Outer integral over E
        def outer_integrand(t_E, m_E=m_E):
            if t_E <= 0 or t_E >= 1:
                return 0

            E, jac_E = m_E(t_E)

            # Dynamically calculate where the Gaussian spike is in E'
            peak_center = E - mu

            # Combine fixed topological singularities with the dynamic Gaussian boundaries
            # mpmath uses its own internal float types, so we explicitly convert w and absval
            dynamic_limits = [
                -mp.inf,
                mp.mpf(w - absval),
                mp.mpf(w),
                mp.mpf(w + absval),
                peak_center - bracket_width,
                peak_center + bracket_width,
                mp.inf
            ]

            # Sort and remove duplicates to create a clean piecewise integration path
            sorted_Etag_limits = sorted(list(set(dynamic_limits)))

            # Run the inner integral over E', letting mpmath natively handle the piecewise segments
            inner_integral = mp.quad(lambda Etag: func(E, Etag), sorted_Etag_limits, method='tanh-sinh', maxdegree=7)

            return inner_integral * jac_E

        # Integrate the mapped outer function
        segment_prob = mp.quad(outer_integrand, [0, 1], method='tanh-sinh', maxdegree=7)
        probability += segment_prob

    print(f"  [T={T}] DONE Calculating for w = {w:.3f}", flush=True)
    return probability


def U(n, Qg, Vl):
    return (Qg + n * e + Cl * Vl) / (Cl + Cr)


def Qn(Vl, n):
    return -Cg * (Cl * Vl + n * e) / Cs


def W(n, Qg, Vl, in_out, left_right):
    # Modified to allow 2e Cooper Pair transfers
    if abs(in_out) not in [1, 2]:
        raise ValueError("in_out must be +/-1 (quasiparticle) or +/-2 (Cooper pair)")

    if left_right == "left":
        return in_out * e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2 - in_out * e * Vl
    elif left_right == "right":
        return in_out * e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2
    else:
        raise ValueError("left_right must be either 'left' or 'right'")


def calculate_current(Vl, N, T_l, T_r, Tdot):
    # Arrays for 1e transitions
    G_L_plus = np.zeros(N + 1)
    G_R_plus = np.zeros(N + 1)
    G_L_minus = np.zeros(N + 1)
    G_R_minus = np.zeros(N + 1)

    # Arrays for 2e transitions (Cooper Pairs)
    G_L_plus2 = np.zeros(N + 1)
    G_R_plus2 = np.zeros(N + 1)
    G_L_minus2 = np.zeros(N + 1)
    G_R_minus2 = np.zeros(N + 1)

    for n in range(N + 1):
        # 1e (Quasiparticle) transitions
        G_L_plus[n] = Gamma(W(n, Qn(Vl, n), Vl, 1, "left"), T_l, Rl)
        G_L_minus[n] = Gamma(W(n, Qn(Vl, n), Vl, -1, "left"), Tdot, Rl)
        G_R_plus[n] = Gamma(W(n, Qn(Vl, n), Vl, 1, "right"), T_r, Rr)
        G_R_minus[n] = Gamma(W(n, Qn(Vl, n), Vl, -1, "right"), Tdot, Rr)

        # 2e (Cooper Pair) transitions
        if n + 2 <= N:
            G_L_plus2[n] = Gamma_cp(W(n, Qn(Vl, n), Vl, 2, "left"), T_l, Ec, Ej)
            G_R_plus2[n] = Gamma_cp(W(n, Qn(Vl, n), Vl, 2, "right"), T_r, Ec, Ej)
        if n - 2 >= 0:
            G_L_minus2[n] = Gamma_cp(W(n, Qn(Vl, n), Vl, -2, "left"), Tdot, Ec, Ej)
            G_R_minus2[n] = Gamma_cp(W(n, Qn(Vl, n), Vl, -2, "right"), Tdot, Ec, Ej)

    # Combine L/R probabilities
    G_plus = G_L_plus + G_R_plus
    G_minus = G_L_minus + G_R_minus
    G_plus2 = G_L_plus2 + G_R_plus2
    G_minus2 = G_L_minus2 + G_R_minus2

    # Boundary conditions
    G_plus[-1] = 0.0
    G_minus[0] = 0.0
    G_plus2[-1], G_plus2[-2] = 0.0, 0.0
    G_minus2[0], G_minus2[1] = 0.0, 0.0

    # Total rate leaving state n
    diag_main = -(G_plus + G_minus + G_plus2 + G_minus2)

    # Off-diagonal elements for the master equation
    diag_sub = G_plus[:-1]  # n -> n+1 (k=-1)
    diag_super = G_minus[1:]  # n -> n-1 (k=+1)
    diag_sub2 = G_plus2[:-2]  # n -> n+2 (k=-2)
    diag_super2 = G_minus2[2:]  # n -> n-2 (k=+2)

    # Construct Matrix
    M = (np.diag(diag_main) +
         np.diag(diag_sub, k=-1) +
         np.diag(diag_super, k=1) +
         np.diag(diag_sub2, k=-2) +
         np.diag(diag_super2, k=2))

    eigenvalues, eigenvectors = eig(M)
    zero_idx = np.argmin(np.abs(eigenvalues))
    p_stat = np.real(eigenvectors[:, zero_idx])
    p_stat = p_stat / np.sum(p_stat)

    # Current includes 1e and 2e contributions
    current_1e = e * np.sum(p_stat * (G_L_plus - G_L_minus))
    current_2e = 2 * e * np.sum(p_stat * (G_L_plus2 - G_L_minus2))

    return current_1e + current_2e


# ==========================================
# Worker Function for the Pool Executor
# ==========================================
def worker_simulate_gradient(multiplier, V_vals, N_states, T0):
    print(f"Worker started for grad {multiplier}", flush=True)

    T_left = T0
    T_dot = T0 + 10 * multiplier * T0
    T_right = T0 + 20 * multiplier * T0

    currents = []
    for V in V_vals:
        print(f"In grad {multiplier}, calculating for Vl = {V:.2f} V...")
        I = calculate_current(V, N=N_states, T_l=T_left, T_r=T_right, Tdot=T_dot)
        currents.append(I)

    print(f"Worker finished for grad {multiplier}", flush=True)

    return {
        "multiplier": multiplier,
        "T_left": T_left,
        "T_dot": T_dot,
        "T_right": T_right,
        "currents": currents
    }


# ==========================================
# Post-Processing: Export & Plot
# ==========================================
def export_and_plot(results, output_dir, V_vals, T0):
    results = sorted(results, key=lambda x: x["multiplier"])

    csv_path = output_dir / "IV_data_all_grads.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        headers = ["Vl (V)"] + [f"Grad_{res['multiplier']}_I" for res in results]
        writer.writerow(headers)
        for i in range(len(V_vals)):
            row = [V_vals[i]] + [res["currents"][i] for res in results]
            writer.writerow(row)

    print(f"Data exported to {csv_path}")

    for res in results:
        m = res["multiplier"]
        I_vals = res["currents"]
        T_l, T_r = res["T_left"], res["T_right"]

        plt.figure(figsize=(8, 6))
        plt.plot(V_vals, I_vals, linestyle='-', color='r')
        plt.title(f"I-V Characteristic with $\\Delta T$ ($T_l={T_l / T0:.2f}T_0$, $T_r={T_r / T0:.2f}T_0$)",
                  fontsize=14)
        plt.xlabel("Left Voltage $V_l$ (V)", fontsize=12)
        plt.ylabel("Steady-State Current I L->R", fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        plot_path = output_dir / f"IV_plot_grad_{m}.png"
        plt.savefig(plot_path)
        plt.close()

    print(f"All {len(results)} plots saved in {output_dir}/")


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == '__main__':
    date_ = datetime.datetime.now()
    run_name_flat = date_.strftime("%Y%m%d_%Hh%Mm%Ss")

    job_id = os.environ.get("SLURM_JOB_ID", "local_run")
    output_folder = Path(__file__).parent / f"SingleIsleTP_Job{job_id}__{run_name_flat}"
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to folder: {output_folder.absolute()}")

    T0 = 0.001
    N_states = 120
    num_points = 101
    V_vals = np.linspace(0, 4, num_points)
    num_of_grads = 20

    # export params
    params_path = output_folder / "parameters.txt"
    with open(params_path, "w") as f:
        f.write(f"e :  {e}\n")
        f.write(f"Vr :  {Vr}\n")
        f.write(f"Cl :  {Cl}\n")
        f.write(f"Cr :  {Cr}\n")
        f.write(f"Rl :  {Rl}\n")
        f.write(f"Rr :  {Rr}\n")
        f.write(f"Rg :  {Rg}\n")
        f.write(f"Cg :  {Cg}\n")
        f.write(f"Cs :  {Cs}\n")
        f.write(f"Ec :  {Ec}\n")
        f.write(f"Ej :  {Ej}\n")
        f.write(f"T0 :  {T0}\n")
        f.write(f"N_states :  {N_states}\n")
        f.write(f"num_points :  {num_points}\n")
        f.write(f"num_of_grads :  {num_of_grads}\n")
        f.write(f"job_id :  {job_id}\n")
    print(f"Parameters saved to {params_path}")

    loaded_state_function = partial(
        worker_simulate_gradient,
        V_vals=V_vals,
        N_states=N_states,
        T0=T0
    )

    print(f"Initializing ProcessPoolExecutor...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        actual_workers = executor._max_workers
        print(f"Running pool with {actual_workers} workers", flush=True)

        results_list = list(executor.map(loaded_state_function, range(num_of_grads)))

    export_and_plot(results_list, output_folder, V_vals, T0)
    print("Execution complete.")