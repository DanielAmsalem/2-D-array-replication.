import os

ratio = 10 / 9
os.environ["OPENBLAS_NUM_THREADS"] = str(ratio)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
num_workers = max(5, int(total_cpus / ratio))
print(f"worker number set to {num_workers} ; for {total_cpus} cpus", flush=True)

import sys
import csv
import numpy as np
from mpmath import quad, mp, exp, sqrt
import matplotlib
import datetime

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eig

# 1) Import the necessary modules for paths and multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

e = 1
Vr = 0
Cl = 2
Cr = 0.01
Rl = 10
Rr = 1  # R2
Rg = 1000 * (Cl + Cr)
Cg = 10 * (Cl + Cr)
Cs = Cg + Cl + Cr


def integrand(T, dE, Ec):
    def conv(E):
        if np.abs(E) < 1e-8:
            zero_limit_gauss = exp(-((dE + Ec) ** 2) / (4 * Ec * T))
            return zero_limit_gauss * sqrt(T / (4 * np.pi * Ec))

        gauss = exp(-((E + dE + Ec) ** 2) / (4 * Ec * T))
        gauss = gauss / sqrt(np.pi * 4 * Ec * T)

        bose_mean = E / (1 - exp(-E / T))
        return bose_mean * gauss

    return conv


def Gamma(w, T, Rt, mu=0.5 / Cg):
    absw = abs(w)
    mp.dps = 40
    probability = quad(integrand(T, w, mu), [-6, -absw - 0.2, 0, absw + 0.1, 7])
    mp.dps = 15
    return probability / (Rt * e * e)


def U(n, Qg, Vl):
    return (Qg + n * e + Cl * Vl) / (Cl + Cr)


def Qn(Vl, n):
    return -Cg * (Cl * Vl + n * e) / Cs


def W(n, Qg, Vl, in_out, left_right):
    if abs(in_out) != 1:
        raise ValueError
    if left_right == "left":
        return in_out * e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2 - in_out * e * Vl
    elif left_right == "right":
        return in_out * e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2
    else:
        raise ValueError("left_right must be either 'left' or 'right'")


def calculate_current(Vl, N, T_l, T_r, Tdot):
    G_L_plus = np.zeros(N + 1)
    G_R_plus = np.zeros(N + 1)
    G_L_minus = np.zeros(N + 1)
    G_R_minus = np.zeros(N + 1)

    for n in range(N + 1):
        G_L_plus[n] = Gamma(W(n, Qn(Vl, n), Vl, 1, "left"), T_l, Rl)
        G_L_minus[n] = Gamma(W(n, Qn(Vl, n), Vl, -1, "left"), Tdot, Rl)
        G_R_plus[n] = Gamma(W(n, Qn(Vl, n), Vl, 1, "right"), T_r, Rr)
        G_R_minus[n] = Gamma(W(n, Qn(Vl, n), Vl, -1, "right"), Tdot, Rr)

    G_plus = G_L_plus + G_R_plus
    G_minus = G_L_minus + G_R_minus

    G_plus[-1] = 0.0
    G_minus[0] = 0.0

    diag_main = -(G_plus + G_minus)
    diag_sub = G_plus[:-1]
    diag_super = G_minus[1:]

    M = np.diag(diag_main) + np.diag(diag_sub, k=-1) + np.diag(diag_super, k=1)

    eigenvalues, eigenvectors = eig(M)
    zero_idx = np.argmin(np.abs(eigenvalues))
    p_stat = np.real(eigenvectors[:, zero_idx])
    p_stat = p_stat / np.sum(p_stat)

    current = e * np.sum(p_stat * (G_L_plus - G_L_minus))
    return current


# ==========================================
# Refactored Worker Function (Single Point)
# ==========================================
def worker_compute_single_point(args):
    # Unpack the specific task parameters
    multiplier, v_idx, V, N_states, T0 = args

    dT_max = 80 * T0
    Tmid = (T0 + 81 * T0) / 2
    dT_unit = 0.5 * dT_max / 20

    T_left = Tmid - multiplier * dT_unit
    if T_left < 0:
        print("NEGATIVE TLEFT ?!?!?!?!", flush=True)
        T_left = T0
    T_dot = Tmid
    T_right = T0 + multiplier * dT_unit

    # Compute the single heavy integration
    I = calculate_current(V, N=N_states, T_l=T_left, T_r=T_right, Tdot=T_dot)

    # Return the indices alongside the result so we can reassemble the array safely
    return multiplier, v_idx, I, T_left, T_dot, T_right


# ==========================================
# Post-Processing: Export & Plot
# ==========================================
def export_and_plot(results, output_dir, V_vals, T0):
    # Ensure results are sorted by multiplier
    results = sorted(results, key=lambda x: x["multiplier"])

    # 1. Export unified CSV
    csv_path = output_dir / "IV_data_all_grads.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)

        # Build the header row
        headers = ["Vl (V)"] + [f"Grad_{res['multiplier']}_I" for res in results]
        writer.writerow(headers)

        # Write the data rows
        for i in range(len(V_vals)):
            row = [V_vals[i]] + [res["currents"][i] for res in results]
            writer.writerow(row)

    print(f"Data exported to {csv_path}")

    # 2. Plotting loop
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
        plt.close()  # Vital: close the figure to free up memory

    print(f"All {len(results)} plots saved in {output_dir}/")


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == '__main__':
    date_ = datetime.datetime.now()
    run_name_flat = date_.strftime("%Y%m%d_%Hh%Mm%Ss")

    job_id = os.environ.get("SLURM_JOB_ID", "local_run")
    # Define output folder via pathlib
    output_folder = Path(__file__).parent / f"SingleIsleTP_Job{job_id}__{run_name_flat}"
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to folder: {output_folder.absolute()}")

    T0 = 0.001
    N_states = 120
    num_points = 101
    V_vals = np.linspace(0, 4, num_points)

    # Range of multipliers: 0 to 19 (inclusive) -> 20 total gradients
    num_of_grads = 20

    # Export params
    params_path = output_folder / "parameters.txt"
    with open(params_path, "w") as f:
        # Global variables
        f.write(f"e :  {e}\n")
        f.write(f"Vr :  {Vr}\n")
        f.write(f"Cl :  {Cl}\n")
        f.write(f"Cr :  {Cr}\n")
        f.write(f"Rl :  {Rl}\n")
        f.write(f"Rr :  {Rr}\n")
        f.write(f"Rg :  {Rg}\n")
        f.write(f"Cg :  {Cg}\n")
        f.write(f"Cs :  {Cs}\n")
        # Run-specific variables
        f.write(f"T0 :  {T0}\n")
        f.write(f"N_states :  {N_states}\n")
        f.write(f"num_points :  {num_points}\n")
        f.write(f"num_of_grads :  {num_of_grads}\n")
        f.write(f"job_id :  {job_id}\n")
    print(f"Parameters saved to {params_path}")

    # 1. Build the flat list of ALL 2020 independent tasks
    tasks = []
    for m in range(num_of_grads):
        for v_idx, V in enumerate(V_vals):
            tasks.append((m, v_idx, V, N_states, T0))

    # 2. Prepare a dictionary to securely catch the out-of-order results
    compiled_data = {m: {"currents": [0.0] * num_points} for m in range(num_of_grads)}

    print(f"Initializing ProcessPoolExecutor for {len(tasks)} tasks...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        actual_workers = executor._max_workers
        print(f"Running pool with {actual_workers} workers", flush=True)

        # 3. Execute the map and stitch the data back together as it finishes
        for i, res in enumerate(executor.map(worker_compute_single_point, tasks)):
            m, v_idx, I, T_left, T_dot, T_right = res

            # Slot the current into its exact correct position
            compiled_data[m]["currents"][v_idx] = I
            compiled_data[m]["T_left"] = T_left
            compiled_data[m]["T_dot"] = T_dot
            compiled_data[m]["T_right"] = T_right

            # Print progress every 100 points
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{len(tasks)} points...", flush=True)

    # 4. Reconstruct the exact dictionary format your export_and_plot function expects
    results_list = []
    for m in range(num_of_grads):
        results_list.append({
            "multiplier": m,
            "T_left": compiled_data[m]["T_left"],
            "T_dot": compiled_data[m]["T_dot"],
            "T_right": compiled_data[m]["T_right"],
            "currents": compiled_data[m]["currents"]
        })

    # Process the outputs exactly as before
    export_and_plot(results_list, output_folder, V_vals, T0)
    print("Execution complete.")