import os
import sys

# --- Cluster Environment Configurations --- BEFORE NUMPY IMPORT
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
num_workers = max(total_cpus - 2, 1)
print(f"Worker number set to {num_workers} for {total_cpus} CPUs", flush=True)

import gc
import math
import numpy as np
from mpmath import mp, exp, sqrt
import mpmath
import multiprocessing
from pathlib import Path
import datetime
from scipy.linalg import eig

# --- Parameters ---
DPS = 30  # Global precision parameter
mp.dps = DPS

N_states = 120  # Global state truncation

# System physics variables
e = 1
Vr = 0
Cl = 2
Cr = 0.5
Rl = 1
Rr = 10
Rg = 100 * Rr
Cg = 10 * Cl
Cs = Cg + Cl + Cr
Ec = mp.mpf(str((e ** 2) / (2 * Cg)))


# ============================================================================
# ENERGY TRANSITION HELPERS
# ============================================================================

def Qn(Vl, n):
    """Calculates the induced charge on the dot at state n."""
    return -e * (n - N_states / 2) + Cl * Vl + Cr * Vr


def W(n, qn, Vl, dn, side):
    """
    Calculates the electrostatic energy difference (dE) for a tunneling event.
    """
    if side == "left":
        voltage = Vl
    elif side == "right":
        voltage = Vr
    else:
        raise ValueError("Invalid side specified. Use 'left' or 'right'.")

    # The charging energy penalty formula
    dE = -dn * e * voltage + (dn * e) ** 2 / (2 * Cs) + (dn * e) * qn / Cs
    return mp.mpf(str(dE))


# ============================================================================
# NORMAL METAL MASTER INTEGRATION ALGORITHMS
# ============================================================================

def integrand(T, dE, Ec_val):
    """
    P- function for high impedance.
    :param dE: Energy difference == dE.
    :param Ec_val: Electrostatic energy of environment == Ec.
    :param T: Temperature.
    :return: P(E)*E*f_BE(-E)
    """

    def conv(E):
        if mp.fabs(E) < 1e-8:
            zero_limit_gauss = mp.exp(-((dE + Ec_val) ** 2) / (4 * Ec_val * T))
            return zero_limit_gauss * mp.sqrt(T / (4 * mp.pi * Ec_val))

        gauss = mp.exp(-((E + dE + Ec_val) ** 2) / (4 * Ec_val * T))
        gauss = gauss / mp.sqrt(mp.pi * 4 * Ec_val * T)

        bose_mean = E / (1 - mp.exp(-E / T))

        return bose_mean * gauss

    return conv


def _calc_segments(val_str, temp_str, Ec_str, dps):
    """
    Top-level worker function to calculate segmented probabilities for Normal Metals.
    """
    mp.dps = dps
    val = mp.mpf(val_str)
    temp = mp.mpf(temp_str)
    Ec_val = mp.mpf(Ec_str)

    printing = np.random.uniform(0, 1)
    if printing < 0.01:  # print ~1 in a 100
        print(f"START calculating w = {float(val):.3f} [T={float(temp)}]", flush=True)

    func = integrand(temp, val, Ec_val)
    absval = abs(val + Ec_val)

    # Updated limits using mpmath's infinity
    limits = [-mp.inf, -absval, mp.mpf('0'), absval, mp.inf]

    probability = mp.mpf('0')

    for i in range(len(limits) - 1):
        a = limits[i]
        b = limits[i + 1]

        if a == -mp.inf:
            segment_prob = mp.quad(lambda t, b=b: func(b - t / (mp.mpf('1') - t)) / ((mp.mpf('1') - t) ** 2), [0, 1],
                                   method='tanh-sinh')
        elif b == mp.inf:
            segment_prob = mp.quad(lambda t, a=a: func(a + t / (mp.mpf('1') - t)) / ((mp.mpf('1') - t) ** 2), [0, 1],
                                   method='tanh-sinh')
        else:
            w = b - a
            if w < mp.mpf('1e-8'):
                continue
            segment_prob = mp.quad(lambda t, a=a, w=w: func(a + t * w) * w, [0, 1])

        probability += segment_prob

    return [float(val), float(probability.real), float(temp), float(Ec_val)]


def Gamma(dE, T, resistance):
    """Wrapper to map exact diag Gamma calls to the normal metal integrator."""
    # Run the integration
    res_list = _calc_segments(str(dE), str(T), str(Ec), DPS)

    # Extract probability (index 1) and calculate the rate multiplier (1 / (e^2 R_T))
    prob = res_list[1]
    rate = prob / (float(e ** 2) * float(resistance))
    return rate


# ============================================================================
# EXACT DIAGONALIZATION ROUTINE (TRIDIAGONAL / METAL ONLY)
# ============================================================================

def calculate_current(Vl, N, T_l, T_r, Tdot):
    G_L_plus = np.zeros(N + 1)
    G_R_plus = np.zeros(N + 1)
    G_L_minus = np.zeros(N + 1)
    G_R_minus = np.zeros(N + 1)

    for n in range(N + 1):
        # 1e Hopping Rates Only
        G_L_plus[n] = Gamma(W(n, Qn(Vl, n), Vl, 1, "left"), T_l, Rl)
        G_L_minus[n] = Gamma(W(n, Qn(Vl, n), Vl, -1, "left"), Tdot, Rl)
        G_R_plus[n] = Gamma(W(n, Qn(Vl, n), Vl, 1, "right"), T_r, Rr)
        G_R_minus[n] = Gamma(W(n, Qn(Vl, n), Vl, -1, "right"), Tdot, Rr)

        gc.collect()

    G_plus = G_L_plus + G_R_plus
    G_minus = G_L_minus + G_R_minus

    # Enforce truncation boundaries
    G_plus[-1] = 0.0
    G_minus[0] = 0.0

    # Tridiagonal Matrix Assembly
    diag_main = -(G_plus + G_minus)
    diag_sub = G_plus[:-1]
    diag_super = G_minus[1:]

    M = (np.diag(diag_main) +
         np.diag(diag_sub, k=-1) +
         np.diag(diag_super, k=1))

    # Solve the Master Equation
    eigenvalues, eigenvectors = eig(M)
    zero_idx = np.argmin(np.abs(eigenvalues))
    p_stat = np.real(eigenvectors[:, zero_idx])
    p_stat = p_stat / np.sum(p_stat)

    # Net Current = (Probability of state n) * (Left Hop Rightwards - Left Hop Leftwards)
    current_1e = e * np.sum(p_stat * (G_L_plus - G_L_minus))

    return current_1e


def worker_single_point(task_args):
    # Unpack the new flip parameter
    m, v_idx, V, N_states_task, T0, cp_dir, flip = task_args

    # Apply the temperature gradient logic based on flip state
    if flip:
        T_left = T0 + 20 * m * T0
        T_dot = T0 + 10 * m * T0
        T_right = T0
    else:
        T_left = T0
        T_dot = T0 + 10 * m * T0
        T_right = T0 + 20 * m * T0

    print(f"Worker computing grad {m}, Vl = {V:.3f} V, flip = {flip}...", flush=True)
    I = calculate_current(V, N=N_states_task, T_l=T_left, T_r=T_right, Tdot=T_dot)

    # Checkpoint saving mechanism - added flip marker to prevent overwriting
    cp_path = Path(cp_dir) / f"grad_{m}_vidx_{v_idx}_flip_{flip}.npy"
    np.save(cp_path, I)

    print(f"Worker DONE computing grad {m}, Vl = {V:.3f} V | I = {I:.5e}", flush=True)
    return True


# ============================================================================
# MAIN CLUSTER SLICING ROUTINE
# ============================================================================

if __name__ == '__main__':
    base_folder = Path(__file__).parent.absolute()
    checkpoint_dir = base_folder / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Simulation Constraints
    T0 = 0.001
    num_points = 101
    V_vals = np.linspace(0, 2, num_points)
    num_of_grads = 20
    flip = True  # Set to True or False to control the gradient direction

    all_tasks = []
    for m in range(num_of_grads):
        for v_idx, V in enumerate(V_vals):
            # Append flip to the task arguments
            all_tasks.append((m, v_idx, V, N_states, T0, checkpoint_dir, flip))

    total_tasks = len(all_tasks)

    # Slurm chunk routing slicing parameters
    try:
        raw_chunk_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        num_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    except ValueError:
        raw_chunk_id = 0
        num_chunks = 1

    # BULLETPROOF LOGIC: Force bounds
    num_chunks = max(1, num_chunks)
    chunk_id = max(0, min(raw_chunk_id, num_chunks - 1))

    # --- PERFECT SLICING VIA NUMPY ---
    split_tasks = np.array_split(all_tasks, num_chunks)
    my_tasks = split_tasks[chunk_id].tolist()

    # Calculate actual indices for logging purposes
    start_idx = sum(len(split_tasks[i]) for i in range(chunk_id))
    end_idx = start_idx + len(my_tasks)

    # --- Logging Block ---
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    param_filename = base_folder / f"parameters_DPS{DPS}_chunk{chunk_id}_{timestamp_str}.txt"
    with open(param_filename, mode='w', encoding='utf-8') as pf:
        pf.write(f"--- Normal Metal Exact Diagonalization Log ---\n")
        pf.write(f"Chunk ID               : {chunk_id} (of {num_chunks} total chunks)\n")
        pf.write(f"Global Precision (DPS) : {DPS}\n")
        pf.write(f"ProcessPool Workers    : {num_workers}\n")
        pf.write(f"Total Tasks in Chunk   : {len(my_tasks)}\n")
        pf.write(f"N_states               : {N_states}\n")
        pf.write(f"T0                     : {T0}\n")
        pf.write(f"Cg                     : {Cg}\n")
        pf.write(f"Ec                     : {float(Ec):.5e}\n")
        pf.write(f"flip                   : {flip}\n")  # Added to logs

    print(f"Slicing Task Group: processing tasks {start_idx} to {end_idx} (Total: {len(my_tasks)}) on Pool.",
          flush=True)

    if len(my_tasks) > 0:
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(worker_single_point, my_tasks)
        print("Slice complete.", flush=True)
    else:
        print("ERROR: Slice resulted in 0 tasks. Check your array geometry.", flush=True)