import os
import sys
import gc
import math
import numpy as np
from mpmath import mp, exp, sqrt
import mpmath
import multiprocessing
from pathlib import Path
import datetime
from scipy.linalg import eig

# --- Cluster Environment Configurations ---
ratio = 1
os.environ["OPENBLAS_NUM_THREADS"] = str(ratio)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
num_workers = max(int(0.9 * total_cpus / ratio), 1)
print(f"Worker number set to {num_workers} for {total_cpus} CPUs", flush=True)

# --- Parameters ---
DPS = 50  # Global precision parameter
mp.dps = DPS

N_states = 120  # Global state truncation
Cg_val = 5

# System physics variables
e = 1
Vr = 0
Cl = 2
Cr = 0.01
Rl = 10
Rr = 1
Rg = 100 * Rl
Cg = 5
Cs = Cg + Cl + Cr
Ec = mp.mpf(str((e ** 2) / (2 * Cg)))
D_gap = mp.mpf('2') * Ec  # Superconducting gap Delta


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
# MASTER INTEGRATION ALGORITHMS
# ============================================================================

def Gamma_cp(dE, T, Ec_val, Rt):
    """Cooper pair transition rate calculation using local Ambegaokar-Baratoff Ej."""
    # Convert inputs strictly to 50-dps mpmath objects
    dE_mp = mp.mpf(str(dE))
    T_mp = mp.mpf(str(T))
    Rt_mp = mp.mpf(str(Rt))
    Ec_mp = mp.mpf(str(Ec_val))

    # Ej = (hbar/2eRt)(pi*gap/2e)*tanh
    # set h=1, e=1 -> hbar = 1/2pi
    # Ej = tanh * gap / 8Rt
    tanh_val = mp.tanh(D_gap / (mp.mpf('2') * T_mp))
    Ej = tanh_val * D_gap / (mp.mpf('8') * Rt_mp)

    # Calculate P(E) Gaussian broadening
    gauss = mp.exp(-((dE_mp + Ec_mp) ** 2) / (mp.mpf('4') * Ec_mp * T_mp))
    gauss = gauss / mp.sqrt(mp.pi * mp.mpf('4') * Ec_mp * T_mp)

    return float(gauss * Ej * Ej * mp.pi)


def dos(E, D):
    if mpmath.fabs(E) <= D:
        return mp.mpf('0')
    val = E * E - D * D
    if val <= 0:
        return mp.mpf('0')
    return mpmath.fabs(E) / sqrt(val)


def f(x, t):
    if x / t > mp.mpf('1e50'):
        return exp(-x / t)
    if x / t < mp.mpf('-1e50'):
        return mp.mpf('1')
    expon = exp(x / t)
    return mp.mpf('1') / (mp.mpf('1') + expon)


def qp_integrand(T, dE, Ec_val, D):
    def conv(E, Etag):
        n_E = dos(E, D)
        if n_E == mp.mpf('0'):
            return mp.mpf('0')

        n_Etag = dos(Etag - dE, D)
        if n_Etag == mp.mpf('0'):
            return mp.mpf('0')

        gauss = exp(-((E - Etag - Ec_val) ** 2) / (4 * Ec_val * T))
        gauss = gauss / sqrt(mp.pi * 4 * Ec_val * T)

        return n_E * n_Etag * f(E, T) * (mp.mpf('1') - f(Etag - dE, T)) * gauss

    return conv


def _calc_segments_gapped_master(args, dps):
    """
    Advanced adaptive segmented integration mapping that routes around density-of-states
    singularities to evaluate quasiparticle rates down to 50 decimal digits of precision.
    """
    val = mp.mpf(args[0])
    temp = mp.mpf(args[1])
    Ec_val = mp.mpf(args[2])
    D_val = mp.mpf(args[3])
    eps = mp.mpf(args[4])
    resistance = mp.mpf(args[5])  # Rl or Rr from exact diag

    mp.dps = dps
    threshold = mp.mpf('1e-20')

    abs_val = mp.fabs(val)
    sigma = mp.sqrt(2 * Ec_val * temp)
    bracket_width = 5 * sigma

    probability = mp.mpf('0')
    theta_max = mp.mpf('12.0')
    signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    if abs_val < D_val + eps:
        def mapped_integrand(theta1, theta2, sign_E, sign_Etag):
            E = sign_E * D_val * mp.cosh(theta1)
            Etag = sign_Etag * D_val * mp.cosh(theta2) + val

            gauss_arg = -((E - Etag - Ec_val) ** 2) / (4 * Ec_val * temp)
            if gauss_arg < -200:
                return mp.mpf('0')
            gauss = mp.exp(gauss_arg) / mp.sqrt(mp.pi * 4 * Ec_val * temp)

            f_E = f(E, temp)
            f_Etag_w = f(Etag - val, temp)

            measure = mp.fabs(E) * mp.fabs(Etag - val)
            return measure * f_E * (mp.mpf('1') - f_Etag_w) * gauss

        for s1, s2 in signs:
            func_quadrant = lambda t1, t2, s1=s1, s2=s2: mapped_integrand(t1, t2, s1, s2)
            res = mp.quad(func_quadrant, [0, theta_max], [0, theta_max], method='gauss-legendre')
            probability += res

        rate = probability.real / (mp.mpf(str(e ** 2)) * resistance)
        return float(rate)

    else:
        func = qp_integrand(temp, val, Ec_val, D_val)

        limits_E_far_neg = [-mp.inf, -abs_val, -D_val - eps]
        limits_E_far_pos = [D_val + eps, abs_val, mp.inf]

        def get_mapping(a, b):
            if a == -mp.inf:
                return lambda t: (b - t / (mp.mpf('1') - t), mp.mpf('1') / ((mp.mpf('1') - t) ** 2))
            elif b == mp.inf:
                return lambda t: (a + t / (mp.mpf('1') - t), mp.mpf('1') / ((mp.mpf('1') - t) ** 2))
            else:
                width = b - a
                if width < threshold:
                    return None
                return lambda t: (a + t * width, width)

        mappings_E = []
        for lims in [limits_E_far_neg, limits_E_far_pos]:
            for i in range(len(lims) - 1):
                m_func = get_mapping(lims[i], lims[i + 1])
                if m_func is not None:
                    mappings_E.append(m_func)

        for m_E in mappings_E:
            def outer_integrand(t_E, m_E=m_E):
                if t_E <= 0 or t_E >= 1:
                    return mp.mpf('0')

                E, jac_E = m_E(t_E)
                peak_center = E - Ec_val

                dynamic_limits = [
                    -mp.inf, mp.mpf(val - D_val), mp.mpf(val), mp.mpf(val + D_val),
                    peak_center - bracket_width, peak_center + bracket_width, mp.inf
                ]

                sorted_Etag_limits = sorted(list(set(dynamic_limits)))
                cleaned_limits = [sorted_Etag_limits[0]]
                for cp in sorted_Etag_limits[1:]:
                    if cp - cleaned_limits[-1] > threshold:
                        cleaned_limits.append(cp)

                inner_integral = mp.quad(lambda Etag: func(E, Etag), cleaned_limits, method='tanh-sinh')
                return inner_integral * jac_E

            segment_prob = mp.quad(outer_integrand, [0, 1], method='tanh-sinh')
            probability += segment_prob

        theta1_max = mp.acosh((D_val + eps) / D_val)

        for s1 in [1, -1]:
            def outer_integrand_near(theta1, s1=s1):
                E = s1 * D_val * mp.cosh(theta1)
                peak_center = E - Ec_val
                prob_inner = mp.mpf('0')

                for s2 in [1, -1]:
                    def inner_near_Etag(theta2, s2=s2):
                        Etag = val + s2 * D_val * mp.cosh(theta2)
                        gauss_arg = -((E - Etag - Ec_val) ** 2) / (4 * Ec_val * temp)
                        if gauss_arg < -200:
                            return mp.mpf('0')
                        gauss = mp.exp(gauss_arg) / mp.sqrt(mp.pi * 4 * Ec_val * temp)
                        f_E = f(E, temp)
                        f_Etag_w = f(Etag - val, temp)

                        measure = mp.fabs(E) * mp.fabs(Etag - val)
                        return measure * f_E * (mp.mpf('1') - f_Etag_w) * gauss

                    prob_inner += mp.quad(inner_near_Etag, [0, theta1_max], method='gauss-legendre')

                dynamic_limits_far = [
                    -mp.inf, val - D_val - eps, val + D_val + eps,
                             peak_center - bracket_width, peak_center + bracket_width, mp.inf
                ]

                sorted_Etag_far = sorted(list(set(dynamic_limits_far)))
                cleaned_far = [sorted_Etag_far[0]]
                for cp in sorted_Etag_far[1:]:
                    if cp - cleaned_far[-1] > threshold:
                        cleaned_far.append(cp)

                for i in range(len(cleaned_far) - 1):
                    a = cleaned_far[i]
                    b = cleaned_far[i + 1]
                    mid = (a + b) / mp.mpf('2.0')

                    if val - D_val - eps < mid < val + D_val + eps:
                        continue

                    def inner_far_Etag(Etag):
                        n_Etag = dos(Etag - val, D_val)
                        if n_Etag == mp.mpf('0'): return mp.mpf('0')
                        gauss_arg = -((E - Etag - Ec_val) ** 2) / (4 * Ec_val * temp)
                        if gauss_arg < -200: return mp.mpf('0')
                        gauss = mp.exp(gauss_arg) / mp.sqrt(mp.pi * 4 * Ec_val * temp)

                        measure_E = mp.fabs(E)
                        return measure_E * n_Etag * f(E, temp) * (mp.mpf('1') - f(Etag - val, temp)) * gauss

                    prob_inner += mp.quad(inner_far_Etag, [a, b], method='tanh-sinh')

                return prob_inner

            res = mp.quad(outer_integrand_near, [0, theta1_max], method='gauss-legendre')
            probability += res

        # Apply standard (1 / (e^2 R_T)) multiplier for standard quasiparticles
        rate = probability.real / (mp.mpf(str(e ** 2)) * resistance)
        return float(rate)


def Gamma(dE, T, resistance):
    """Wrapper to map exact diag Gamma calls to the new master integrator."""
    mp.dps = DPS
    args = [str(dE), str(T), str(Ec), str(D_gap), '1e-10', str(resistance)]
    return _calc_segments_gapped_master(args, DPS)


# ============================================================================
# EXACT DIAGONALIZATION ROUTINE
# ============================================================================

def calculate_current(Vl, N, T_l, T_r, Tdot):
    G_L_plus = np.zeros(N + 1)
    G_R_plus = np.zeros(N + 1)
    G_L_minus = np.zeros(N + 1)
    G_R_minus = np.zeros(N + 1)

    G_L_plus2 = np.zeros(N + 1)
    G_R_plus2 = np.zeros(N + 1)
    G_L_minus2 = np.zeros(N + 1)
    G_R_minus2 = np.zeros(N + 1)

    for n in range(N + 1):
        G_L_plus[n] = Gamma(W(n, Qn(Vl, n), Vl, 1, "left"), T_l, Rl)
        G_L_minus[n] = Gamma(W(n, Qn(Vl, n), Vl, -1, "left"), Tdot, Rl)
        G_R_plus[n] = Gamma(W(n, Qn(Vl, n), Vl, 1, "right"), T_r, Rr)
        G_R_minus[n] = Gamma(W(n, Qn(Vl, n), Vl, -1, "right"), Tdot, Rr)

        if n + 2 <= N:
            G_L_plus2[n] = Gamma_cp(W(n, Qn(Vl, n), Vl, 2, "left"), T_l, Ec, Rt=Rl)
            G_R_plus2[n] = Gamma_cp(W(n, Qn(Vl, n), Vl, 2, "right"), T_r, Ec, Rt=Rr)
        if n - 2 >= 0:
            G_L_minus2[n] = Gamma_cp(W(n, Qn(Vl, n), Vl, -2, "left"), Tdot, Ec, Rt=Rl)
            G_R_minus2[n] = Gamma_cp(W(n, Qn(Vl, n), Vl, -2, "right"), Tdot, Ec, Rt=Rr)

        gc.collect()

    G_plus = G_L_plus + G_R_plus
    G_minus = G_L_minus + G_R_minus
    G_plus2 = G_L_plus2 + G_R_plus2
    G_minus2 = G_L_minus2 + G_R_minus2

    G_plus[-1] = 0.0
    G_minus[0] = 0.0
    G_plus2[-1], G_plus2[-2] = 0.0, 0.0
    G_minus2[0], G_minus2[1] = 0.0, 0.0

    diag_main = -(G_plus + G_minus + G_plus2 + G_minus2)
    diag_sub = G_plus[:-1]
    diag_super = G_minus[1:]
    diag_sub2 = G_plus2[:-2]
    diag_super2 = G_minus2[2:]

    M = (np.diag(diag_main) +
         np.diag(diag_sub, k=-1) +
         np.diag(diag_super, k=1) +
         np.diag(diag_sub2, k=-2) +
         np.diag(diag_super2, k=2))

    eigenvalues, eigenvectors = eig(M)
    zero_idx = np.argmin(np.abs(eigenvalues))
    p_stat = np.real(eigenvectors[:, zero_idx])
    p_stat = p_stat / np.sum(p_stat)

    current_1e = e * np.sum(p_stat * (G_L_plus - G_L_minus))
    current_2e = 2 * e * np.sum(p_stat * (G_L_plus2 - G_L_minus2))

    return current_1e + current_2e


def worker_single_point(task_args):
    m, v_idx, V, N_states_task, T0, cp_dir = task_args
    T_left = T0
    T_dot = T0 + 10 * m * T0
    T_right = T0 + 20 * m * T0

    print(f"Worker computing grad {m}, Vl = {V:.3f} V...", flush=True)
    I = calculate_current(V, N=N_states_task, T_l=T_left, T_r=T_right, Tdot=T_dot)

    cp_path = Path(cp_dir) / f"grad_{m}_vidx_{v_idx}.npy"
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
    V_vals = np.linspace(0, 4, num_points)
    num_of_grads = 20

    all_tasks = []
    for m in range(num_of_grads):
        for v_idx, V in enumerate(V_vals):
            all_tasks.append((m, v_idx, V, N_states, T0, checkpoint_dir))

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
        pf.write(f"--- Exact Diagonalization Execution Log ---\n")
        pf.write(f"Chunk ID               : {chunk_id} (of {num_chunks} total chunks)\n")
        pf.write(f"Global Precision (DPS) : {DPS}\n")
        pf.write(f"ProcessPool Workers    : {num_workers}\n")
        pf.write(f"Total Tasks in Chunk   : {len(my_tasks)}\n")
        pf.write(f"N_states               : {N_states}\n")
        pf.write(f"T0                     : {T0}\n")
        pf.write(f"Cg                     : {Cg}\n")
        pf.write(f"Ec                     : {float(Ec):.5e}\n")
        pf.write(f"D_gap                  : {float(D_gap):.5e}\n")

    print(f"Slicing Task Group: processing tasks {start_idx} to {end_idx} (Total: {len(my_tasks)}) on Pool.",
          flush=True)

    if len(my_tasks) > 0:
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(worker_single_point, my_tasks)
        print("Slice complete.", flush=True)
    else:
        print("ERROR: Slice resulted in 0 tasks. Check your array geometry.", flush=True)