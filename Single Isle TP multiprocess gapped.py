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
import functools
from scipy.integrate import quad
from scipy.optimize import fsolve

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

# Superconducting Gap Parameters
D_ratio = 0.2
Delta_0_float = D_ratio * float(Ec)


# ============================================================================
# EXACT BCS GAP SOLVER
# ============================================================================

@functools.lru_cache(maxsize=None)
def get_gap_at_T(T, Delta_0):
    """
    Numerically solves the BCS self-consistency equation for a single Temperature.
    Memoized with lru_cache to prevent redundant integral solving.
    """
    Tc = Delta_0 / 1.764

    if T >= Tc:
        return 0.0
    if T < 0.01 * Tc:
        return Delta_0

    def bcs_integral(Delta, T_val):
        if Delta <= 0:
            return 1e9
        integrand = lambda E: 2.0 / (np.exp(np.sqrt(E ** 2 + Delta ** 2) / T_val) + 1.0) / np.sqrt(E ** 2 + Delta ** 2)
        with np.errstate(over='ignore'):
            val, _ = quad(integrand, 0, 100 * Delta_0)
        return val - np.log(Delta_0 / Delta)

    # Solve the integral. We use Delta_0 as the safe initial guess
    sol = fsolve(bcs_integral, Delta_0, args=(T,))[0]
    return float(sol)


# ============================================================================
# ENERGY TRANSITION HELPERS (RESTORED LEGACY LOGIC)
# ============================================================================

def U(n, Qg, Vl):
    """Calculates the instantaneous electrostatic potential of the island."""
    return (Qg + n * e + Cl * Vl) / (Cl + Cr)


def Qn(Vl, n):
    """Calculates the steady-state induced gate charge Qg."""
    return -Cg * (Cl * Vl + n * e) / Cs


def W(n, Qg, Vl, in_out, left_right):
    """
    Calculates the exact electrostatic energy difference (dE) for a tunneling event.
    """
    if abs(in_out) not in [1, 2]:
        raise ValueError("in_out must be 1 (quasiparticle) or 2 (Cooper pair)")

    if left_right == "left":
        dE = in_out * e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2 - in_out * e * Vl
    elif left_right == "right":
        # Included Vr here for physical completeness, even though Vr = 0
        dE = in_out * e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2 - in_out * e * Vr
    else:
        raise ValueError("left_right must be either 'left' or 'right'")

    return mp.mpf(str(dE))


# ============================================================================
# MASTER INTEGRATION ALGORITHMS
# ============================================================================

def Gamma_cp(dE, T, Ec_val, Rt, D_local):
    """Cooper pair transition rate using the local Temperature-dependent gap."""
    if D_local <= 1e-8:
        return 0.0  # Above Tc, Cooper pairs cease to exist

    # Convert inputs strictly to 50-dps mpmath objects
    dE_mp = mp.mpf(str(dE))
    T_mp = mp.mpf(str(T))
    Rt_mp = mp.mpf(str(Rt))
    Ec_mp = mp.mpf(str(Ec_val))
    D_mp = mp.mpf(str(D_local))

    # Ej = tanh * gap / 8Rt
    tanh_val = mp.tanh(D_mp / (mp.mpf('2') * T_mp))
    Ej = tanh_val * D_mp / (mp.mpf('8') * Rt_mp)

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


# ============================================================================
# MEMOIZATION CACHE
# ============================================================================
@functools.lru_cache(maxsize=None)
def _cached_gapped_master_call(dE_str, T_str, Ec_str, D_gap_str, eps_str, resistance_str):
    """Hidden cached caller to prevent redundant 50-DPS integrations."""
    args = [dE_str, T_str, Ec_str, D_gap_str, eps_str, resistance_str]
    return float(_calc_segments_gapped_master(args, DPS))


def Gamma(dE, T, resistance, D_local):
    """Wrapper mapping to Quasiparticle integrator with explicit D_local passed."""
    if D_local <= 1e-8:
        # If D is strictly 0 (above Tc), standard Normal Metal integrators should ideally be used.
        # But setting D to a microscopic non-zero prevents divide-by-zero if passed to SC DOS logic.
        D_local = 1e-12

    mp.dps = DPS
    return _cached_gapped_master_call(str(dE), str(T), str(Ec), str(D_local), '1e-10', str(resistance))


# ============================================================================
# EXACT DIAGONALIZATION ROUTINE
# ============================================================================

def calculate_current(Vl, N, T_l, T_r, Tdot):
    # Determine precise gaps at local temperatures
    D_l = get_gap_at_T(T_l, Delta_0_float)
    D_r = get_gap_at_T(T_r, Delta_0_float)
    D_dot = get_gap_at_T(Tdot, Delta_0_float)

    G_L_plus = np.zeros(N + 1)
    G_R_plus = np.zeros(N + 1)
    G_L_minus = np.zeros(N + 1)
    G_R_minus = np.zeros(N + 1)

    G_L_plus2 = np.zeros(N + 1)
    G_R_plus2 = np.zeros(N + 1)
    G_L_minus2 = np.zeros(N + 1)
    G_R_minus2 = np.zeros(N + 1)

    for n in range(N + 1):
        # Calculate the steady state gate charge for state n just once per loop
        Qg_current = Qn(Vl, n)

        G_L_plus[n] = Gamma(W(n, Qg_current, Vl, 1, "left"), T_l, Rl, D_l)
        G_L_minus[n] = Gamma(W(n, Qg_current, Vl, -1, "left"), Tdot, Rl, D_dot)
        G_R_plus[n] = Gamma(W(n, Qg_current, Vl, 1, "right"), T_r, Rr, D_r)
        G_R_minus[n] = Gamma(W(n, Qg_current, Vl, -1, "right"), Tdot, Rr, D_dot)

        if n + 2 <= N:
            G_L_plus2[n] = Gamma_cp(W(n, Qg_current, Vl, 2, "left"), T_l, Ec, Rl, D_l)
            G_R_plus2[n] = Gamma_cp(W(n, Qg_current, Vl, 2, "right"), T_r, Ec, Rr, D_r)
        if n - 2 >= 0:
            G_L_minus2[n] = Gamma_cp(W(n, Qg_current, Vl, -2, "left"), Tdot, Ec, Rl, D_dot)
            G_R_minus2[n] = Gamma_cp(W(n, Qg_current, Vl, -2, "right"), Tdot, Ec, Rr, D_dot)

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
    # Unpack the new flip parameter
    m, v_idx, V, N_states_task, T0, cp_dir, flip = task_args

    # --- RESILIENCE CHECK ---
    # Construct path and check if calculation is already done
    cp_path = Path(cp_dir) / f"grad_{m}_vidx_{v_idx}_flip_{flip}.npy"
    if cp_path.exists():
        print(f"Worker SKIPPING grad {m}, Vl = {V:.3f} V, flip = {flip} (Already computed).", flush=True)
        return True

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

    np.save(cp_path, I)

    print(f"Worker DONE computing grad {m}, Vl = {V:.3f} V | I = {I:.5e}", flush=True)
    return True


# ============================================================================
# MAIN CLUSTER SLICING ROUTINE
# ============================================================================

if __name__ == '__main__':
    base_folder = Path(__file__).parent.absolute()

    # Dynamic D Checkpoint mapping
    checkpoint_dir = base_folder / f"checkpoints_D{D_ratio}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Simulation Constraints
    T0 = 0.001
    num_points = 101
    V_vals = np.linspace(0, 2, num_points)
    num_of_grads = 20
    flip_states = [False, True]

    all_tasks = []
    for m in range(num_of_grads):
        for flip in flip_states:
            for v_idx, V in enumerate(V_vals):
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
    param_filename = base_folder / f"parameters_D{D_ratio}_DPS{DPS}_chunk{chunk_id}_{timestamp_str}.txt"
    with open(param_filename, mode='w', encoding='utf-8') as pf:
        pf.write(f"--- Superconducting Exact Diagonalization Execution Log ---\n")
        pf.write(f"Chunk ID               : {chunk_id} (of {num_chunks} total chunks)\n")
        pf.write(f"Global Precision (DPS) : {DPS}\n")
        pf.write(f"ProcessPool Workers    : {num_workers}\n")
        pf.write(f"Total Tasks in Chunk   : {len(my_tasks)}\n")
        pf.write(f"N_states               : {N_states}\n")
        pf.write(f"T0                     : {T0}\n")
        pf.write(f"Cg                     : {Cg}\n")
        pf.write(f"Ec                     : {float(Ec):.5e}\n")
        pf.write(f"D_ratio                : {D_ratio}\n")
        pf.write(f"Delta_0                : {Delta_0_float:.5e}\n")

    print(f"Slicing Task Group: processing tasks {start_idx} to {end_idx} (Total: {len(my_tasks)}) on Pool.",
          flush=True)

    if len(my_tasks) > 0:
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(worker_single_point, my_tasks)
        print("Slice complete.", flush=True)
    else:
        print("ERROR: Slice resulted in 0 tasks. Check your array geometry.", flush=True)