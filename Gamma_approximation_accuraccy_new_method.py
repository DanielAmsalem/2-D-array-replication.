import os

ratio = 1
os.environ["OPENBLAS_NUM_THREADS"] = str(ratio)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
num_workers = int(0.9 * total_cpus / ratio)
print(f"Worker number set to {num_workers} for {total_cpus} CPUs", flush=True)

import concurrent.futures
import mpmath
import numpy as np
from mpmath import mp, exp, sqrt
from pathlib import Path
import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Parameters ---
DPS = 50  # Global precision parameter
e = 1
Vr = 0
Cl = 2
Cr = 1
Rl = 1
Rr = 10
Rg = 1000 * (Cl + Cr)
Cg = 10 * (Cl + Cr)
Cs = Cg + Cl + Cr


def qp_integrand(T, dE, Ec, D):
    def conv(E, Etag):
        n_E = dos(E, D)
        if n_E == 0:
            return 0

        n_Etag = dos(Etag - dE, D)
        if n_Etag == 0:
            return 0

        gauss = exp(-((E - Etag - Ec) ** 2) / (4 * Ec * T))
        # FIXED: np.pi replaced with mp.pi to prevent precision truncation
        gauss = gauss / sqrt(mp.pi * 4 * Ec * T)

        return n_E * n_Etag * f(E, T) * (1 - f(Etag - dE, T)) * gauss

    return conv


def dos(E, D):
    if mpmath.fabs(E) <= D:
        return 0
    val = E * E - D * D
    if val <= 0:
        return 0
    return mpmath.fabs(E) / sqrt(val)


def f(x, t):
    """Fermi-Dirac distribution using mpmath exponents for precision preservation."""
    if x / t > 1e10:
        return exp(-x / t)
    if x / t < -1e10:
        return mp.mpf('1')
    expon = exp(x / t)
    return mp.mpf('1') / (mp.mpf('1') + expon)


def U(n, Qg, Vl):
    return (Qg + n * e + Cl * Vl) / (Cl + Cr)


def Qn(Vl, n):
    return -Cg * (Cl * Vl + n * e) / Cs


def W(n, Qg, Vl, in_out, left_right):
    if abs(in_out) != 1:
        raise ValueError
    if left_right == "left":
        return e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2 - e * Vl
    elif left_right == "right":
        return e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2
    else:
        raise ValueError("left_right must be either 'left' or 'right'")


def _calc_segments_gapped_master(args, dps):
    """
    Master top-level worker function to calculate segmented probabilities for quasiparticles in 2D.
    Dynamically routes integration paths to isolate and annihilate singularities near the gap,
    while utilizing fast, dynamic piece-wise integrations for the smooth far-field.
    """
    val, temp, Ec, D = args
    mp.dps = dps

    print(f"START calculating w = {val:.3f} [T={temp}]", flush=True)

    # Step 1: Define the small constant parameter
    eps = mp.mpf('0.05')

    # We use abs(val) to determine if the gaussian peak is near the gap
    abs_val = mp.fabs(val)

    sigma = mp.sqrt(2 * Ec * temp)
    bracket_width = 5 * sigma

    probability = mp.mpf('0')
    theta_max = 12.0
    signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    # ---------------------------------------------------------
    # CASE 1 (Step 2): Gaussian is near or inside the gap
    # ---------------------------------------------------------
    if abs_val < D + eps:
        def mapped_integrand(theta1, theta2, sign_E, sign_Etag):
            E = sign_E * D * mp.cosh(theta1)
            Etag = sign_Etag * D * mp.cosh(theta2) + val

            gauss_arg = -((E - Etag - Ec) ** 2) / (4 * Ec * temp)
            if gauss_arg < -200:
                return mp.mpf('0')
            gauss = mp.exp(gauss_arg) / mp.sqrt(mp.pi * 4 * Ec * temp)

            f_E = f(E, temp)
            f_Etag_w = f(Etag - val, temp)

            measure = mp.fabs(E) * mp.fabs(Etag - val)
            return measure * f_E * (mp.mpf('1') - f_Etag_w) * gauss

        for s1, s2 in signs:
            func_quadrant = lambda t1, t2, s1=s1, s2=s2: mapped_integrand(t1, t2, s1, s2)
            # Trivial constant limits; handled instantly with gauss-legendre
            res = mp.quad(func_quadrant, [0, theta_max], [0, theta_max], method='gauss-legendre')
            probability += res

        print(f"DONE  calculating w = {val:.3f} [T={temp}]", flush=True)
        return [val, float(probability.real), temp, Ec]

    # ---------------------------------------------------------
    # CASE 2 (Steps 3, 4, 5, 6): Gaussian is far from the gap
    # ---------------------------------------------------------
    else:
        func = qp_integrand(temp, val, Ec, D)

        # Step 4 & 5: The Outer Integral (Far-Field Regions)
        limits_E_far_neg = [-mp.inf, -abs_val, -D - eps]
        limits_E_far_pos = [D + eps, abs_val, mp.inf]

        def get_mapping(a, b):
            if a == -mp.inf:
                return lambda t: (b - t / (mp.mpf('1') - t), mp.mpf('1') / ((mp.mpf('1') - t) ** 2))
            elif b == mp.inf:
                return lambda t: (a + t / (mp.mpf('1') - t), mp.mpf('1') / ((mp.mpf('1') - t) ** 2))
            else:
                width = b - a
                if width < 1e-8:
                    return None
                return lambda t: (a + t * width, width)

        mappings_E = []
        for lims in [limits_E_far_neg, limits_E_far_pos]:
            for i in range(len(lims) - 1):
                m = get_mapping(lims[i], lims[i + 1])
                if m is not None:
                    mappings_E.append(m)

        for m_E in mappings_E:
            def outer_integrand(t_E, m_E=m_E):
                if t_E <= 0 or t_E >= 1:
                    return mp.mpf('0')

                E, jac_E = m_E(t_E)
                peak_center = E - Ec

                # Etag topological boundaries safely use D directly
                dynamic_limits = [
                    -mp.inf,
                    mp.mpf(val - D),
                    mp.mpf(val),
                    mp.mpf(val + D),
                    peak_center - bracket_width,
                    peak_center + bracket_width,
                    mp.inf
                ]

                sorted_Etag_limits = sorted(list(set(dynamic_limits)))

                # Filter microscopic 1e-8 segments purely in physical space
                cleaned_limits = [sorted_Etag_limits[0]]
                for cp in sorted_Etag_limits[1:]:
                    if cp - cleaned_limits[-1] > 1e-8:
                        cleaned_limits.append(cp)

                inner_integral = mp.quad(lambda Etag: func(E, Etag), cleaned_limits, method='tanh-sinh', maxdegree=7)

                return inner_integral * jac_E

            segment_prob = mp.quad(outer_integrand, [0, 1], method='tanh-sinh', maxdegree=7)
            probability += segment_prob

        # Step 6: The Inner Integrals (Near the Gap)
        # Calculates E strictly in [-D-eps, -D] and [D, D+eps] using singularity-free transforms
        def mapped_integrand_near(theta1, theta2, sign_E, sign_Etag):
            E = sign_E * D * mp.cosh(theta1)
            Etag = sign_Etag * D * mp.cosh(theta2) + val

            gauss_arg = -((E - Etag - Ec) ** 2) / (4 * Ec * temp)
            if gauss_arg < -200:
                return mp.mpf('0')
            gauss = mp.exp(gauss_arg) / mp.sqrt(mp.pi * 4 * Ec * temp)

            f_E = f(E, temp)
            f_Etag_w = f(Etag - val, temp)

            measure = mp.fabs(E) * mp.fabs(Etag - val)
            return measure * f_E * (mp.mpf('1') - f_Etag_w) * gauss

        # D * cosh(theta1_max) = D + eps
        theta1_max = mp.acosh((D + eps) / D)

        for s1, s2 in signs:
            func_quadrant = lambda t1, t2, s1=s1, s2=s2: mapped_integrand_near(t1, t2, s1, s2)

            # Since theta1_max and theta_max are constants, caching is perfectly safe.
            # No 't' transformations needed. No Gaussian peaks missed.
            res = mp.quad(func_quadrant, [0, theta1_max], [0, theta_max], method='gauss-legendre')
            probability += res

        print(f"DONE  calculating w = {val:.3f} [T={temp}]", flush=True)
        return [val, float(probability.real), temp, Ec]


# --- Parallel Worker Function ---
def compute_gamma_worker(w, T, Rt, mu=0.5 / Cg):
    """Top-level wrapper to ensure it can be pickled by multiprocessing."""
    D = 2 * mu
    args = (w, T, mu, D)
    return float(_calc_segments_gapped_master(args, DPS)[1])


# --- Execution Block ---
if __name__ == '__main__':
    DIR = Path(__file__).parent
    w_values = np.linspace(-1, 1, 40, endpoint=False)
    T_values = [0.001, 0.01, 0.1]

    # 1. Flatten the parameter space so ALL jobs can be queued at once
    w_args = []
    T_args = []
    Rt_args = []

    for T in T_values:
        w_args.extend(w_values)
        T_args.extend([T] * len(w_values))
        Rt_args.extend([1] * len(w_values))

    total_tasks = len(w_args)
    print(f"Submitting all {total_tasks} tasks simultaneously...", flush=True)

    # 2. Execute everything in one massive pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # map preserves the exact order of the flat list we submitted
        all_gamma_results = list(executor.map(compute_gamma_worker, w_args, T_args, Rt_args))

    print("Calculations complete. Generating plot...", flush=True)

    # 3. Re-split the results array and plot
    plt.figure(figsize=(8, 5))
    chunk_size = len(w_values)

    for i, T in enumerate(T_values):
        # Slice out the 40 results belonging to the current T
        gamma_chunk = all_gamma_results[i * chunk_size: (i + 1) * chunk_size]
        plt.plot(w_values, gamma_chunk, linewidth=2, label=f'T = {T}')

    plt.title(r"Gamma vs w, $\Delta = 0.2*E_c$")
    plt.xlabel("w")
    plt.ylabel("Gamma")

    # plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Save output to disk
    # FIXED: Date format removes colons, and DPS parameter is dynamically inserted into the filename
    filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_mp_dps{DPS}.png"
    output_file = DIR / filename

    # FIXED: dpi reduced to 600 to prevent Matplotlib MemoryError
    plt.savefig(fname=output_file, dpi=600, bbox_inches="tight")
    print(f"Plot saved successfully to {output_file}", flush=True)