import os

ratio = 1
os.environ["OPENBLAS_NUM_THREADS"] = str(ratio)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
num_workers = max(1, int(0.9 * total_cpus / ratio))
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
e = 1
Vr = 0
Cl = 2
Cr = 1
Rl = 1
Rr = 10
Rg = 1000 * (Cl + Cr)
Cg = 5 * (Cl + Cr)
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


def Gamma(w, T, Rt, mu=0.5 / Cg):
    mp.dps = 40
    func = integrand(T, w, mu)
    absval = abs(w)

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
    if abs(in_out) != 1:
        raise ValueError
    if left_right == "left":
        return e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2 - e * Vl
    elif left_right == "right":
        return e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2
    else:
        raise ValueError("left_right must be either 'left' or 'right'")


# --- Parallel Worker Function ---
def compute_gamma_worker(w, T, Rt):
    """Top-level wrapper to ensure it can be pickled by multiprocessing."""
    return float(Gamma(w, T, Rt))


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

    plt.title(r"Gamma vs w, $\Delta = 0$")
    plt.xlabel("w")
    plt.ylabel("Gamma")

    # plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Save output to disk
    filename = str(datetime.datetime.now()) + "_mp_dps40.png"
    output_file = DIR / filename
    plt.savefig(fname=output_file, dpi=2100, bbox_inches="tight")
    print(f"Plot saved successfully to {output_file}", flush=True)