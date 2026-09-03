import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
num_workers = max(total_cpus - 5, 1)
print(f"Worker number set to {num_workers} for {total_cpus} CPUs", flush=True)

import csv
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
DPS = 40  # Global precision parameter
Cg = 10


def qp_integrand(T, dE, Ec, D):
    def conv(E, Etag):
        n_E = dos(E, D)
        if n_E == mp.mpf('0'):
            return mp.mpf('0')

        n_Etag = dos(Etag - dE, D)
        if n_Etag == mp.mpf('0'):
            return mp.mpf('0')

        gauss = exp(-((E - Etag - Ec) ** 2) / (4 * Ec * T))
        gauss = gauss / sqrt(mp.pi * 4 * Ec * T)

        return n_E * n_Etag * f(E, T) * (mp.mpf('1') - f(Etag - dE, T)) * gauss

    return conv


def dos(E, D):
    if mpmath.fabs(E) <= D:
        return mp.mpf('0')
    val = E * E - D * D
    if val <= 0:
        return mp.mpf('0')
    return mpmath.fabs(E) / sqrt(val)


def f(x, t):
    """Fermi-Dirac distribution using mpmath exponents for precision preservation."""
    if x / t > mp.mpf('1e50'):
        return exp(-x / t)
    if x / t < mp.mpf('-1e50'):
        return mp.mpf('1')
    expon = exp(x / t)
    return mp.mpf('1') / (mp.mpf('1') + expon)


def _calc_segments_gapped_master(args, dps):
    """
    Master top-level worker function to calculate segmented probabilities for quasiparticles in 2D.
    """
    val = mp.mpf(args[0])
    temp = mp.mpf(args[1])
    Ec = mp.mpf(args[2])
    D = mp.mpf(args[3])
    eps = mp.mpf(args[4])

    mp.dps = dps
    threshold = mp.mpf('1e-20')

    print(f"START calculating w = {float(val):.3f} [T={float(temp)}, eps={float(eps)}]", flush=True)

    abs_val = mp.fabs(val)
    sigma = mp.sqrt(2 * Ec * temp)
    bracket_width = 5 * sigma

    probability = mp.mpf('0')
    theta_max = mp.mpf('12.0')
    signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

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
            res = mp.quad(func_quadrant, [0, theta_max], [0, theta_max], method='gauss-legendre')
            probability += res

        print(f"DONE  calculating w = {float(val):.3f} [T={float(temp)}, eps={float(eps)}]", flush=True)
        return [args[0], str(probability.real), args[1], args[2]]

    else:
        func = qp_integrand(temp, val, Ec, D)

        limits_E_far_neg = [-mp.inf, -abs_val, -D - eps]
        limits_E_far_pos = [D + eps, abs_val, mp.inf]

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
                m = get_mapping(lims[i], lims[i + 1])
                if m is not None:
                    mappings_E.append(m)

        for m_E in mappings_E:
            def outer_integrand(t_E, m_E=m_E):
                if t_E <= 0 or t_E >= 1:
                    return mp.mpf('0')

                E, jac_E = m_E(t_E)
                peak_center = E - Ec

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
                cleaned_limits = [sorted_Etag_limits[0]]
                for cp in sorted_Etag_limits[1:]:
                    if cp - cleaned_limits[-1] > threshold:
                        cleaned_limits.append(cp)

                inner_integral = mp.quad(lambda Etag: func(E, Etag), cleaned_limits, method='tanh-sinh')
                return inner_integral * jac_E

            segment_prob = mp.quad(outer_integrand, [0, 1], method='tanh-sinh')
            probability += segment_prob

        theta1_max = mp.acosh((D + eps) / D)

        for s1 in [1, -1]:
            def outer_integrand_near(theta1, s1=s1):
                E = s1 * D * mp.cosh(theta1)
                peak_center = E - Ec
                prob_inner = mp.mpf('0')

                for s2 in [1, -1]:
                    def inner_near_Etag(theta2, s2=s2):
                        Etag = val + s2 * D * mp.cosh(theta2)
                        gauss_arg = -((E - Etag - Ec) ** 2) / (4 * Ec * temp)
                        if gauss_arg < -200:
                            return mp.mpf('0')
                        gauss = mp.exp(gauss_arg) / mp.sqrt(mp.pi * 4 * Ec * temp)
                        f_E = f(E, temp)
                        f_Etag_w = f(Etag - val, temp)

                        measure = mp.fabs(E) * mp.fabs(Etag - val)
                        return measure * f_E * (mp.mpf('1') - f_Etag_w) * gauss

                    prob_inner += mp.quad(inner_near_Etag, [0, theta1_max], method='gauss-legendre')

                dynamic_limits_far = [
                    -mp.inf,
                    val - D - eps,
                    val + D + eps,
                    peak_center - bracket_width,
                    peak_center + bracket_width,
                    mp.inf
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

                    if val - D - eps < mid < val + D + eps:
                        continue

                    def inner_far_Etag(Etag):
                        n_Etag = dos(Etag - val, D)
                        if n_Etag == mp.mpf('0'):
                            return mp.mpf('0')
                        gauss_arg = -((E - Etag - Ec) ** 2) / (4 * Ec * temp)
                        if gauss_arg < -200:
                            return mp.mpf('0')
                        gauss = mp.exp(gauss_arg) / mp.sqrt(mp.pi * 4 * Ec * temp)
                        f_E = f(E, temp)
                        f_Etag_w = f(Etag - val, temp)

                        measure_E = mp.fabs(E)
                        return measure_E * n_Etag * f_E * (mp.mpf('1') - f_Etag_w) * gauss

                    prob_inner += mp.quad(inner_far_Etag, [a, b], method='tanh-sinh')

                return prob_inner

            res = mp.quad(outer_integrand_near, [0, theta1_max], method='gauss-legendre')
            probability += res

        print(f"DONE  calculating w = {float(val):.3f} [T={float(temp)}, eps={float(eps)}]", flush=True)
        return [args[0], str(probability.real), args[1], args[2]]


def compute_gamma_worker(w_str, T_str, gap_ratio_str, mu_str, eps_str):
    mp.dps = DPS
    D_mp = mp.mpf(gap_ratio_str) * mp.mpf(mu_str)
    args = (w_str, T_str, mu_str, str(D_mp), eps_str)
    return _calc_segments_gapped_master(args, DPS)[1]


# --- Execution Block ---
if __name__ == '__main__':
    DIR = Path(__file__).parent

    mp.dps = DPS
    w_start = mp.mpf('2')
    w_end = mp.mpf('0')
    num_points = 40

    w_values_str = [str(w_start + mp.mpf(i) * (w_end - w_start) / mp.mpf(num_points - 1)) for i in range(num_points)]
    w_values_float = [float(w) for w in w_values_str]

    gap_ratio_str = '2.0'
    eps_str = '1e-10'

    mu_mp = mp.mpf('0.5') / mp.mpf(str(Cg))
    mu_str = str(mu_mp)
    D_mp = mp.mpf(gap_ratio_str) * mu_mp

    # Temperature iteration sequence
    n_values = [0, 1, 3, 5, 7, 9, 11, 15, 19, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96, 112, 128, 144, 160, 176,
                192]

    # Setup CSV tracking files
    summary_filename = DIR / f"pos_Wc_thresholds_DPS{DPS}_Cg{Cg}_D2_0.csv"
    detailed_filename = DIR / f"pos_Detailed_Data_DPS{DPS}_Cg{Cg}_D2_0.csv"

    print(f"\n==============================================")
    print(f"PHASE 1: Building Master Task List for ALL temperatures...")

    # Pre-allocate master lists for the mega-pool
    w_args_all = []
    T_args_all = []
    GAP_args_all = []
    MU_args_all = []
    EPS_args_all = []

    # Flatten all temperatures and energy points into a 1D queue
    for n in n_values:
        T_mp = mp.mpf('0.001') + mp.mpf('0.006') * mp.mpf(n) / mp.mpf('20')
        T_str = str(T_mp)

        w_args_all.extend(w_values_str)
        T_args_all.extend([T_str] * num_points)
        GAP_args_all.extend([gap_ratio_str] * num_points)
        MU_args_all.extend([mu_str] * num_points)
        EPS_args_all.extend([eps_str] * num_points)

    total_tasks = len(w_args_all)
    print(f"Total tasks assembled: {total_tasks}")
    print(f"==============================================", flush=True)

    print(f"\nPHASE 2: Saturating {num_workers} CPUs with multiprocessing pool...", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        all_gamma_results_str = list(executor.map(
            compute_gamma_worker,
            w_args_all, T_args_all, GAP_args_all, MU_args_all, EPS_args_all
        ))

    print("\nPHASE 3: Parallel Execution Complete. Extracting Analytical Data & Plotting...", flush=True)

    with open(summary_filename, mode='w', newline='') as summary_file, \
            open(detailed_filename, mode='w', newline='') as detailed_file:

        # Setup writers
        sum_writer = csv.writer(summary_file)
        sum_writer.writerow(['n', 'T', 'Wc_n'])

        det_writer = csv.writer(detailed_file)
        det_writer.writerow(
            ['n', 'T', 'w', 'Gamma_numeric', 'Err'])

        # Iterate through the flattened results, chunking by num_points
        for idx, n in enumerate(n_values):
            T_mp = mp.mpf('0.001') + mp.mpf('0.006') * mp.mpf(n) / mp.mpf('20')
            T_str = str(T_mp)
            T_float = float(T_mp)

            print(f"--> Processing Analytical Evaluation for n = {n} (T = {T_float:.5f})", flush=True)

            # Slice the master results list for the current temperature chunk
            chunk_start = idx * num_points
            chunk_end = chunk_start + num_points
            gamma_chunk_str = all_gamma_results_str[chunk_start:chunk_end]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            gamma_plot = []
            y_approx = []
            diffs = []

            for w_str, g_str in zip(w_values_str, gamma_chunk_str):
                w_mp = mp.mpf(w_str)
                g_mp = mp.mpf(g_str)
                gamma_plot.append(float(g_mp))

                diff3 = mp.fabs(g_mp - 0)
                diffs.append(diff3)

                err3 = float(mp.log(diff3)) if diff3 > 0 else np.nan
                y_approx.append(err3)

                # --- Write all exact 50-digit strings to detailed CSV ---
                det_writer.writerow([
                    n, T_str, w_str, g_str, str(diff3)
                ])

            # --- Extract Wc_n Boundary ---
            target_error = mp.mpf('1e-9')
            Wc_n = "N/A"
            for w_val, diff in zip(reversed(w_values_float), reversed(diffs)):
                if diff < target_error:
                    Wc_n = w_val
                    break

            # Write summary CSV
            sum_writer.writerow([n, T_float, Wc_n])
            summary_file.flush()

            # Plotting routine
            lbl_main = f"T={T_float:.5f} (n={n})"
            ax1.plot(w_values_float, gamma_plot, color='#1f77b4', linestyle='-', linewidth=2, label=lbl_main)

            ax2.plot(w_values_float, y_approx, color='#ff7f0e', linestyle='-', linewidth=2,
                     label="Elliptic + Gaussian")

            ax2.axhline(y=np.log(1e-9), color='red', linestyle=':', label='1e-9 Threshold')
            if Wc_n != "N/A":
                ax2.axvline(x=Wc_n, color='green', linestyle='--', label=f'Wc = {Wc_n:.2f}')

            ax1.set_title(fr"$\Gamma(w)$ vs $w$ (n={n}, fixed $\epsilon = 10^{{-10}}$)")
            ax1.set_xlabel("w")
            ax1.set_ylabel(r"$\Gamma$")
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(loc='upper right')

            ax2.set_title(r"Log error at very positive $w$")
            ax2.set_xlabel("w")
            ax2.set_ylabel(r"$\ln(\text{Error})$")
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(loc='upper left', fontsize='small')

            plt.tight_layout()

            filename = DIR / f"pos_graph_n{n}_DPS{DPS}_eps{eps_str}_Cg{Cg}_D2_0.png"
            plt.savefig(fname=filename, dpi=600, bbox_inches="tight")
            plt.close(fig)

    print(
        f"\nAll iterations complete! Summary saved to {summary_filename.name}, raw traces saved to {detailed_filename.name}",
        flush=True)