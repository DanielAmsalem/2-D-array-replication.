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
Rg = 1000
Cg = 10 * (Cl + Cr)
Cs = Cg + Cl + Cr


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
    # Parse strictly into mpmath floats to guarantee precision from the start
    val = mp.mpf(args[0])
    temp = mp.mpf(args[1])
    Ec = mp.mpf(args[2])
    D = mp.mpf(args[3])
    eps = mp.mpf(args[4])

    mp.dps = dps

    # Threshold logic: strictly 1e-10 to balance precision and runtime stability
    threshold = mp.mpf('1e-20')

    print(f"START calculating w = {float(val):.3f} [T={float(temp)}, eps={float(eps)}]", flush=True)

    abs_val = mp.fabs(val)
    sigma = mp.sqrt(2 * Ec * temp)
    bracket_width = 5 * sigma

    probability = mp.mpf('0')
    theta_max = mp.mpf('12.0')
    signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    # ---------------------------------------------------------
    # CASE 1: Gaussian is near or inside the gap
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
            res = mp.quad(func_quadrant, [0, theta_max], [0, theta_max], method='gauss-legendre')
            probability += res

        print(f"DONE  calculating w = {float(val):.3f} [T={float(temp)}, eps={float(eps)}]", flush=True)
        return [args[0], str(probability.real), args[1], args[2]]

    # ---------------------------------------------------------
    # CASE 2: Gaussian is far from the gap
    # ---------------------------------------------------------
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

        # Step 6: Inner Integrals (Near the Gap)
        # Evaluates E in [-D-eps, -D] and [D, D+eps]. E' is split into near-gap and far-field.
        theta1_max = mp.acosh((D + eps) / D)

        for s1 in [1, -1]:
            def outer_integrand_near(theta1, s1=s1):
                E = s1 * D * mp.cosh(theta1)
                peak_center = E - Ec
                prob_inner = mp.mpf('0')

                # Part A: Etag near its gap [-D-eps+w, -D+w] U [D+w, D+eps+w]
                # Safely evaluated with isolated cosh transform
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

                # Part B: Etag far from its gap (-inf, -D-eps+w] U [D+eps+w, inf)
                # Evaluated via tanh-sinh and dynamic arrays
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

                    # Skip the region already handled by Part A (and the absolute gap itself)
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

                        # Outer substitution scale applies, inner is natively evaluated
                        measure_E = mp.fabs(E)
                        return measure_E * n_Etag * f_E * (mp.mpf('1') - f_Etag_w) * gauss

                    prob_inner += mp.quad(inner_far_Etag, [a, b], method='tanh-sinh')

                return prob_inner

            res = mp.quad(outer_integrand_near, [0, theta1_max], method='gauss-legendre')
            probability += res

        print(f"DONE  calculating w = {float(val):.3f} [T={float(temp)}, eps={float(eps)}]", flush=True)
        return [args[0], str(probability.real), args[1], args[2]]


# --- Parallel Worker Function ---
def compute_gamma_worker(w_str, T_str, Rt, g_ratio_str, mu_str, eps_str):
    """
    Top-level wrapper to ensure pickling and 50dps precision maintenance.
    Receives strings, performs setup in mpmath, calls master function, returns string result.
    """
    mp.dps = DPS
    D_mp = mp.mpf(g_ratio_str) * mp.mpf(mu_str)
    args = (w_str, T_str, mu_str, str(D_mp), eps_str)
    return _calc_segments_gapped_master(args, DPS)[1]


# --- Execution Block ---
if __name__ == '__main__':
    DIR = Path(__file__).parent

    mp.dps = DPS
    w_start = mp.mpf('-3')
    w_end = mp.mpf('-0.1')
    num_points = 40

    # Exact 50 dps w values cast strictly to string
    w_values_str = [str(w_start + mp.mpf(i) * (w_end - w_start) / mp.mpf(num_points - 1)) for i in range(num_points)]
    w_values_float = [float(w) for w in w_values_str]

    T_values_str = ['0.001', '0.01', '0.1']
    gap_ratios_str = ['2']
    EPS_values_str = ['1e-10']

    mu_mp = mp.mpf('0.5') / mp.mpf(str(Cg))
    mu_str = str(mu_mp)

    # Wrap the entire calculation and plotting process in the eps loop
    for eps_str in EPS_values_str:
        print(f"\n==============================================")
        print(f"Starting calculations for EPS = {eps_str}")
        print(f"==============================================", flush=True)

        w_args = []
        T_args = []
        Rt_args = []
        GAP_args = []
        EPS_args = []

        for g_ratio_str in gap_ratios_str:
            for T_str in T_values_str:
                w_args.extend(w_values_str)
                T_args.extend([T_str] * len(w_values_str))
                Rt_args.extend([1] * len(w_values_str))
                GAP_args.extend([g_ratio_str] * len(w_values_str))
                EPS_args.extend([eps_str] * len(w_values_str))

        total_tasks = len(w_args)
        print(f"Submitting all {total_tasks} tasks simultaneously...", flush=True)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            all_gamma_results_str = list(executor.map(
                compute_gamma_worker,
                w_args, T_args, Rt_args, GAP_args, [mu_str] * total_tasks, EPS_args
            ))

        print("Calculations complete. Evaluating analytical errors in mpmath precision...", flush=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        chunk_size = len(w_values_float)

        colors = {0.001: '#1f77b4', 0.01: '#ff7f0e', 0.1: '#2ca02c'}

        idx = 0
        for g_ratio_str in gap_ratios_str:
            D_mp = mp.mpf(g_ratio_str) * mu_mp

            for T_str in T_values_str:
                gamma_chunk_str = all_gamma_results_str[idx: idx + chunk_size]
                idx += chunk_size

                T_float = float(T_str)
                c = colors[T_float]
                lbl_main = f"T={T_float}, gap={g_ratio_str}"

                gamma_plot = []
                ln_err_y1 = []
                ln_err_y2 = []
                ln_err_y3 = []

                for w_str, g_str in zip(w_values_str, gamma_chunk_str):
                    w_mp = mp.mpf(w_str)
                    g_mp = mp.mpf(g_str)

                    gamma_plot.append(float(g_mp))

                    x_mp = D_mp / w_mp

                    # 4th, 20th, and 40th order alternating series sums
                    y1 = (-mu_mp - w_mp) * sum((-x_mp ** 2) ** k for k in range(3))  # 4th order
                    y2 = (-mu_mp - w_mp) * sum((-x_mp ** 2) ** k for k in range(11))  # 20th order
                    y3 = (-mu_mp - w_mp) * sum((-x_mp ** 2) ** k for k in range(21))  # 40th order

                    diff1 = mp.fabs(g_mp - y1)
                    diff2 = mp.fabs(g_mp - y2)
                    diff3 = mp.fabs(g_mp - y3)

                    err1 = float(mp.log(diff1)) if diff1 > 0 else np.nan
                    err2 = float(mp.log(diff2)) if diff2 > 0 else np.nan
                    err3 = float(mp.log(diff3)) if diff3 > 0 else np.nan

                    ln_err_y1.append(err1)
                    ln_err_y2.append(err2)
                    ln_err_y3.append(err3)

                ax1.plot(w_values_float, gamma_plot, color=c, linestyle='-', linewidth=2, label=lbl_main)

                ax2.plot(w_values_float, ln_err_y1, color=c, linestyle='-', linewidth=2, label=f"T={T_float}, $O(x^4)$")
                ax2.plot(w_values_float, ln_err_y2, color=c, linestyle='--', linewidth=2,
                         label=f"T={T_float}, $O(x^{{20}})$")
                ax2.plot(w_values_float, ln_err_y3, color=c, linestyle=':', linewidth=2.5,
                         label=f"T={T_float}, $O(x^{{40}})$")

        ax1.set_title(fr"$\Gamma(w)$ vs $w$, fixed $\epsilon = {eps_str}$")
        ax1.set_xlabel("w")
        ax1.set_ylabel(r"$\Gamma$")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')

        ax2.set_title(r"Log Error: $\ln(|\Gamma(w) - y_n|)$ evaluated at $50$ dps")
        ax2.set_xlabel("w")
        ax2.set_ylabel(r"$\ln(\text{Error})$")
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

        plt.tight_layout()

        filename = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S") + f"_mp_dps{DPS}_eps{eps_str}_multi_gap_analysis.png"
        output_file = DIR / filename

        plt.savefig(fname=output_file, dpi=600, bbox_inches="tight")
        print(f"Plot saved successfully to {output_file}", flush=True)

        # Critically important to close the figure before the next loop
        plt.close(fig)