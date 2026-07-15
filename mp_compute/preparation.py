import warnings
from pathlib import Path

from dataclasses import replace
import numpy as np
import numpy.typing as npt
from mpmath import quad, mp

import Functions as F
from define_objects import ExperimentInitialState

import concurrent.futures
import multiprocessing
import os
import json
import tempfile
import time

def compute_distributed_R_matrices(
        stdR: float,
        R: float,
        array_size: int,
        near_left: list,
        near_right: list
) -> tuple[list, npt.NDArray]:
    R_t_ij = 2 ** np.random.uniform(
        low=np.log2(max(R - stdR, 0.01)),
        high=np.log2(R + stdR),
        size=(array_size, array_size), )

    # make symmetric
    R_t_ij = 0.5 * (R_t_ij + R_t_ij.T)

    R_i = 2 ** np.random.uniform(low=np.log2(max(R - stdR, 0.01)), high=np.log2(R + stdR), size=array_size)

    R_t_i = [val if idx in set(near_left + near_right) else 0 for idx, val in enumerate(R_i)]

    return R_t_i, R_t_ij


def compute_fixed_R_matrices(
        R: float,
        array_size: int,
        near_left: list[int],
        near_right: list[int],
) -> tuple[list, npt.NDArray]:
    R_t_ij = np.full((array_size, array_size), R)
    R_i = np.full(array_size, R)
    R_t_i = [
        val if idx in set(near_left + near_right) else 0 for idx, val in enumerate(R_i)
    ]

    return R_t_i, R_t_ij


def compute_C_inverse(Ch: npt.NDArray, Cv: npt.NDArray, row_num: int, periodic_y: bool) -> npt.NDArray:
    diagonal = Ch[:, :-1] + Ch[:, 1:] + Cv[:-1, :] + Cv[1:, :]
    second_diagonal = np.copy(Ch[:, 1:])
    second_diagonal[:, -1] = 0
    second_diagonal = second_diagonal.flatten()
    second_diagonal = second_diagonal[:-1]
    n_diagonal = np.copy(Cv[1:-1, :])
    C_mat = (
            np.diagflat(diagonal)
            - np.diagflat(second_diagonal, k=1)
            - np.diagflat(second_diagonal, k=-1)
            - np.diagflat(n_diagonal, k=row_num)
            - np.diagflat(n_diagonal, k=-row_num)
    )
    if periodic_y:
        offset = (row_num - 1) * row_num
        wrap_vals = Cv[0, :]
        wrap_flat = wrap_vals.flatten()

        C_mat -= np.diagflat(wrap_flat, k=offset)
        C_mat -= np.diagflat(wrap_flat, k=-offset)

    return np.linalg.inv(C_mat)  # define inverse


def compute_distributed_C_matrices(
        C: float,
        sig: float,
        row_num: int,
        array_size: int,
        near_left: list[int],
        near_right: list[int],
        periodic_y: bool,
        C_to_Cix_ratio
):
    Ch = np.random.normal(0, sig, size=(row_num, row_num + 1))
    Cv = np.random.normal(0, sig, size=(row_num + 1, row_num))

    all_Cs = np.concatenate([Ch.ravel(), Cv.ravel()])
    if np.all(all_Cs >= 0):
        pass
    else:
        min_val = -np.min(all_Cs) + 0.1

    # Ch, Cv = Ch + max(min_val, C), Cv + max(min_val, C)
    Ch, Cv = Ch + min_val + C, Cv + min_val + C

    # update Cix to be a factor Cix_ration smaller:
    Ch[:, 0] /= C_to_Cix_ratio
    Ch[:, -1] /= C_to_Cix_ratio

    # get Cix in sparse form.
    Cl = Ch[:, :-1].copy()
    Cl[:, 1:] = 0
    Cr = Ch[:, 1:].copy()
    Cr[:, :-1] = 0

    all_Cs = np.concatenate([Ch.ravel(), Cv.ravel()])
    side_Cs = np.concatenate([Cl.ravel(), Cr.ravel()])

    Cix = np.zeros(array_size)
    for i in near_left:
        Cix[i] = Cl[i // row_num][0]
    for i in near_right:
        Cix[i] = Cr[i // row_num][-1] # used to be 0 instead of -1 but doesnt matter, when Vr=0 this never contributes.

    return Cix, compute_C_inverse(Ch, Cv, row_num, periodic_y), np.mean(all_Cs), np.mean(side_Cs), np.std(
        all_Cs), np.std(side_Cs)


def compute_fixed_C_matrices(
        C: float,
        row_num: int,
        array_size: int,
        near_left: list[int],
        near_right: list[int],
        periodic_y: bool
) -> tuple[npt.NDArray, npt.NDArray]:
    Ch = np.random.normal(C, 0, size=(row_num, row_num + 1))
    Cv = np.random.normal(C, 0, size=(row_num + 1, row_num))

    Cix = np.zeros(array_size)
    for i in near_left:
        Cix[i] = np.random.normal(C, 0)
    for i in near_right:
        Cix[i] = np.random.normal(C / 2, 0)

    return Cix, compute_C_inverse(Ch, Cv, row_num, periodic_y=periodic_y)


def define_tau_inverse_matrix(
        C_inverse: npt.NDArray,
        mean_Cg: float,
        mean_Rg: float,
        array_size: int,
) -> npt.NDArray:
    res = C_inverse + np.diagflat([1 / mean_Cg] * array_size)
    return -res / mean_Rg


def recalculate_tau_dependencies(C_inverse: npt.NDArray, mean_Cg: float, mean_Rg: float, array_size: int) -> dict:
    """
    Recalculates all physics matrices and time steps that depend on Cg and Rg.
    """
    Tau_inv = define_tau_inverse_matrix(C_inverse, mean_Cg, mean_Rg, array_size)
    InvTauEigenValues, InvTauEigenVectors = np.linalg.eig(Tau_inv)
    InvTauEigenVectorsInv = np.linalg.inv(InvTauEigenVectors)
    default_dt = -0.1 / np.min(InvTauEigenValues)  # time in which Qg don't change much
    timeStep = -2 / np.max(InvTauEigenValues)
    Tau = np.linalg.inv(Tau_inv)
    matrixQnPart = -Tau / (mean_Cg * mean_Rg) - np.eye(Tau.shape[0])

    return {
        "Cg": np.array([mean_Cg] * array_size),
        "Rg": np.array([mean_Rg] * array_size),
        "Ec": 1 / (2 * mean_Cg),
        "CondRg": mean_Rg,
        "Tau_inv": Tau_inv,
        "InvTauEigenVectors": InvTauEigenVectors,
        "InvTauEigenValues": InvTauEigenValues,
        "InvTauEigenVectorsInv": InvTauEigenVectorsInv,
        "default_dt": default_dt,
        "timeStep": timeStep,
        "Tau": Tau,
        "matrixQnPart": matrixQnPart
    }


def update_init_Cg_Rg(init: ExperimentInitialState, new_mean_Cg: float, new_mean_Rg: float) -> ExperimentInitialState:
    """
    Safely updates an existing initialization object with a new Cg and Rg,
    recalculating all cascading dependencies while preserving the spatial grid.
    """
    updates = recalculate_tau_dependencies(init.C_inv, new_mean_Cg, new_mean_Rg, init.array_size)
    return replace(init, **updates)


def prepare_initial_state(loop_count: int, unitless_T0: float, flip: bool, periodic_y: bool,
                          Cg_C_ratio: float, Rg_R_ratio: float, stdR_R_ratio: float,
                          sigC_C_ratio: float) -> ExperimentInitialState:
    distribute_R = True
    distribute_C = True

    C: float = 1  # ALWAYS CHOOSE C = kb/e^2 SUCH THAT T0 EQUALS T0_UNITLESS
    R: float = 10
    mean_Cg = Cg_C_ratio * C  # default 10
    mean_Rg = Rg_R_ratio * R  # default 100
    stdR = stdR_R_ratio * R  # default 0.9
    sig = sigC_C_ratio * C  # default 0.5
    C_to_Cix_ratio = 1

    row_num = 7
    array_size = row_num ** 2
    islands = list(range(array_size))
    near_right = islands[(row_num - 1):: row_num]
    near_left = islands[0::row_num]

    if distribute_R:
        R_t_i, R_t_ij = compute_distributed_R_matrices(stdR, R, array_size, near_left, near_right)
    else:
        R_t_i, R_t_ij = compute_fixed_R_matrices(R, array_size, near_left, near_right)

    if distribute_C:
        Cix, C_inverse, mean_allCs, mean_sideCs, std_allCs, std_sideCs = compute_distributed_C_matrices(
            C, sig, row_num, array_size, near_left, near_right,
            periodic_y=periodic_y, C_to_Cix_ratio=C_to_Cix_ratio)
    else:
        Cix, C_inverse = compute_fixed_C_matrices(C, row_num, array_size, near_left, near_right,
                                                  periodic_y=periodic_y)
        mean_allCs, mean_sideCs = C, C
        std_allCs, std_sideCs = 0, 0

    # Retrieve all the Tau dependencies cleanly
    tau_dependencies = recalculate_tau_dependencies(C_inverse, mean_Cg, mean_Rg, array_size)

    return ExperimentInitialState(
        e=(e := 1),
        kB=(kB := 1),
        row_num=row_num,
        array_size=array_size,
        islands=islands,
        near_left=near_left,
        near_right=near_right,
        Vright=0,
        max_count=50000,
        distribute_R=distribute_R,
        distribute_C=distribute_C,
        T0=unitless_T0 * e * e / (C * kB),  # ALWAYS CHOOSE UNITS SUCH THAT T0 EQUALS T0_UNITLESS
        resolution=0.000001,
        Steady_state_rep=100,
        Volts=abs(e) / C,  # normalized voltage unit
        Amp=abs(e) / (C * R),  # normalized current unit
        C_avg=C,
        R_avg=R,
        loop_count=loop_count,
        R_t_ij=R_t_ij,
        R_t_i=R_t_i,
        Cix=Cix,
        C_inv=C_inverse,
        sig=sig,
        stdR=stdR,
        mean_allCs=mean_allCs,
        mean_sideCs=mean_sideCs,
        std_allCs=std_allCs,
        std_sideCs=std_sideCs,
        flip=flip,
        periodic_y=periodic_y,
        **tau_dependencies  # Unpacks Ec, Cg, Rg, CondRg, Tau_inv, etc. into the class
    )


def _calc_segments(val_str, temp_str, Ec_str, dps):
    """
    Top-level worker function to calculate segmented probabilities.
    """
    mp.dps = dps
    val = mp.mpf(val_str)
    temp = mp.mpf(temp_str)
    Ec = mp.mpf(Ec_str)

    printing = np.random.uniform(0, 1)
    if printing < 0.01:  # print ~1 in a 100
        print(f"START calculating w = {float(val):.3f} [T={float(temp)}]", flush=True)

    func = F.integrand(temp, val, Ec)
    absval = abs(val + Ec)

    # Updated limits using mpmath's infinity
    limits = [-mp.inf, -absval, mp.mpf('0'), absval, mp.inf]

    probability = mp.mpf('0')

    for i in range(len(limits) - 1):
        a = limits[i]
        b = limits[i + 1]

        if a == -mp.inf:
            # (-inf, b] to [0, 1]
            segment_prob = mp.quad(lambda t, b=b: func(b - t / (mp.mpf('1') - t)) / ((mp.mpf('1') - t) ** 2), [0, 1],
                                   method='tanh-sinh')
        elif b == mp.inf:
            # [a, inf) to [0, 1]
            segment_prob = mp.quad(lambda t, a=a: func(a + t / (mp.mpf('1') - t)) / ((mp.mpf('1') - t) ** 2), [0, 1],
                                   method='tanh-sinh')
        else:
            w = b - a
            if w < mp.mpf('1e-8'):
                continue

            # [0, 1] limits for finite memory leak mapping
            segment_prob = mp.quad(lambda t, a=a, w=w: func(a + t * w) * w, [0, 1])

        probability += segment_prob

    # Cast back to float for final lightweight array assembly
    return [float(val), float(probability.real), float(temp), float(Ec)]


def compute_gamma_worker_standard(val_str, temp_str, Ec_str, dps):
    """Picklable wrapper for multiprocessing pool starmap."""
    return _calc_segments(val_str, temp_str, Ec_str, dps)


def prepare_table_triplets(init_state, expected_list, pos_energy_bound, neg_energy_bound, max_workers):
    """
    Orchestrates the calculation of standard Gamma integrals over the SLURM CPU pool.
    Uses maxtasksperchild to force OS-level memory flushes, preventing
    mpmath quadrature cache leaks over tens of thousands of tasks.
    """
    DPS = 30
    mp.dps = DPS

    print(f"Bounds: {pos_energy_bound}, {neg_energy_bound} | Res: {init_state.resolution}", flush=True)
    num_of_calc = (pos_energy_bound - neg_energy_bound) / init_state.resolution
    num_points = round(num_of_calc)

    print(f"Computing for energies {pos_energy_bound} > dE > {neg_energy_bound}", flush=True)
    print(np.array(expected_list))

    Ec_str = str(init_state.Ec)

    # 1. generate pure high-precision space mimicking np.linspace
    w_start = mp.mpf(str(pos_energy_bound))
    w_end = mp.mpf(str(neg_energy_bound))

    if num_points > 1:
        w_values_str = [str(w_start + mp.mpf(i) * (w_end - w_start) / mp.mpf(num_points - 1)) for i in
                        range(num_points)]
    else:
        w_values_str = [str(w_start)]

    # Flatten the nested loops into a list of tuples for starmap
    task_args = []
    for w_str in w_values_str:
        for temp in expected_list:
            task_args.append((w_str, str(temp), Ec_str, DPS))

    total_tasks = len(task_args)
    print(f"Submitting {total_tasks} standard integral calculations to pool...", flush=True)

    results = []

    # execute tasks in parallel using OS-flushing multiprocessing pool
    with multiprocessing.Pool(processes=max_workers, maxtasksperchild=10) as pool:
        # pool.starmap unpacks the tuples and guarantees results are yielded in order
        futures = pool.starmap(compute_gamma_worker_standard, task_args)

        for result_row in futures:
            results.append(result_row)

    # 4. restore global precision for the main process and format the array
    mp.dps = 15
    return np.array(results, dtype=np.float64).reshape(-1, 4)


def _calc_segments_gapped_master(args, dps):
    """
    2D INTEGRATION METHOD
    Master top-level worker function to calculate segmented probabilities for quasiparticles in 2D.
    """
    val = mp.mpf(args[0])
    temp = mp.mpf(args[1])
    Ec = mp.mpf(args[2])
    D = mp.mpf(args[3])
    eps = mp.mpf(args[4])

    mp.dps = dps
    threshold = mp.mpf('1e-20')

    printing = np.random.uniform(0, 1)
    if printing < 0.01:  # print ~1 in a 100
        print(f"START calculating w = {float(val):.3f} [T={float(temp)}]", flush=True)

    abs_val = mp.fabs(val)
    sigma = mp.sqrt(2 * Ec * temp)
    bracket_width = 5 * sigma

    probability = mp.mpf('0')
    theta_max = mp.mpf('12.0')
    signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    # CASE 1: Near or inside the gap
    if abs_val < D + eps:
        def mapped_integrand(theta1, theta2, sign_E, sign_Etag):
            E = sign_E * D * mp.cosh(theta1)
            Etag = sign_Etag * D * mp.cosh(theta2) + val

            gauss_arg = -((E - Etag - Ec) ** 2) / (4 * Ec * temp)
            if gauss_arg < -200:
                return mp.mpf('0')
            gauss = mp.exp(gauss_arg) / mp.sqrt(mp.pi * 4 * Ec * temp)

            f_E = F.f(E, temp)
            f_Etag_w = F.f(Etag - val, temp)

            measure = mp.fabs(E) * mp.fabs(Etag - val)
            return measure * f_E * (mp.mpf('1') - f_Etag_w) * gauss

        for s1, s2 in signs:
            func_quadrant = lambda t1, t2, s1=s1, s2=s2: mapped_integrand(t1, t2, s1, s2)
            res = mp.quad(func_quadrant, [0, theta_max], [0, theta_max], method='gauss-legendre')
            probability += res

        return [args[0], str(probability.real), args[1], args[2]]

    # CASE 2: Far from the gap
    else:
        func = F.qp_integrand(temp, val, Ec, D)

        limits_E_far_neg = [-mp.inf, -abs_val, -D - eps]
        limits_E_far_pos = [D + eps, abs_val, mp.inf]

        mappings_E = []
        for lims in [limits_E_far_neg, limits_E_far_pos]:
            for i in range(len(lims) - 1):
                m = F.get_mapping(lims[i], lims[i + 1], threshold)
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
                        f_E = F.f(E, temp)
                        f_Etag_w = F.f(Etag - val, temp)

                        measure = mp.fabs(E) * mp.fabs(Etag - val)
                        return measure * f_E * (mp.mpf('1') - f_Etag_w) * gauss

                    prob_inner += mp.quad(inner_near_Etag, [0, theta1_max], method='gauss-legendre')

                dynamic_limits_far = [
                    -mp.inf, val - D - eps, val + D + eps,
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

                    if val - D - eps < mid < val + D + eps:
                        continue

                    def inner_far_Etag(Etag):
                        n_Etag = F.dos(Etag - val, D)
                        if n_Etag == mp.mpf('0'):
                            return mp.mpf('0')
                        gauss_arg = -((E - Etag - Ec) ** 2) / (4 * Ec * temp)
                        if gauss_arg < -200:
                            return mp.mpf('0')
                        gauss = mp.exp(gauss_arg) / mp.sqrt(mp.pi * 4 * Ec * temp)
                        f_E = F.f(E, temp)
                        f_Etag_w = F.f(Etag - val, temp)

                        measure_E = mp.fabs(E)
                        return measure_E * n_Etag * f_E * (mp.mpf('1') - f_Etag_w) * gauss

                    prob_inner += mp.quad(inner_far_Etag, [a, b], method='tanh-sinh')

                return prob_inner

            res = mp.quad(outer_integrand_near, [0, theta1_max], method='gauss-legendre')
            probability += res

        return [args[0], str(probability.real), args[1], args[2]]


def compute_gamma_worker_single(packed_args):
    """
    Calculates a SINGLE integral and returns its exact matrix indices
    so the Master Process can seamlessly reassemble the chunk.
    """
    w_str, T_str, Ec_str, D_str, eps_str, dps, chunk_idx, w_idx, T_idx = packed_args

    # Optional: ensure independent print probabilities across Linux forks
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    args = (w_str, T_str, Ec_str, D_str, eps_str)
    res = _calc_segments_gapped_master(args, dps)

    return chunk_idx, w_idx, T_idx, res


def prepare_table_triplets_gapped(init_state, expected_list, pos_energy_bound, neg_energy_bound, max_workers,
                                  gap_ratio):
    """
    Orchestrates the calculation of Gapped Gamma integrals using a flattened
    Master-Aggregator queue to guarantee 100% worker utilization.
    Includes exact BCS gap calculation for temperature-dependent Deltas.
    """
    import os
    import json
    import tempfile

    DPS = 30
    mp.dps = DPS
    print(f"dps = {DPS}")

    Ec_mp = mp.mpf(str(init_state.Ec))
    mu_str = str(Ec_mp)

    # Calculate the zero-temperature gap D_0
    D_0_float = float(init_state.Ec) * gap_ratio
    D_0_str = str(mp.mpf(str(D_0_float)))

    # Pre-calculate the exact temperature-dependent gaps for the whole profile
    print(f"Solving exact BCS self-consistency equation for {len(expected_list)} temperatures...", flush=True)
    exact_deltas_float = F.exact_bcs_gap(expected_list, D_0_float)

    # Convert to 30-DPS mpmath strings immediately to prevent float noise in workers
    exact_deltas_str = [str(mp.mpf(str(d))) for d in exact_deltas_float]
    # ------------------------------------------------

    eps_str = '1e-10'

    # --- INFER 'n' FROM TEMPERATURE PROFILE ---
    # T0 is always given by the init
    if len(expected_list) > 1:
        n_inferred = round(20.0 * (expected_list[1] - expected_list[0]) / init_state.T0)
    else:
        n_inferred = 0

    # --- Setup Checkpoint Directory ---
    # We use D_0_str in the directory name so the base gap parameter defines the folder
    checkpoint_dir = os.path.join("checkpoints", f"run_Ec_{mu_str}_D_{D_0_str}_Tstd_{n_inferred}_dynamicD")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints mapped to: {checkpoint_dir} (Inferred n={n_inferred})", flush=True)

    print(pos_energy_bound, neg_energy_bound, init_state.resolution)
    num_of_calc = (pos_energy_bound - neg_energy_bound) / init_state.resolution
    num_points = round(num_of_calc)

    # Clean temperature strings to eliminate float noise
    expected_list_strings = [str(round(float(T), 8)) for T in expected_list]

    # mimicking np.linspace(pos, neg)
    w_start = mp.mpf(str(pos_energy_bound))
    w_end = mp.mpf(str(neg_energy_bound))

    if num_points > 1:
        w_values_str = [str(w_start + mp.mpf(i) * (w_end - w_start) / mp.mpf(num_points - 1)) for i in
                        range(num_points)]
    else:
        w_values_str = [str(w_start)]

    # --- The Chunking Logic ---
    CHUNK_SIZE = 10
    w_chunks = [w_values_str[i:i + CHUNK_SIZE] for i in range(0, len(w_values_str), CHUNK_SIZE)]

    final_ordered_chunks = [None] * len(w_chunks)
    task_args = []
    loaded_count = 0

    # --- Decipher Completed Batches ---
    for chunk_idx, w_chunk_list in enumerate(w_chunks):
        file_name = f"task_chunk_{chunk_idx:04d}.json"
        if len(w_chunks) > 10000:
            file_name = f"task_chunk_{chunk_idx}.json"
        file_path = os.path.join(checkpoint_dir, file_name)

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    chunk_res = json.load(f)
                    final_ordered_chunks[chunk_idx] = chunk_res
                    loaded_count += 1
            except json.JSONDecodeError:
                # Corrupted file -> Flatten into single tasks
                for w_idx, w_str in enumerate(w_chunk_list):
                    for T_idx, T_str in enumerate(expected_list_strings):
                        # --- MODIFIED: Inject the specific gap for this T_idx ---
                        D_str_T = exact_deltas_str[T_idx]
                        task_args.append((w_str, T_str, mu_str, D_str_T, eps_str, DPS, chunk_idx, w_idx, T_idx))
        else:
            # Not yet computed -> Flatten into single tasks
            for w_idx, w_str in enumerate(w_chunk_list):
                for T_idx, T_str in enumerate(expected_list_strings):
                    # --- MODIFIED: Inject the specific gap for this T_idx ---
                    D_str_T = exact_deltas_str[T_idx]
                    task_args.append((w_str, T_str, mu_str, D_str_T, eps_str, DPS, chunk_idx, w_idx, T_idx))

    total_tasks_left = len(task_args)
    print(
        f"Found {loaded_count}/{len(w_chunks)} completed batches. Submitting {total_tasks_left} individual tasks to {max_workers} workers...",
        flush=True)

    # --- Setup Master Aggregator Buffers ---
    chunk_buffers = {}
    chunk_target_counts = {}
    for chunk_idx, w_chunk_list in enumerate(w_chunks):
        if final_ordered_chunks[chunk_idx] is None:
            chunk_buffers[chunk_idx] = []
            # How many individual integrals are required to complete this specific chunk?
            chunk_target_counts[chunk_idx] = len(w_chunk_list) * len(expected_list_strings)

    # --- Process Remaining Tasks Asynchronously ---
    if total_tasks_left > 0:
        with multiprocessing.Pool(processes=max_workers, maxtasksperchild=10) as pool:

            # imap_unordered yields results the exact millisecond ANY worker finishes
            for returned_chunk_idx, w_idx, T_idx, returned_result in pool.imap_unordered(compute_gamma_worker_single,
                                                                                         task_args):

                # Hand the result to the Master Process buffer
                chunk_buffers[returned_chunk_idx].append((w_idx, T_idx, returned_result))

                # --- ATOMIC CHECKPOINT TRIGGER ---
                # If the buffer has received all the pieces for this chunk from the various workers:
                if len(chunk_buffers[returned_chunk_idx]) == chunk_target_counts[returned_chunk_idx]:

                    # 1. Sort the buffer to perfectly match the original matrix nested-loop order
                    chunk_buffers[returned_chunk_idx].sort(key=lambda x: (x[0], x[1]))

                    # 2. Extract just the clean results
                    sorted_results = [item[2] for item in chunk_buffers[returned_chunk_idx]]

                    # 3. Master Process handles the Atomic Write securely
                    file_name = f"task_chunk_{returned_chunk_idx:04d}.json"
                    if len(w_chunks) > 10000:
                        file_name = f"task_chunk_{returned_chunk_idx}.json"

                    file_path = os.path.join(checkpoint_dir, file_name)
                    temp_fd, temp_path = tempfile.mkstemp(dir=checkpoint_dir, prefix=f"tmp_chunk_{returned_chunk_idx}_")
                    try:
                        with os.fdopen(temp_fd, 'w') as f:
                            json.dump(sorted_results, f)
                        os.replace(temp_path, file_path)
                    except Exception as e:
                        print(f"Failed to save aggregated checkpoint {returned_chunk_idx}: {e}", flush=True)
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                    # 4. Slot into final array and free memory
                    final_ordered_chunks[returned_chunk_idx] = sorted_results
                    del chunk_buffers[returned_chunk_idx]

    # --- Flatten and Format final array ---
    final_results = []
    for chunk_data in final_ordered_chunks:
        for res in chunk_data:
            final_results.append([float(res[0]), float(res[1]), float(res[2]), float(res[3])])

    # global precision for main
    mp.dps = 15
    return np.array(final_results, dtype=np.float64).reshape(-1, 4)


def output_table_triplets(table_triplets: npt.NDArray, outfile: Path) -> None:
    table_val = table_triplets[:, 0]
    table_prob = table_triplets[:, 1]
    table_T = table_triplets[:, 2]
    mu = table_triplets[0, 3]

    np.savez(outfile.as_posix(), val=table_val, prob=table_prob, temp=table_T, mu=mu)


def validate_table_triplets_file(
        triplets_file: Path, init_state: ExperimentInitialState,
        T
) -> bool:
    if not triplets_file.exists():
        warnings.warn(
            f"table triplets file does not exist (path searched: {triplets_file})"
        )
        return False

    data = np.load(triplets_file.as_posix())
    table_val = data["val"]
    table_T = data["temp"]
    mu = data["mu"]

    T_in_file = np.unique(table_T)

    # Compare with the current T, Ec
    if F.contains_allclose(needles=T, haystack=T_in_file):
        print("Existing table found for the current temperature list.")
        print(f"First val: {table_val[0]}, Last val: {table_val[-1]}")
        if np.isclose(init_state.Ec, mu):
            print("mu = Ec = " + str(mu))
            return True
        else:
            print("mu = " + str(mu))
            print("Ec = " + str(init_state.Ec))
            ValueError("mu doesn't match Ec")
    else:
        print(f"Existing table doesn't match T list for {triplets_file}")
        print(T_in_file)
        print(T)
        raise ValueError
