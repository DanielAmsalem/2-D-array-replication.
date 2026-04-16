import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
from mpmath import quad, mp

import Functions as F
from define_objects import ExperimentInitialState

import concurrent.futures


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
        Cix[i] = Cr[i // row_num][0]

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


def prepare_initial_state(loop_count: int, unitless_T0: float, flip: bool, periodic_y: bool) -> ExperimentInitialState:
    distribute_R = True
    distribute_C = True

    C: float = 1  # ALWAYS CHOOSE C = kb/e^2 SUCH THAT T0 EQUALS T0_UNITLESS
    R: float = 10
    mean_Cg = 10 * C
    mean_Rg = 100 * R
    stdR = 0.9 * R
    sig = 0.5 * C
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

    Tau_inv = define_tau_inverse_matrix(C_inverse, mean_Cg, mean_Rg, array_size)
    InvTauEigenValues, InvTauEigenVectors = np.linalg.eig(Tau_inv)
    InvTauEigenVectorsInv = np.linalg.inv(InvTauEigenVectors)
    default_dt = -0.1 / np.min(InvTauEigenValues)  # time in which Qg don't change much
    timeStep = -2 / np.max(InvTauEigenValues)
    Tau = np.linalg.inv(Tau_inv)

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
        Rg=np.array([mean_Rg] * array_size),
        Cg=np.array([mean_Cg] * array_size),
        Ec=1 / (2 * mean_Cg),
        resolution=0.000001,
        Steady_state_rep=100,
        Volts=abs(e) / C,  # normalized voltage unit
        Amp=abs(e) / (C * R),  # normalized current unit
        CondRg=mean_Rg,
        C_avg=C,
        R_avg=R,
        loop_count=loop_count,
        R_t_ij=R_t_ij,
        R_t_i=R_t_i,
        Cix=Cix,
        C_inv=C_inverse,
        Tau_inv=Tau_inv,
        InvTauEigenVectors=InvTauEigenVectors,
        InvTauEigenValues=InvTauEigenValues,
        InvTauEigenVectorsInv=InvTauEigenVectorsInv,
        default_dt=default_dt,
        timeStep=timeStep,
        Tau=Tau,
        matrixQnPart=-Tau / (mean_Cg * mean_Rg) - np.eye(Tau.shape[0]), # Tau / (mean_Cg * mean_Rg) - np.eye(Tau.shape[0])
        sig=sig,
        stdR=stdR,
        mean_allCs=mean_allCs,
        mean_sideCs=mean_sideCs,
        std_allCs=std_allCs,
        std_sideCs=std_sideCs,
        flip=flip,
        periodic_y=periodic_y,
    )


def _calc_segments(args):
    """
    Top-level worker function to calculate segmented probabilities.
    """
    val, temp, Ec = args

    mp.dps = 40

    func = F.integrand(temp, val, Ec)
    absval = abs(val + Ec)
    limits = [-1.5, - absval, 0, absval, 2]

    probability = 0

    for i in range(len(limits) - 1):
        a = limits[i]
        b = limits[i + 1]
        w = b - a
        if w < 1e-8:
            continue

        # fixed [0, 1] limits for memory leak mapping
        segment_prob = mp.quad(lambda t, a=a, w=w: func(a + t * w) * w, [0, 1], method='tanh-sinh')
        probability += segment_prob

    return [val, float(probability.real), temp, Ec]


def prepare_table_triplets(init_state, expected_list, pos_energy_bound, neg_energy_bound, max_workers):
    print(pos_energy_bound, neg_energy_bound, init_state.resolution)
    num_of_calc = (pos_energy_bound - neg_energy_bound) / init_state.resolution
    vals_to_calc = np.linspace(pos_energy_bound, neg_energy_bound, num=round(num_of_calc))

    T_list_to_compute = np.array(expected_list)
    print(f"computing for energies {pos_energy_bound} > dE > {neg_energy_bound}")
    print(T_list_to_compute)

    # 1. Flatten the nested loops into a list of tasks
    tasks = []
    for val in vals_to_calc:
        for temp in T_list_to_compute:
            tasks.append((val, temp, init_state.Ec))

    total_to_calc = len(tasks)
    rows = []

    # 2. Execute tasks in parallel using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # executor.map handles distributing the tasks and guarantees
        # the results are yielded in the original submission order.
        # chunksize groups tasks to reduce communication overhead.
        results = executor.map(_calc_segments, tasks, chunksize=10)

        for result_row in results:
            rows.append(result_row)

    # 3. Restore global precision for the main process and format the array
    mp.dps = 15
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


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
