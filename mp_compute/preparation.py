import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
from mpmath import quad, mp

import Functions as F
from models import ExperimentInitialState


def compute_distributed_R_matrices(
        stdR: float,
        R: float,
        array_size: int,
        near_left: list,
        near_right: list,
) -> tuple[npt.NDArray, npt.NDArray]:
    R_t_ij = 2 ** np.random.uniform(
        low=np.log2(max(R - stdR, 0.01)),
        high=np.log2(R + stdR),
        size=(array_size, array_size),
    )
    R_i = 2 ** np.random.uniform(
        low=np.log2(max(R - stdR, 0.01)), high=np.log2(R + stdR), size=array_size
    )
    R_t_i = [
        val if idx in set(near_left + near_right) else 0 for idx, val in enumerate(R_i)
    ]

    return R_t_i, R_t_ij


def compute_fixed_R_matrices(
        R: float,
        array_size: int,
        near_left: list[int],
        near_right: list[int],
) -> tuple[npt.NDArray, npt.NDArray]:
    R_t_ij = np.full((array_size, array_size), R)
    R_i = np.full(array_size, R)
    R_t_i = [
        val if idx in set(near_left + near_right) else 0 for idx, val in enumerate(R_i)
    ]

    return R_t_i, R_t_ij


def compute_C_inverse(Ch: npt.NDArray, Cv: npt.NDArray, row_num: int) -> npt.NDArray:
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
    return np.linalg.inv(C_mat)  # define inverse


def compute_distributed_C_matrices(
        C: float,
        sig: float,
        row_num: int,
        array_size: int,
        near_left: list[int],
        near_right: list[int],
) -> tuple[npt.NDArray, npt.NDArray]:
    Ch = np.random.normal(0, sig, size=(row_num, row_num + 1))
    Cv = np.random.normal(0, sig, size=(row_num + 1, row_num))

    all_Cs = np.concatenate([Ch.ravel(), Cv.ravel()])
    if np.all(all_Cs >= 0):
        pass
    else:
        min_val = -np.min(all_Cs) + 0.1

    # Ch, Cv = Ch + max(min_val, C), Cv + max(min_val, C)
    Ch, Cv = Ch + min_val + C, Cv + min_val + C
    all_Cs = np.concatenate([Ch.ravel(), Cv.ravel()])

    Cl = np.random.normal(0, sig / 3, size=(1, array_size))
    Cr = np.random.normal(0, sig / 3, size=(1, array_size))

    side_Cs = np.concatenate([Cl.ravel(), Cr.ravel()])

    if np.all(side_Cs >= 0):
        pass
    else:
        min_val = -np.min(side_Cs) + 0.1

    # Cl, Cr = Cl + max(min_val, C), Cr + max(min_val, C/2)
    Cl, Cr = Cl + min_val + C, Cr + min_val + C / 2
    side_Cs = np.concatenate([Cl.ravel(), Cr.ravel()])

    Cix = np.zeros(array_size)
    for i in near_left:
        Cix[i] = Cl[0][i]
    for i in near_right:
        Cix[i] = Cr[0][i]

    for c in np.concatenate([side_Cs.ravel(), all_Cs.ravel()]):
        if c < 0:
            print(c)
            raise ValueError("Negative")

    return Cix, compute_C_inverse(Ch, Cv, row_num)


def compute_fixed_C_matrices(
        C: float,
        row_num: int,
        array_size: int,
        near_left: npt.NDArray,
        near_right: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    Ch = np.random.normal(C, 0, size=(row_num, row_num + 1))
    Cv = np.random.normal(C, 0, size=(row_num + 1, row_num))

    Cix = np.zeros(array_size)
    for i in near_left:
        Cix[i] = np.random.normal(C, 0)
    for i in near_right:
        Cix[i] = np.random.normal(C / 2, 0)

    return Cix, compute_C_inverse(Ch, Cv, row_num)


def define_tau_matrix(
        C_inverse: npt.NDArray,
        mean_Cg: float,
        mean_Rg: float,
        array_size: int,
) -> npt.NDArray:
    res = C_inverse + np.diagflat([1 / mean_Cg] * array_size)
    a = np.array([mean_Rg] * array_size)  # flattening to column
    reshaped = a.reshape((a.size, 1))
    return -res / np.repeat(reshaped, res.shape[1], axis=1)


def prepare_initial_state(loop_count: int, unitless_T0: float) -> ExperimentInitialState:
    distribute_R = True
    distribute_C = True

    C: float = 1
    R: float = 10
    mean_Cg = 10 * C
    mean_Rg = 100 * R
    stdR = 0.9 * R
    sig = 0.5 * C

    row_num = 10
    array_size = row_num ** 2
    islands = list(range(array_size))
    near_right = islands[(row_num - 1):: row_num]
    near_left = islands[0::row_num]

    if distribute_R:
        R_t_i, R_t_ij = compute_distributed_R_matrices(
            stdR, R, array_size, near_left, near_right
        )
    else:
        R_t_i, R_t_ij = compute_fixed_R_matrices(R, array_size, near_left, near_right)

    if distribute_C:
        Cix, C_inverse = compute_distributed_C_matrices(
            C, sig, row_num, array_size, near_left, near_right
        )
    else:
        Cix, C_inverse = compute_fixed_C_matrices(
            C, row_num, array_size, near_left, near_right
        )

    Tau_inv = define_tau_matrix(C_inverse, mean_Cg, mean_Rg, array_size)
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
        T0=unitless_T0 * e * e / (C * kB),
        Rg=np.array([mean_Rg] * array_size),
        Cg=np.array([mean_Cg] * array_size),
        Ec=(Ec := 1 / (2 * mean_Cg)),
        pos_energy_bound=-0.01,  # -0.01 for T=0.001; 0.08 for T=0.01; 1.3 for T=0.1 at cg = 10
        neg_energy_bound=-0.09,  # -0.09 for T=0.001; -0.19 for T=0.01; -1.4 for T=0.1 at cg = 10
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
        matrixQnPart=Tau / (mean_Cg * mean_Rg) - np.eye(Tau.shape[0]),
    )


def prepare_table_triplets(init_state: ExperimentInitialState, expected_list) -> npt.NDArray:
    rr = 0
    mp.dps = 30

    num_of_calc = (
                          init_state.pos_energy_bound - init_state.neg_energy_bound
                  ) / init_state.resolution
    vals_to_calc = np.linspace(
        init_state.pos_energy_bound, init_state.neg_energy_bound, num=round(num_of_calc)
    )
    rows = []

    T_list_to_compute = np.array(expected_list) * init_state.T0
    print(T_list_to_compute)

    for val in vals_to_calc:
        for temp in T_list_to_compute:
            probability = quad(F.integrand(temp, val, init_state.Ec), [-1, 1])
            rows.append(
                np.array(
                    [val, float(probability.real), temp, init_state.Ec],
                    dtype=np.float32,
                )
            )
            rr += 1

            total_to_calc = len(vals_to_calc) * len(T_list_to_compute)
            print(
                f"done {rr} out of {total_to_calc} -- {100 * rr / total_to_calc:.2f}%"
            )

    mp.dps = 15
    return np.ndarray(rows, dtype=np.float32).reshape(-1, 4)


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
            # ValueError("mu doesn't match Ec")
    else:
        print("Existing table doesn't match T list")
        print(T_in_file)
        print(T)
        # raise ValueError
