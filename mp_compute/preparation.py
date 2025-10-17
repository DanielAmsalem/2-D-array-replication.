import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
from mpmath import quad, mp

import Conditions as Cond
import Functions as F
from models import ExperimentInitialState


def prepare_initial_state(loop_count: int) -> ExperimentInitialState:
    return ExperimentInitialState(
        e=Cond.e,
        kB=Cond.kB,
        Tau_inv=Cond.Tau_inv,
        InvTauEigenValues=Cond.InvTauEigenValues,
        InvTauEigenVectorsInv=Cond.InvTauEigenVectorsInv,
        InvTauEigenVectors=Cond.InvTauEigenVectors,
        matrixQnPart=Cond.matrixQnPart,
        Cix=Cond.Cix,
        array_size=Cond.array_size,
        row_num=Cond.row_num,
        islands=Cond.islands,
        near_left=Cond.near_left,
        near_right=Cond.near_right,
        loop_count=loop_count,
        R_t_ij=Cond.R_t_ij,
        R_t_i=Cond.R_t_i,
        CondRg=Cond.Rg,
        Rg=np.array([Cond.Rg] * Cond.array_size),
        Cg=(Cg := np.array([Cond.Cg] * Cond.array_size)),
        C_avg=Cond.C,
        R_avg=Cond.R,
        default_dt=Cond.default_dt,
        Tau=Cond.Tau,
        C_inv=Cond.C_inverse,
        Volts=abs(Cond.e) / Cond.C,  # normalized voltage unit
        Amp=abs(Cond.e) / (Cond.C * Cond.R),  # normalized current unit
        Vright=0,
        pos_energy_bound=-0.01,  # -0.01 for T=0.001; 0.08 for T=0.01; 1.3 for T=0.1 at cg = 10
        neg_energy_bound=-0.09,  # -0.09 for T=0.001; -0.19 for T=0.01; -1.4 for T=0.1 at cg = 10
        max_count=50000,
        distribute_R=True,
        distribute_C=True,
        # T should always be written as np.linspace(T0, T0 + row_num * T_std, row_num) prev to 12/09/25 used to be minus
        # for fixed temp take np.ones(row_num) * T
        # for some flipped gradient use np.flip(T, axis=0)
        T0=(T0 := 0.001 * Cond.e * Cond.e / (Cond.C * Cond.kB)),
        T_std=T0 / 20,
        T=(T := [T0]),
        Ec=1 / (2 * np.mean(Cg)),
        resolution=0.000001,
        Steady_state_rep=100,
        expected_error=0.01 * (Cond.row_num - 1) * np.sqrt(max(T) / T0),
        timeStep=Cond.timeStep,
    )


def prepare_table_triplets(init_state: ExperimentInitialState) -> npt.NDArray:
    rr = 0
    mp.dps = 30

    num_of_calc = (
        init_state.pos_energy_bound - init_state.neg_energy_bound
    ) / init_state.resolution
    vals_to_calc = np.linspace(
        init_state.pos_energy_bound, init_state.neg_energy_bound, num=round(num_of_calc)
    )
    rows = []

    for val in vals_to_calc:
        for temp in init_state.T:
            probability = quad(F.integrand(temp, val, init_state.Ec), [-1, 1])
            rows.append(
                np.array(
                    [val, float(probability.real), temp, init_state.Ec],
                    dtype=np.float32,
                )
            )
            rr += 1

            total_to_calc = len(vals_to_calc) * len(init_state.T)
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
    triplets_file: Path, init_state: ExperimentInitialState
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
    if F.contains_allclose(needles=np.array(init_state.T), haystack=T_in_file):
        print("Existing table found for the current temperature list.")
        print(f"First val: {table_val[0]}, Last val: {table_val[-1]}")
        if np.isclose(init_state.Ec, mu):
            print("mu = Ec = " + str(mu))
            return True
        else:
            print("mu = " + str(mu))
            raise ValueError("mu doesn't match Ec")
    else:
        print("Existing table doesn't match T list")
        print(T_in_file)
        print(np.array(init_state.T))
        raise ValueError
