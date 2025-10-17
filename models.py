from dataclasses import dataclass
from pathlib import Path

import numpy.typing as npt


@dataclass
class ExperimentInitialState:
    # Tunneling Parameters
    e: int
    kB: float

    Tau_inv: npt.NDArray
    InvTauEigenValues: npt.NDArray
    InvTauEigenVectorsInv: npt.NDArray
    InvTauEigenVectors: npt.NDArray
    matrixQnPart: npt.NDArray
    Cix: npt.NDArray

    array_size: int
    row_num: int

    loop_count: int

    islands: npt.NDArray
    near_left: npt.NDArray
    near_right: npt.NDArray

    distribute_R: bool
    distribute_C: bool

    R_t_ij: npt.NDArray
    R_t_i: npt.NDArray
    CondRg: float
    Rg: npt.NDArray
    Cg: npt.NDArray
    C_avg: float
    R_avg: float
    default_dt: float
    Tau: npt.NDArray
    C_inv: npt.NDArray
    pos_energy_bound: float
    neg_energy_bound: float

    timeStep: float

    Volts: float
    Amp: float
    Vright: float
    max_count: int

    T0: float
    T_std: float
    T: list[float]
    Ec: float
    resolution: float

    Steady_state_rep: int
    expected_error: float


@dataclass
class ExportFiles:
    prepare_table_triplets_file: Path
    results_file: Path
    report_file: Path


@dataclass
class SteadyStateResult:
    loop_index: int
    error_count: int
    I_vec: npt.NDArray
