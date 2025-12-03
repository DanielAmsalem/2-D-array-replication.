from dataclasses import dataclass
from pathlib import Path

import numpy.typing as npt


@dataclass(frozen=True)
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

    islands: list[int]
    near_left: list[int]
    near_right: list[int]

    distribute_R: bool
    distribute_C: bool

    R_t_ij: npt.NDArray
    R_t_i: list
    CondRg: float
    Rg: npt.NDArray
    Cg: npt.NDArray
    C_avg: float
    R_avg: float
    default_dt: float
    Tau: npt.NDArray
    C_inv: npt.NDArray

    timeStep: float

    Volts: float
    Amp: float
    Vright: float
    max_count: int

    T0: float
    Ec: float
    resolution: float

    Steady_state_rep: int
    sig: float
    stdR: float
    mean_allCs: float
    mean_sideCs: float
    std_allCs: float
    std_sideCs: float

    flip: bool


@dataclass
class IMPORT_EXPORT:
    plot_results: bool
    export_path: Path
    prepare_table_triplets_file_list: list[Path]
    csv_table_path: Path
    results_dir_path: Path

@dataclass
class SteadyStateResult:
    loop_index: int
    error_count: int
    I_vec: npt.NDArray
    Jx: npt.NDArray
    Jy: npt.NDArray

