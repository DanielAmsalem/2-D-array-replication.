from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import datetime
import warnings

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

from models import Export, SteadyStateResult
from gamma_functions import Get_Steady_State
from preparation import (
    prepare_initial_state,
    validate_table_triplets_file,
    prepare_table_triplets,
    output_table_triplets,
)
import Functions as F
import curve_plotter

from dataclasses import asdict
import orjson
import csv

EXPORT_PATH = Path(__file__).parent.parent / "export"


def main(export: Export) -> None:
    loop_count = 5
    T0_unitless = 0.001
    null_path_name = EXPORT_PATH / "table_triplets.npz"
    pos_energy_boundT0 = -0.01,  # -0.01 for T=0.001; 0.14 for T=0.01; 1.7 for T=0.1 at cg = 10
    neg_energy_boundT0 = -0.09,  # -0.09 for T=0.001; -0.24 for T=0.01; -1.8 for T=0.1 at cg = 10

    init = prepare_initial_state(loop_count=loop_count,
                                 unitless_T0=T0_unitless)

    date_ = datetime.datetime.now()
    run_name = date_.strftime("%Y%m%d, %Hh%Mm%Ss")

    ### report parameters of run to report file
    outfile = Path(EXPORT_PATH / f"{run_name}.json")
    raw_fields = asdict(init)
    serialized_init_data = orjson.dumps(raw_fields, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")
    outfile.write_text(serialized_init_data)

    if not validate_table_triplets_file(null_path_name, init, [init.T0]):
        table_triplets = prepare_table_triplets(init, [init.T0], pos_energy_boundT0, neg_energy_boundT0)
        output_table_triplets(table_triplets, null_path_name)
        table_val = table_triplets[:, 0]
        table_prob = table_triplets[:, 1]
        table_T = [init.T0]
    else:
        table_triplets = np.load(null_path_name.as_posix())
        table_val = table_triplets["val"]
        table_prob = table_triplets["prob"]
        table_T = np.unique(table_triplets["temp"]).tolist()

    V_diff = 4
    steps = 100
    Vleft = np.linspace(
        init.Vright * init.Volts,
        (init.Vright + V_diff) * init.Volts,
        num=steps,
    )
    V_doubled = np.concatenate([Vleft, Vleft[-2::-1]])
    cycles = len(V_doubled)

    ## first run
    with ProcessPoolExecutor() as executor:
        loaded_state_function = partial(
            Get_Steady_State,
            init=init,
            V_cycle=V_doubled,
            cycles=cycles,
            table_val=table_val,
            table_prob=table_prob,
            table_T=table_T,
            T=[init.T0],
            expected_error=0.01 * (init.row_num - 1),
            pos_energy_bound=pos_energy_boundT0,
            neg_energy_bound=neg_energy_boundT0
        )

        results: list[SteadyStateResult] = list(
            executor.map(loaded_state_function, range(init.loop_count))
        )

    ## find smallest I(V) > 2
    I_vec_avg, I_vec_std = curve_plotter.iv_curve_computer(init=init,
                                                           filename=run_name,
                                                           results=results,
                                                           Vleft=Vleft,
                                                           repetition=0)

    index_2 = np.searchsorted(I_vec_avg, 2 * init.Amp, side="right")
    if index_2 < len(I_vec_avg):
        V2 = Vleft[index_2]  # find voltage for the current at 2 amper
    else:
        raise ValueError("no I(V) larger than 2")

    ### get bounds for dE for each temp from csv table
    with open("table.csv") as f:
        rows = list(csv.reader(f))
        neg = [row[1] for row in rows]
        pos = [row[2] for row in rows]

    ## run thermopower until I(V)<0
    current_at_V0 = True
    repetition = 0  # how many T_std the gradient had before starting repetition.

    while current_at_V0:
        ### check if there is valid table for new dT
        repetition += 1
        if repetition > 20:
            print("Tstd>T0, finished all runs for Tstd<=T0")
            current_at_V0 = False

        # new temperature profile
        T_std = repetition * init.T0 / 20
        T_list_to_compute = [init.T0 + i * init.T0 * T_std for i in range(init.row_num)]

        if not validate_table_triplets_file(export.prepare_table_triplets_file_list[repetition],
                                            init,
                                            np.array(T_list_to_compute)):
            warnings.warn(f"no validated table, skipped rep{T_std}")
            continue
        else:
            table_triplets = np.load(export.prepare_table_triplets_file_list[repetition].as_posix())
            table_val = table_triplets["val"]
            table_prob = table_triplets["prob"]
            table_T = np.unique(table_triplets["temp"]).tolist()

        ### run repetition for new dT
        with ProcessPoolExecutor() as executor:
            loaded_state_function = partial(
                Get_Steady_State,
                init=init,
                V_cycle=V_doubled,
                cycles=cycles,
                table_val=table_val,
                table_prob=table_prob,
                table_T=table_T,
                T=(T := np.linspace(init.T0, init.T0 + init.row_num * T_std, init.row_num)),
                expected_error=0.01 * (init.row_num - 1) * np.sqrt(max(T) / init.T0),
                pos_energy_bound=pos[repetition-3],
                neg_energy_bound=neg[repetition-3],
            )

            results: list[SteadyStateResult] = list(
                executor.map(loaded_state_function, range(init.loop_count))
            )

        ## find I(V) with gradient dT
        I_vec_avg, I_vec_std = curve_plotter.iv_curve_computer(init=init,
                                                               filename=run_name,
                                                               results=results,
                                                               Vleft=Vleft,
                                                               repetition=repetition)

        if I_vec_avg[index_2] < 0:
            current_at_V0 = False


if __name__ == "__main__":
    main(
        Export(
            plot_results=True,
            prepare_table_triplets_file_list=[EXPORT_PATH / f"table_triplets_Tstd{n}_20.npz" for n in range(20)],
            results_file=EXPORT_PATH / "tmp",
        )
    )
