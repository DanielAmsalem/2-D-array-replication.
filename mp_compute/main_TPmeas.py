from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import datetime
import warnings
import Functions as F
import numpy as np
from define_objects import IMPORT_EXPORT, SteadyStateResult, ExperimentInitialState
from gamma_functions import Get_Steady_State
from preparation import (
    prepare_initial_state,
    validate_table_triplets_file,
    prepare_table_triplets,
    output_table_triplets,
)
import curve_plotter
from dataclasses import asdict
import orjson
import csv
import math
import time


def main(import_export: IMPORT_EXPORT, run_name) -> None:
    # FIXED PARAMETERS
    loop_count = 100
    T0_unitless = 0.001
    repetition = 2  # int : m -> the first gradient to check will be dT=(m+1)Tstd
    last_repetition_to_do = 19  # int : n -> the last repetition has dT = n*Tstd
    flip = True
    first_run = False
    rep_json = True
    null_path_name = import_export.export_path / f"table_triplets_T0_e{round(math.log10(T0_unitless))}.npz"
    pos_energy_boundT0 = -0.01  # -0.01 for T=0.001; 0.14 for T=0.01; 1.7 for T=0.1 at cg = 10
    neg_energy_boundT0 = -0.09  # -0.09 for T=0.001; -0.24 for T=0.01; -1.8 for T=0.1 at cg = 10

    # choose a specific run
    run_to_get_init_from = "20251117_18h02m00s"
    results_dir_of_past_run = Path(__file__).parent.parent / f"results_{run_to_get_init_from}"
    infile = Path(results_dir_of_past_run / f"{run_to_get_init_from}.json")
    if infile.exists():
        json_txt = infile.read_text()
        raw_fields = orjson.loads(json_txt)
        # recreate old init state
        init_str = ExperimentInitialState(**raw_fields)
        # if is old 20251117_18h02m00s run, put init_str = ExperimentInitialState(**raw_fields, flip=flip)
        init = F.fix_types(init_str)
        print("success")

    else:
        # create new initial state
        init = prepare_initial_state(loop_count=loop_count, unitless_T0=T0_unitless, flip=flip) # BEFORE NEXT RUN ADD FLIP=FLIP HERE!!!!!!

    ### report init state to report file
    if rep_json:
        outfile = Path(import_export.results_dir_path / f"{run_name}.json")
        raw_fields = asdict(init)
        serialized_init_data = orjson.dumps(raw_fields, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")
        outfile.write_text(serialized_init_data)

    if not validate_table_triplets_file(null_path_name, init, [init.T0]):
        table_triplets = prepare_table_triplets(init, [init.T0],
                                                pos_energy_bound=pos_energy_boundT0,
                                                neg_energy_bound=neg_energy_boundT0)
        output_table_triplets(table_triplets, null_path_name)
        table_val = table_triplets[:, 0]
        table_prob = table_triplets[:, 1]
        table_T = [init.T0]
    else:
        table_triplets = np.load(null_path_name.as_posix())
        table_val = table_triplets["val"]
        table_prob = table_triplets["prob"]
        table_T = np.unique(table_triplets["temp"]).tolist()

    # RUN PARAMETERS
    V_diff = 4
    steps = 100
    Vleft = np.linspace(init.Vright * init.Volts, (init.Vright + V_diff) * init.Volts, num=steps)
    V_doubled = np.concatenate([Vleft, Vleft[-2::-1]])
    cycles = len(V_doubled)
    T = [init.T0] * init.row_num
    expected_err = 0.01 * (init.row_num - 1)

    if first_run:
        t0 = time.time()
        with ProcessPoolExecutor() as executor:
            loaded_state_function = partial(
                Get_Steady_State,
                init=init,
                V_cycle=V_doubled,
                cycles=cycles,
                table_val=table_val,
                table_prob=table_prob,
                flip=flip,
                table_T=table_T,
                T=T,
                expected_error=expected_err,
                pos_energy_bound=pos_energy_boundT0,
                neg_energy_bound=neg_energy_boundT0,
                repetition=0
            )

            results: list[SteadyStateResult] = list(executor.map(loaded_state_function, range(init.loop_count)))

        ### find smallest I(V) > 2
        I_vec_avg, I_vec_std = curve_plotter.iv_curve_compute_and_save_csv(init=init,
                                                                           filename=run_name,
                                                                           results=results,
                                                                           Vleft=Vleft,
                                                                           repetition=0,
                                                                           results_path=import_export.results_dir_path)

        ### report run specific output
        curve_plotter.report_param(init=init,
                                   filename=run_name,
                                   repetition=0,
                                   T=T,
                                   expected_error=expected_err,
                                   loop_count=init.loop_count,
                                   T_std=0,
                                   t0=t0,
                                   results_path=import_export.results_dir_path)

        index_2 = np.searchsorted(I_vec_avg, 2 * init.Amp, side="right")
        if index_2 >= len(I_vec_avg):
            warnings.warn("no I(V) larger than 2 in initial Tstd=0 run")
    else:
        print("skipped first run")
        index_2 = 90

    ### get bounds for dE for each temp from csv table
    with open(import_export.csv_table_path) as f:
        rows = list(csv.reader(f))
        neg = [row[1] for row in rows]
        pos = [row[2] for row in rows]

    ### run thermopower until I(V)<0
    current_at_V0 = True

    while current_at_V0:
        repetition += 1
        if repetition > last_repetition_to_do:
            print("Tstd>T0, finished all runs for Tstd<=T0")
            current_at_V0 = False
            continue

        # new temperature profile
        T_std = repetition * init.T0 / 20
        T_list_to_compute = [init.T0 + i * T_std for i in range(init.row_num)]

        ### check if there is valid table for new dT
        if not validate_table_triplets_file(import_export.prepare_table_triplets_file_list[repetition],
                                            init,
                                            np.array(T_list_to_compute)):
            warnings.warn(f"no validated table, skipped rep{T_std}")
            continue
        else:
            table_triplets = np.load(import_export.prepare_table_triplets_file_list[repetition].as_posix())
            table_val = table_triplets["val"]
            table_prob = table_triplets["prob"]
            table_T = np.unique(table_triplets["temp"]).tolist()

        ### run repetition for new dT
        with ProcessPoolExecutor() as executor:
            t0 = time.time()
            T = np.linspace(init.T0, init.T0 + init.row_num * T_std, init.row_num)
            loaded_state_function = partial(
                Get_Steady_State,
                init=init,
                V_cycle=V_doubled,
                cycles=cycles,
                table_val=table_val,
                table_prob=table_prob,
                table_T=table_T,
                flip=flip,
                T=T_list_to_compute,
                expected_error=expected_err * np.sqrt(max(T) / init.T0),
                pos_energy_bound=float(pos[repetition - 3]),
                neg_energy_bound=float(neg[repetition - 3]),
                repetition=repetition
            )

            results: list[SteadyStateResult] = list(
                executor.map(loaded_state_function, range(init.loop_count))
            )

        ### find I(V) with gradient dT
        I_vec_avg, I_vec_std = curve_plotter.iv_curve_compute_and_save_csv(init=init,
                                                                           filename=run_name,
                                                                           results=results,
                                                                           Vleft=Vleft,
                                                                           repetition=repetition,
                                                                           results_path=import_export.results_dir_path)

        ### report run specific parameters
        curve_plotter.report_param(init=init,
                                   filename=run_name,
                                   repetition=repetition,
                                   T=T_list_to_compute,
                                   expected_error=expected_err,
                                   loop_count=init.loop_count,
                                   T_std=0,
                                   t0=t0,
                                   results_path=import_export.results_dir_path)

        if I_vec_avg[index_2] < 0:
            current_at_V0 = False


if __name__ == "__main__":
    date_ = datetime.datetime.now()
    run_name_flat = date_.strftime("%Y%m%d_%Hh%Mm%Ss")

    EXPORT_PATH = Path(__file__).parent.parent / "export"
    MP_COMPUTE_PATH = Path(__file__).parent.parent / "mp_compute"
    RESULTS_DIR_PATH = Path(__file__).parent.parent / f"results_{run_name_flat}"
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)

    main(
        IMPORT_EXPORT(
            plot_results=True,
            export_path=EXPORT_PATH,
            prepare_table_triplets_file_list=[EXPORT_PATH / f"table_triplets_Tstd{n}_20.npz" for n in range(20)],
            csv_table_path=MP_COMPUTE_PATH / f"table.csv",
            results_dir_path=RESULTS_DIR_PATH
        ),
        run_name=run_name_flat
    )
