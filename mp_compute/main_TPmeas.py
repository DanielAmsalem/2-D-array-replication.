import os

ratio = 2
os.environ["OPENBLAS_NUM_THREADS"] = str(ratio)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
num_workers = int(total_cpus / ratio)
print(f"worker number set to {num_workers} ; for {total_cpus} cpus", flush=True)

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
    # RUN TYPE
    flip = False
    print(f"flip = {flip}", flush=True)
    first_run = True  #############################
    rep_json = True
    periodic_y = True  # periodic boundary conditions in y-axis
    plot_ongoing_voltage_map = False

    # FIXED PARAMETERS
    V_capture = 4
    pos_energy_boundT0 = 0.01  # -0.01 for T=0.001; 0.14 for T=0.01; 1.7 for T=0.1 at cg = 10
    neg_energy_boundT0 = -0.11  # -0.09 for T=0.001; -0.24 for T=0.01; -1.8 for T=0.1 at cg = 10

    # EXPERIMENT PARAMETERS
    loop_count = max(num_workers, 100)
    T0_unitless = 0.001
    repetition = 0  # int : m -> the first gradient to check will be dT=(m+1)Tstd

    ######## CHANGABLES ###############
    last_repetition_to_do = 501  # int : n -> the last repetition has dT = n*Tstd
    repetition_list = list(range(1, 10, 4))
    gap_ratio = 0
    ###################################

    # MESSAGES
    print(f"loop max: {loop_count}", flush=True)
    print(f"gap_ratio = {gap_ratio}", flush=True)
    print(f"repeating for dT=n*Tstd, n = {repetition_list}", flush=True)
    if loop_count != 1:
        plot_ongoing_voltage_map = False

    null_path_name = import_export.export_path / f"64bit_table_triplets_T0_e{round(math.log10(T0_unitless))}.npz"

    # choose a specific run
    run_to_get_init_from = "20251207_17h43m26s"
    results_dir_of_past_run = Path(__file__).parent.parent / f"results_{run_to_get_init_from}"
    infile = Path(results_dir_of_past_run / f"{run_to_get_init_from}.json")
    if infile.exists():
        json_txt = infile.read_text()
        raw_fields = orjson.loads(json_txt)
        # recreate old init state
        init_str = ExperimentInitialState(**raw_fields)
        init = F.fix_types(init_str, loop_count)
        init = F.swap_in_init("flip", flip, init)
        if init.T0 != T0_unitless:
            print("T0 is different in reference file or Temperature units != 1. switching.")
            init = F.swap_in_init("T0", T0_unitless, init)
            print(f"T0 is now {init.T0}")
        print(f"success, starting run for {run_to_get_init_from}", flush=True)

    else:
        # create new initial state
        init = prepare_initial_state(loop_count=loop_count, unitless_T0=T0_unitless, flip=flip, periodic_y=periodic_y)
        print("CREATED NEW INIT FILE")

    ### report init state to report file
    if rep_json:
        outfile = Path(import_export.results_dir_path / f"{run_name}.json")
        raw_fields = asdict(init)
        serialized_init_data = orjson.dumps(raw_fields, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")
        outfile.write_text(serialized_init_data)
        print("STORED INIT IN JSON", flush=True)

    Rx, Ry = curve_plotter.extract_nn_resistances(init.R_t_ij, init.row_num, init.R_t_i,
                                                  near_left=init.near_left,
                                                  near_right=init.near_right,
                                                  periodic_y=periodic_y, )
    print("created resistance maps, now saving plots...", flush=True)
    curve_plotter.plot_resistance_maps(Rx, Ry, n=init.row_num,
                                       results_path=import_export.results_dir_path, show=False)
    print("plotting capacitance map...", flush=True)
    curve_plotter.plot_capacitance_map(init.C_inv, n=init.row_num,
                                       results_path=import_export.results_dir_path, show=False, periodic_y=periodic_y)

    if not validate_table_triplets_file(null_path_name, init, [init.T0]) and first_run:
        table_triplets = prepare_table_triplets(init, [init.T0],
                                                pos_energy_bound=pos_energy_boundT0,
                                                neg_energy_bound=neg_energy_boundT0,
                                                max_workers=num_workers)
        output_table_triplets(table_triplets, null_path_name)
        table_val = table_triplets[:, 0]
        table_prob = table_triplets[:, 1]
        table_T = [init.T0]
    elif first_run:
        table_triplets = np.load(null_path_name.as_posix())
        table_val = table_triplets["val"]
        table_prob = table_triplets["prob"]
        table_T = np.unique(table_triplets["temp"]).tolist()

    # RUN PARAMETERS
    V_diff = 4
    steps = 100
    Vleft = np.linspace(init.Vright * init.Volts, (init.Vright + V_diff) * init.Volts, num=steps)
    V_capture_idx = np.searchsorted(Vleft, V_capture, side='left')
    V_capture = float(Vleft[V_capture_idx])
    V_doubled = np.concatenate([Vleft, Vleft[-2::-1]])
    cycles = len(V_doubled)
    T = [init.T0] * init.row_num
    expected_err = F.calc_expected_dist_std(T, init.T0)

    if first_run:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            actual_workers = executor._max_workers
            print(f"running with {actual_workers} workers", flush=True)
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
                repetition=0,
                capture_heatmap_at_idx=V_capture_idx,
                periodic_y=periodic_y,
                plot_ongoing_voltage_map=plot_ongoing_voltage_map,
                gap_ratio=gap_ratio,
            )

            results: list[SteadyStateResult] = list(executor.map(loaded_state_function, range(init.loop_count)))

        ### find smallest I(V) > 2
        curve_plotter.iv_curve_compute_and_save_csv(init=init,
                                                    filename=run_name,
                                                    results=results,
                                                    Vleft=Vleft,
                                                    repetition=0,
                                                    results_path=import_export.results_dir_path,
                                                    get_heatmap=True,
                                                    heatmap_at_V=V_capture)

        ### get errors
        err = 0
        for run in results:
            err += run.error_count

        ### report run specific output
        curve_plotter.report_param(init=init,
                                   filename=run_name,
                                   repetition=0,
                                   T=T,
                                   expected_error=expected_err,
                                   loop_count=init.loop_count,
                                   T_std=0,
                                   t0=t0,
                                   results_path=import_export.results_dir_path,
                                   tot_error_count=err,
                                   gap_ratio=gap_ratio,
                                   )

    else:
        print("skipped first run")

    ### get bounds for dE for each temp from csv table
    bounds_dict = {}
    with open(import_export.csv_table_path) as f:
        for row in csv.reader(f):
            # Ensure row is not empty and row[0] is a valid integer
            if row and row[0].strip().lstrip('-').isdigit():
                rep_idx = int(row[0])
                bounds_dict[rep_idx] = {
                    "neg": float(row[1]),
                    "pos": float(row[2])
                }

    ### run thermopower until I(V)<0
    current_at_V0 = True

    while current_at_V0:
        repetition += 1
        if repetition > last_repetition_to_do:
            print("Tstd>T0, finished all runs for Tstd<=T0", flush=True)
            current_at_V0 = False
            continue
        if not int(repetition) in repetition_list:
            print(f"repetition {repetition} was skipped")
            continue

        if repetition not in bounds_dict:
            print(f"repetition {repetition} missing from bounds table, skipped")
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
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            t0 = time.time()
            T = T_list_to_compute
            if flip:
                T = np.flip(T_list_to_compute)
            expected_err = F.calc_expected_dist_std(T, init.T0)
            print(expected_err, flush=True)
            loaded_state_function = partial(
                Get_Steady_State,
                init=init,
                V_cycle=V_doubled,
                cycles=cycles,
                table_val=table_val,
                table_prob=table_prob,
                table_T=table_T,
                flip=flip,
                T=T,
                expected_error=expected_err,
                pos_energy_bound=bounds_dict[repetition]["pos"],
                neg_energy_bound=bounds_dict[repetition]["neg"],
                repetition=repetition,
                capture_heatmap_at_idx=V_capture_idx,
                periodic_y=periodic_y,
                plot_ongoing_voltage_map=plot_ongoing_voltage_map,
                gap_ratio=gap_ratio,
            )

            results: list[SteadyStateResult] = list(
                executor.map(loaded_state_function, range(init.loop_count))
            )

        ### find I(V) with gradient dT
        curve_plotter.iv_curve_compute_and_save_csv(init=init,
                                                    filename=run_name,
                                                    results=results,
                                                    Vleft=Vleft,
                                                    repetition=repetition,
                                                    results_path=import_export.results_dir_path,
                                                    get_heatmap=True,
                                                    heatmap_at_V=V_capture)

        err = 0
        for run in results:
            err += run.error_count

        ### report run specific parameters
        curve_plotter.report_param(init=init,
                                   filename=run_name,
                                   repetition=repetition,
                                   T=T,
                                   expected_error=expected_err,
                                   loop_count=init.loop_count,
                                   T_std=T_std,
                                   t0=t0,
                                   results_path=import_export.results_dir_path,
                                   tot_error_count=err,
                                   gap_ratio=gap_ratio)


if __name__ == "__main__":
    date_ = datetime.datetime.now()
    run_name_flat = date_.strftime("%Y%m%d_%Hh%Mm%Ss")

    EXPORT_PATH = Path(__file__).parent.parent / "export"
    MP_COMPUTE_PATH = Path(__file__).parent.parent / "mp_compute"
    RESULTS_DIR_PATH = Path(__file__).parent.parent / f"results_{run_name_flat}"
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Created results directory at {RESULTS_DIR_PATH}")

    main(
        IMPORT_EXPORT(
            plot_results=True,
            export_path=EXPORT_PATH,
            prepare_table_triplets_file_list=[EXPORT_PATH / f"64bit_table_triplets_Tstd{n}_20.npz" for n in range(501)],
            csv_table_path=MP_COMPUTE_PATH / f"table.csv",
            results_dir_path=RESULTS_DIR_PATH
        ),
        run_name=run_name_flat
    )
