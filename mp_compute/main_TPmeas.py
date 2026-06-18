import os

ratio = 3 / 2
os.environ["OPENBLAS_NUM_THREADS"] = str(ratio)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
num_workers = max(int(total_cpus / ratio), 80)
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
    update_init_Cg_Rg
)
import curve_plotter
from dataclasses import asdict
import orjson
import csv
import math
import time
import re

####### slurm parameter parsing from job name ######
job_name = os.environ.get('SLURM_JOB_NAME', 'TPmeas1_11_4_Cg2')
pattern = r"(Reverse_?)?TPmeas(\d+)_(\d+)_(\d+)_Cg(\d+)"
match = re.search(pattern, job_name)

if match:
    # match.group(1) will be 'Reverse' or 'Reverse_' if it exists, otherwise None
    is_reverse = match.group(1) is not None
    x = int(match.group(2))
    last_rep = int(match.group(3))
    jumps = int(match.group(4))
    Cg = int(match.group(5))

    repetition_list = list(range(x, last_rep, jumps))
    print(f"Parsed from Job Name '{job_name}': flip={is_reverse}, repetition_list={repetition_list}, Cg={Cg}", flush=True)

else:
    raise NameError(f"Job Name is improperly formatted : {job_name}")

Cg_list = [2, 5, 10, 20, 50]
##############################################################


def main(import_export: IMPORT_EXPORT, run_name, mean_Cg, first_rep) -> None:
    # RUN TYPE
    flip = is_reverse
    first_run = False
    if first_rep == 0:
        first_run = True
        first_rep = 1
    rep_json = True
    periodic_y = True  # periodic boundary conditions in y-axis
    plot_ongoing_voltage_map = False

    # FIXED PARAMETERS
    V_capture = 4
    if Cg in Cg_list:
        if Cg == 20:
            pos_energy_boundT0 = 0.02  # -0.01 for T=0.001; 0.14 for T=0.01; 1.7 for T=0.1 at cg = 10
            neg_energy_boundT0 = -0.07  # -0.09 for T=0.001; -0.24 for T=0.01; -1.8 for T=0.1 at cg = 10
        elif Cg == 50:
            pos_energy_boundT0 = 0.03
            neg_energy_boundT0 = -0.05
        elif Cg == 10:
            pos_energy_boundT0 = -0.01
            neg_energy_boundT0 = -0.11
        elif Cg == 5:
            pos_energy_boundT0 = -0.02
            neg_energy_boundT0 = -0.18
        elif Cg == 2:
            pos_energy_boundT0 = -0.12
            neg_energy_boundT0 = -0.37
        else:
            raise ValueError("what")
    else:
        raise ValueError("Cg must be in Cg_list")

    # EXPERIMENT PARAMETERS
    loop_count = max(num_workers, 1000)
    repetition = 0  # int : m -> the first gradient to check will be dT=(m+1)Tstd

    ######## CHANGABLES ###############
    last_repetition_to_do = 501  # int : n -> the last repetition has dT = n*Tstd
    repetition_list = list(range(first_rep, last_rep, jumps))
    T0_unitless = 0.001
    gap_ratio = 0
    mean_Rg = 100
    stdR = 2
    sig = 0.05
    ###################################

    # MESSAGES
    print(f"############# MAIN PARAMETERS ##################")
    print(f"flip = {flip}", flush=True)
    print(f"loop max: {loop_count}", flush=True)
    print(f"gap_ratio = {gap_ratio}", flush=True)
    print(f"Cg = {Cg}")
    print(f"stdR = {stdR}")
    print(f"sig = {sig}")
    print(f"repeating for dT=n*Tstd, n = {repetition_list}", flush=True)
    print(f"############# INITZIALIZING GRID ##################")
    if loop_count != 1:
        plot_ongoing_voltage_map = False

    null_path_name = (import_export.export_path /
                      f"64bit_table_triplets_T0_e{round(math.log10(T0_unitless))}_Cg{mean_Cg}.npz")

    # choose a specific run
    run_to_get_init_from = "20260606_22h05m04s"  # sig = 0.5, stdR=0.9 "20251207_17h43m26s" ; sig = 0.5, stdR = 4.8 "20260605_19h36m00s" ; sig = 0.05, stdR =2 "20260606_22h05m04s"
    results_dir_of_past_run = Path(__file__).parent.parent / f"results_{run_to_get_init_from}"
    infile = Path(results_dir_of_past_run / f"{run_to_get_init_from}.json")
    if infile.exists():
        json_txt = infile.read_text()
        raw_fields = orjson.loads(json_txt)
        # recreate old init state
        init_str = ExperimentInitialState(**raw_fields)
        init = F.fix_types(init_str, loop_count)

        # if old init has inappropriate variables
        init = F.swap_in_init("flip", flip, init)
        if init.T0 != T0_unitless:
            print("T0 is different in reference file or Temperature units != 1. switching.")
            init = F.swap_in_init("T0", T0_unitless, init)
            print(f"T0 is now {init.T0}")

        # different Cg or Rg
        if init.Cg[0] != mean_Cg or init.Rg[0] != mean_Rg:
            print(f"Mismatch in Cg or Rg from past run. Updating physics matrices...", flush=True)
            init = update_init_Cg_Rg(init, mean_Cg, mean_Rg)
            print(f"Successfully updated Cg to {init.Cg[0]} and Rg to {init.Rg[0]}. New Ec = {init.Ec}", flush=True)

        print(f"success, starting run for {run_to_get_init_from}", flush=True)

    else:
        # create new initial state
        init = prepare_initial_state(loop_count=loop_count, unitless_T0=T0_unitless, flip=flip, periodic_y=periodic_y,
                                     Cg_C_ratio=mean_Cg, Rg_R_ratio=mean_Rg, stdR_R_ratio=stdR, sigC_C_ratio=sig)
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
    print("############# RUN VIRTUAL EXPERIMENT ##################", flush=True)

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
                # Check that columns 1 and 2 actually exist and aren't empty
                if len(row) >= 3 and row[1].strip() != "" and row[2].strip() != "":
                    rep_idx = int(row[0])
                    bounds_dict[rep_idx] = {
                        "neg": float(row[1]),
                        "pos": float(row[2])
                    }
                else:
                    # Skip rows like "21,,," where bounds are missing
                    continue

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
            prepare_table_triplets_file_list=[EXPORT_PATH /
                                              f"64bit_table_triplets_Tstd{n}_20_Cg_{Cg}.npz" for n in range(501)],
            csv_table_path=MP_COMPUTE_PATH / f"table_Cg{Cg}.csv",
            results_dir_path=RESULTS_DIR_PATH
        ),
        run_name=run_name_flat,
        mean_Cg=Cg,
        first_rep=x
    )
