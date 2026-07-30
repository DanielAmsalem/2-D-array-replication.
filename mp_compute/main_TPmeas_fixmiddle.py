import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
num_workers = min(total_cpus - 5, 80)
print(f"worker number set to {num_workers} ; for {total_cpus} cpus", flush=True)

from concurrent.futures import ProcessPoolExecutor, as_completed
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
    prepare_table_triplets_gapped,
    prepare_table_triplets_NIS,
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
import pickle
from plot_graph_from_csv import plot_graph_from_csv

####### slurm parameter parsing from job name ######
job_name = os.environ.get('SLURM_JOB_NAME', 'TPmeas1_11_4_Tmid15_4_Cg10_D2_0')
pattern = r"(Reverse_?)?TPmeas(\d+)_(\d+)_(\d+)_Tmid(\d+)_(\d+)_Cg(\d+)_D(\d+)_(\d+)"
match = re.search(pattern, job_name)

if match:
    # match.group(1) will be 'Reverse' or 'Reverse_' if it exists, otherwise None
    is_reverse = match.group(1) is not None
    x = int(match.group(2))
    last_rep = int(match.group(3))
    jumps = int(match.group(4))
    Tmid_units = int(match.group(5))
    Tmid_pastdigit = int(match.group(6))
    Tmid = Tmid_units + Tmid_pastdigit / (10 ** len(str(Tmid_pastdigit)))
    Cg = int(match.group(7))
    gap_int = int(match.group(8))
    gap_tenth = int(match.group(9))
    gap_ratio = gap_int + gap_tenth / 10

    repetition_list = list(range(x, last_rep, jumps))
    print(f"Parsed from Job Name '{job_name}': flip={is_reverse}, repetition_list={repetition_list}, Cg={Cg}, "
          f"Tmid={Tmid}, D={gap_ratio}", flush=True)

else:
    raise NameError(f"Job Name is improperly formatted : {job_name}")

Cg_list = [2, 10]
Cg_list_gapped = [10]
gap_list = [2]


##############################################################


def main(import_export: IMPORT_EXPORT, run_name, mean_Cg, first_rep, is_resumed_shadow=False) -> None:
    # Set up dedicated checkpoint directory
    checkpoint_dir = import_export.results_dir_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # RUN TYPE
    flip = is_reverse
    first_run = False
    if first_rep == 0:
        first_run = True
        first_rep = 1
    rep_json = True
    periodic_y = True  # periodic boundary conditions in y-axis
    plot_ongoing_voltage_map = False
    V_capture = 4

    # VERIFY GAP RATIO AND CG
    if gap_ratio > 1e-3:
        if (Cg not in Cg_list_gapped) or (gap_ratio not in gap_list):
            raise ValueError(f"Cg must be in Cg_list_gapped, Cg = {Cg} ; "
                             f"gap ratio must be between in gap_list, D = {gap_ratio}")
    else:
        if Cg not in Cg_list:
            raise ValueError(f"Cg must be in Cg_list, Cg = {Cg}")

    # FIXED PARAMETERS (Read directly from CSV for rep=0 to accurately capture T_mid logic boundaries)
    pos_energy_boundT0 = None
    neg_energy_boundT0 = None

    with open(import_export.csv_table_path, encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[1].strip() != "" and row[2].strip() != "":
                try:
                    rep_idx = int(row[0].strip())
                except ValueError:
                    continue

                if rep_idx == 0:
                    pos_energy_boundT0 = float(row[2].strip())
                    neg_energy_boundT0 = float(row[1].strip())
                    print(f"T0 pos : {pos_energy_boundT0}, T0 neg : {neg_energy_boundT0}", flush=True)
                    break

    # VERIFICATION CHECK
    if pos_energy_boundT0 is None or neg_energy_boundT0 is None:
        raise ValueError(
            f"Failed to find n=0 bounds in the CSV table located at {import_export.csv_table_path} check CSV contents")

    # EXPERIMENT PARAMETERS
    loop_count = max(num_workers, 960)
    repetition = 0  # int : m -> the first gradient to check will be dT=(m+1)Tstd

    ######## CHANGABLES ###############
    last_repetition_to_do = 501  # int : n -> the last repetition has dT = n*Tstd
    repetition_list_shadow = list(range(first_rep, last_rep, jumps))
    if last_rep == 20:
        repetition_list_shadow += [20]
    T0_unitless = 0.001
    mean_Rg = 100
    stdR = 2
    sig = 0.05
    ###################################
    ########### FIX MIDDLE ############
    constT = Tmid
    ###################################

    # MESSAGES
    print(f"############# MAIN PARAMETERS ##################")
    print(f"flip = {flip}", flush=True)
    print(f"loop max: {loop_count}", flush=True)
    print(f"gap_ratio = {gap_ratio}", flush=True)
    print(f"Cg = {Cg}")
    print(f"stdR = {stdR}")
    print(f"sig = {sig}")
    print(f"repeating for dT=n*max_std/20, n = {repetition_list_shadow}", flush=True)
    print(f"############# INITZIALIZING GRID ##################")
    if loop_count != 1:
        plot_ongoing_voltage_map = False

    # base null path for new constT
    constT_str = str(constT).replace('.', '_')
    if gap_ratio > 1e-3:
        null_path_name = (import_export.export_path /
                          f"64bit_GAP{gap_int}_{gap_tenth}_table_triplets_"
                          f"Tmid_{constT_str}_e{round(math.log10(T0_unitless))}_Cg{mean_Cg}.npz")

        # Define the N-I-S Filepath
        nis_null_path_name = (import_export.export_path /
                              f"64bit_GAP{gap_int}_{gap_tenth}_NIS_table_triplets_"
                              f"Tmid_{constT_str}_e{round(math.log10(T0_unitless))}_Cg{mean_Cg}.npz")
    else:
        null_path_name = (import_export.export_path /
                          f"64bit_table_triplets_Tmid_{constT_str}_e{round(math.log10(T0_unitless))}_Cg{mean_Cg}.npz")
        nis_null_path_name = null_path_name

    # choose a specific run based on whether we are resuming or starting fresh
    if is_resumed_shadow:
        run_to_get_init_from = run_name
        results_dir_of_past_run = import_export.results_dir_path
    else:
        # sig = 0.5, stdR=0.9 "20251207_17h43m26s" ; sig = 0.5, stdR = 4.8 "20260605_19h36m00s" ;
        # sig = 0.05, stdR =2 "20260606_22h05m04s"
        run_to_get_init_from = "20260606_22h05m04s"
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
        if gap_ratio > 1e-3:
            if init.resolution != 1e-4:
                init = F.swap_in_init("resolution", 1e-4, init)
                print(f"set resolution times 100 res=1e-4", flush=True)
        if init.T0 != T0_unitless:
            print("T0 is different in reference file or Temperature units != 1. switching.")
            init = F.swap_in_init("T0", T0_unitless, init)
            print(f"T0 is now {init.T0}")

        # different Cg or Rg
        if init.Cg[0] != mean_Cg or init.Rg[0] != mean_Rg:
            print(f"Mismatch in Cg or Rg from past run. Updating physics matrices...", flush=True)
            init = update_init_Cg_Rg(init, mean_Cg, mean_Rg)
            print(f"Successfully updated Cg to {init.Cg[0]} and Rg to {init.Rg[0]}. New Ec = {init.Ec}", flush=True)

        # Check if ANY element in the matrix is below the threshold
        if np.any(init.R_t_ij < 0.1):
            min_Rt = np.min(init.R_t_ij)
            shift_amount = 0.1 - min_Rt
            print(f"min Rt = {min_Rt:.4f}. Shifting entire array by +{shift_amount:.4f}...", flush=True)
            R_t_ij_shifted = init.R_t_ij + shift_amount
            init = F.swap_in_init("R_t_ij", R_t_ij_shifted, init)

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

    if not is_resumed_shadow:
        Rx, Ry = curve_plotter.extract_nn_resistances(init.R_t_ij, init.row_num, init.R_t_i,
                                                      near_left=init.near_left,
                                                      near_right=init.near_right,
                                                      periodic_y=periodic_y, )
        print("created resistance maps, now saving plots...", flush=True)
        curve_plotter.plot_resistance_maps(Rx, Ry, n=init.row_num,
                                           results_path=import_export.results_dir_path, show=False)
        print("plotting capacitance map...", flush=True)
        curve_plotter.plot_capacitance_map(init.C_inv, n=init.row_num,
                                           results_path=import_export.results_dir_path, show=False,
                                           periodic_y=periodic_y)

    # =========================================================================
    # 1. S-I-S TABLE GENERATION
    # =========================================================================
    if not validate_table_triplets_file(null_path_name, init, [init.T0 * constT]) and first_run:
        print("S-I-S Table not found or invalid. Generating...", flush=True)
        if gap_ratio > 1e-3:
            table_triplets = prepare_table_triplets_gapped(init, [init.T0 * constT],
                                                           pos_energy_bound=pos_energy_boundT0,
                                                           neg_energy_bound=neg_energy_boundT0,
                                                           max_workers=num_workers,
                                                           gap_ratio=gap_ratio,
                                                           midfix=True)
        else:
            table_triplets = prepare_table_triplets(init, [init.T0 * constT],
                                                    pos_energy_bound=pos_energy_boundT0,
                                                    neg_energy_bound=neg_energy_boundT0,
                                                    max_workers=num_workers)
        output_table_triplets(table_triplets, null_path_name)

        table_val = table_triplets[:, 0]
        table_prob = table_triplets[:, 1]
        table_T = [init.T0 * constT]
    elif first_run:
        print("Valid S-I-S Table found. Loading...", flush=True)
        table_triplets = np.load(null_path_name.as_posix())
        table_val = table_triplets["val"]
        table_prob = table_triplets["prob"]
        table_T = np.unique(table_triplets["temp"]).tolist()

    # =========================================================================
    # 2. N-I-S TABLE GENERATION
    # =========================================================================
    if not validate_table_triplets_file(nis_null_path_name, init,
                                        [init.T0 * constT]) and first_run and gap_ratio > 1e-3:
        print("N-I-S Table not found or invalid. Generating...", flush=True)
        nis_table_triplets_raw = prepare_table_triplets_NIS(init, [init.T0 * constT],
                                                            pos_energy_bound=pos_energy_boundT0,
                                                            neg_energy_bound=neg_energy_boundT0,
                                                            max_workers=num_workers,
                                                            gap_ratio=gap_ratio,
                                                            midfix=True)
        output_table_triplets(nis_table_triplets_raw, nis_null_path_name)
        nis_table_val = nis_table_triplets_raw[:, 0]
        nis_table_prob = nis_table_triplets_raw[:, 1]
    elif first_run and gap_ratio > 1e-3:
        print("Valid N-I-S Table found. Loading...", flush=True)
        nis_table_triplets = np.load(nis_null_path_name.as_posix())
        nis_table_val = nis_table_triplets["val"]
        nis_table_prob = nis_table_triplets["prob"]
    elif first_run:
        nis_table_val = None
        nis_table_prob = None
    # RUN PARAMETERS
    V_diff = 4
    steps = 100
    Vleft = np.linspace(init.Vright * init.Volts, (init.Vright + V_diff) * init.Volts, num=steps)
    V_capture_idx = np.searchsorted(Vleft, V_capture, side='left')
    V_capture = float(Vleft[V_capture_idx])
    V_doubled = np.concatenate([Vleft, Vleft[-2::-1]])
    cycles = len(V_doubled)
    T = [init.T0 * constT] * init.row_num
    print("############# ISLAND APPROPRIATE ERROR & GAPS ##################", flush=True)
    Delta_0 = gap_ratio * init.Ec
    gap_array = F.exact_bcs_gap(T, Delta_0)
    expected_err = F.calc_expected_dist_std(T, init.T0, gap_array, init.R_t_ij, init.Ec)
    print(f"expected_err : {expected_err}", flush=True)
    print(f"gaps for each island : {gap_array}", flush=True)

    print("############# RUN VIRTUAL EXPERIMENT ##################", flush=True)

    if first_run:
        marker_file = import_export.results_dir_path / ".completed_rep0"
        if marker_file.exists():
            print("Repetition 0 already fully completed in previous run. Skipping execution block.", flush=True)
        else:
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
                    nis_table_val=nis_table_val,
                    nis_table_prob=nis_table_prob,
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
                    gap_array=gap_array
                )

                # --- CHECKPOINT LOADING / PARTIAL SUBMISSION ---
                results: list[SteadyStateResult] = [None] * init.loop_count
                futures = {}

                for i in range(init.loop_count):
                    ckpt_path = checkpoint_dir / f"ckpt_rep0_idx{i}.pkl"
                    if ckpt_path.exists():
                        try:
                            with open(ckpt_path, "rb") as f:
                                results[i] = pickle.load(f)
                        except Exception as e:
                            print(f"Warning: Failed to load {ckpt_path}, recomputing. Error: {e}")
                            ckpt_path.unlink(missing_ok=True)

                    if results[i] is None:
                        futures[executor.submit(loaded_state_function, i)] = i

                completed_count = init.loop_count - len(futures)
                if completed_count > 0:
                    print(f"Checkpoint Resume: {completed_count} workers pre-loaded. {len(futures)} submitted.",
                          flush=True)

                # --- FAIL-FAST SUBMISSION WITH INCREMENTAL SAVING ---
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        res = future.result()
                        results[idx] = res

                        # Save checkpoint instantly
                        ckpt_path = checkpoint_dir / f"ckpt_rep0_idx{idx}.pkl"
                        with open(ckpt_path, "wb") as f:
                            pickle.dump(res, f)

                    except Exception as e:
                        print(f"\nCRITICAL ERROR: Worker {idx} crashed instantly during first run!", flush=True)
                        for f_cancel in futures:
                            f_cancel.cancel()
                        raise

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
                                       gap_ratio=gap_ratio)

            # SUCCESS: Mark as complete and clean up .pkl files
            marker_file.touch()
            for f in checkpoint_dir.glob("ckpt_rep0_idx*.pkl"):
                f.unlink(missing_ok=True)

    else:
        print("skipped first run")

    ### get bounds for dE for each temp from csv table
    bounds_dict = {}
    with open(import_export.csv_table_path, encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[1].strip() != "" and row[2].strip() != "":
                try:
                    rep_idx = int(row[0].strip())
                    bounds_dict[rep_idx] = {
                        "neg": float(row[1].strip()),
                        "pos": float(row[2].strip())
                    }
                except ValueError:
                    continue

    ### run thermopower until I(V)<0
    current_at_V0 = True

    while current_at_V0:
        repetition += 1
        if repetition > last_repetition_to_do:
            print("Tstd>T0, finished all runs for Tstd<=T0", flush=True)
            current_at_V0 = False
            continue
        if not int(repetition) in repetition_list_shadow:
            print(f"repetition {repetition} was skipped")
            continue

        if repetition not in bounds_dict:
            print(f"repetition {repetition} missing from bounds table, skipped")
            continue

        marker_file = import_export.results_dir_path / f".completed_rep{repetition}"
        if marker_file.exists():
            print(f"Repetition {repetition} already completely finished. Skipping.", flush=True)
            continue

        # NEW TEMPERATURE PROFILE LOGIC: Fixed Middle
        T_mid = constT * init.T0
        # the maximum difference in temps between islands is when Tleft = T0:
        max_std = 2 * (T_mid - init.T0) / (init.row_num - init.row_num % 2)
        T_std = repetition * max_std / 20
        first_site_T = T_mid - ((init.row_num - init.row_num % 2) / 2) * T_std
        T_list_to_compute = [first_site_T + i * T_std for i in range(init.row_num)]
        print(f"T list : {T_list_to_compute}", flush=True)

        ### check if there is valid table for new dT
        # --- PATH DEFINITIONS ---
        bulk_path = import_export.prepare_table_triplets_file_list[repetition]
        if gap_ratio > 1e-3:
            nis_path_str = bulk_path.as_posix().replace("_table_triplets_", "_NIS_table_triplets_")
            nis_path = Path(nis_path_str)
        else:
            nis_path = bulk_path

        # --- VALIDATION CHECKS ---
        valid_bulk = validate_table_triplets_file(bulk_path, init, np.array(T_list_to_compute))
        valid_nis = True
        if gap_ratio > 1e-3:
            # For N-I-S, the array only expects to see the boundaries
            edge_T_list = [T_list_to_compute[0]] if T_list_to_compute[0] == T_list_to_compute[-1] else [
                T_list_to_compute[0], T_list_to_compute[-1]]
            valid_nis = validate_table_triplets_file(nis_path, init, np.array(edge_T_list))

        # --- THE 'SKIP' LOGIC ---
        if not (valid_bulk and valid_nis):
            warnings.warn(
                f"Missing or invalid tables (BULK valid: {valid_bulk}, EDGE valid: {valid_nis}), skipped rep {T_std}")
            continue

        # --- SAFE LOADING ---
        table_triplets = np.load(bulk_path.as_posix())
        table_val = table_triplets["val"]
        table_prob = table_triplets["prob"]
        table_T = np.unique(table_triplets["temp"]).tolist()

        if gap_ratio > 1e-3:
            nis_table_triplets = np.load(nis_path.as_posix())
            nis_table_val = nis_table_triplets["val"]
            nis_table_prob = nis_table_triplets["prob"]
        else:
            nis_table_val = None
            nis_table_prob = None

        ### run repetition for new dT
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            t0 = time.time()
            T = T_list_to_compute
            if flip:
                T = np.flip(T_list_to_compute)

            print("############# ISLAND APPROPRIATE ERROR & GAPS ##################", flush=True)
            gap_array = F.exact_bcs_gap(T, Delta_0)
            expected_err = F.calc_expected_dist_std(T, init.T0, gap_array, init.R_t_ij, init.Ec)
            print(f"expected_err : {expected_err}", flush=True)
            print(f"gaps for each island : {gap_array}", flush=True)

            # ---> STEP 3.3: Splice into KMC Arguments
            loaded_state_function = partial(
                Get_Steady_State,
                init=init,
                V_cycle=V_doubled,
                cycles=cycles,
                table_val=table_val,
                table_prob=table_prob,
                nis_table_val=nis_table_val,
                nis_table_prob=nis_table_prob,
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
                gap_array=gap_array
            )

            # --- CHECKPOINT LOADING / PARTIAL SUBMISSION ---
            results: list[SteadyStateResult] = [None] * init.loop_count
            futures = {}

            for i in range(init.loop_count):
                ckpt_path = checkpoint_dir / f"ckpt_rep{repetition}_idx{i}.pkl"
                if ckpt_path.exists():
                    try:
                        with open(ckpt_path, "rb") as f:
                            results[i] = pickle.load(f)
                    except Exception as e:
                        print(f"Warning: Failed to load {ckpt_path}, recomputing. Error: {e}")
                        ckpt_path.unlink(missing_ok=True)

                if results[i] is None:
                    futures[executor.submit(loaded_state_function, i)] = i

            completed_count = init.loop_count - len(futures)
            if completed_count > 0:
                print(
                    f"Checkpoint Resume: Rep {repetition} pre-loaded {completed_count} workers. {len(futures)} submitted.",
                    flush=True)

            # --- FAIL-FAST SUBMISSION WITH INCREMENTAL SAVING ---
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    res = future.result()
                    results[idx] = res

                    # Save checkpoint instantly
                    ckpt_path = checkpoint_dir / f"ckpt_rep{repetition}_idx{idx}.pkl"
                    with open(ckpt_path, "wb") as f:
                        pickle.dump(res, f)

                except Exception as e:
                    print(f"\nCRITICAL ERROR: Worker {idx} crashed instantly during rep {repetition}!", flush=True)
                    for f_cancel in futures:
                        f_cancel.cancel()
                    raise

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

        # SUCCESS: Mark as complete and clean up .pkl files explicitly in the subfolder
        marker_file.touch()
        for f in checkpoint_dir.glob(f"ckpt_rep{repetition}_idx*.pkl"):
            f.unlink(missing_ok=True)

    ### plot all new csvs
    print(f"plotting all new csv in {import_export.results_dir_path}")
    plot_graph_from_csv(run_names=[f"results_{run_name}"], directory=import_export.results_dir_path)


def find_resume_directory(base_dir: Path, current_job_name: str) -> Path:
    """
    Scans the base directory for 'results_*' folders.
    Looks for checkpoint_meta.json to guarantee a 1:1 match with SLURM_JOB_NAME.
    """
    # Sort by newest first just in case
    dirs = sorted([d for d in base_dir.glob("results_*") if d.is_dir()],
                  key=lambda x: x.stat().st_mtime, reverse=True)

    for p in dirs:
        meta_file = p / "checkpoint_meta.json"
        if meta_file.exists():
            try:
                meta = orjson.loads(meta_file.read_text())
                if meta.get("slurm_job_name") == current_job_name:
                    return p
            except Exception:
                pass
    return None


if __name__ == "__main__":
    EXPORT_PATH = Path(__file__).parent.parent / "export"
    MP_COMPUTE_PATH = Path(__file__).parent.parent / "mp_compute"
    BASE_RESULTS_DIR = Path(__file__).parent.parent

    # 1. Check if this is an interrupted run we can hijack
    resume_dir = find_resume_directory(BASE_RESULTS_DIR, job_name)

    if resume_dir:
        RESULTS_DIR_PATH = resume_dir
        run_name_flat = resume_dir.name.replace("results_", "")
        print(f"RESUMING existing run at {RESULTS_DIR_PATH} (Run Name: {run_name_flat})", flush=True)
        is_resumed = True
    else:
        date_ = datetime.datetime.now()
        run_name_flat = date_.strftime("%Y%m%d_%Hh%Mm%Ss")
        RESULTS_DIR_PATH = BASE_RESULTS_DIR / f"results_{run_name_flat}"
        RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
        print(f"Created NEW results directory at {RESULTS_DIR_PATH}")
        is_resumed = False

        # Save exact metadata to allow future resumption, strictly tracking Tmid and D parameters
        meta_data = {
            "slurm_job_name": job_name,
            "created_at": run_name_flat,
            "Cg": Cg,
            "Tmid_units": Tmid_units,
            "Tmid_pastdigit": Tmid_pastdigit,
            "gap_ratio": gap_ratio,
            "is_reverse": is_reverse,
            "x": x,
            "last_rep": last_rep,
            "jumps": jumps
        }
        meta_file = RESULTS_DIR_PATH / "checkpoint_meta.json"
        meta_file.write_text(orjson.dumps(meta_data).decode("utf-8"))

    if gap_ratio > 1e-3:
        tables_list = [EXPORT_PATH /
                       f"64bit_GAP{gap_int}_{gap_tenth}_table_triplets_Tmid{Tmid_units}_{Tmid_pastdigit}_Tstd{n}_20_Cg_{Cg}.npz"
                       for n in range(501)]
        csv_table_path = MP_COMPUTE_PATH / f"gapped_table_Tmid{Tmid_units}_{Tmid_pastdigit}_Cg{Cg}_D{gap_int}_{gap_tenth}.csv"
    else:
        tables_list = [EXPORT_PATH / f"64bit_table_triplets_Tmid{Tmid_units}_{Tmid_pastdigit}_Tstd{n}_20_Cg_{Cg}.npz"
                       for n in range(501)]
        csv_table_path = MP_COMPUTE_PATH / f"table_Tmid{Tmid_units}_{Tmid_pastdigit}_Cg{Cg}.csv"

    main(
        import_export=IMPORT_EXPORT(
            plot_results=True,
            export_path=EXPORT_PATH,
            prepare_table_triplets_file_list=tables_list,
            csv_table_path=csv_table_path,
            results_dir_path=RESULTS_DIR_PATH
        ),
        run_name=run_name_flat,
        mean_Cg=Cg,
        first_rep=x,
        is_resumed_shadow=is_resumed
    )