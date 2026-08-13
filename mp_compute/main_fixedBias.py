import os
import multiprocessing

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
num_workers = min(total_cpus - 5, 80)
print(f"worker number set to {num_workers} ; for {total_cpus} cpus", flush=True)

# SLURM Job Array Parsing
task_id_str = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
TASK_ID = int(task_id_str)
LOOPS_PER_TASK = 80
START_LOOP_IDX = TASK_ID * LOOPS_PER_TASK
END_LOOP_IDX = START_LOOP_IDX + LOOPS_PER_TASK
print(f"Executing Task ID: {TASK_ID} | Loop range: {START_LOOP_IDX} to {END_LOOP_IDX - 1}", flush=True)

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
import datetime
import warnings
import Functions as F
import numpy as np
from define_objects import IMPORT_EXPORT, SteadyStateResult, ExperimentInitialState
# Importing the new Get_Steady_State function
from gamma_functions import Get_Steady_State_fixed_bias
from preparation import (
    prepare_initial_state,
    update_init_Cg_Rg
)
from dataclasses import asdict
import orjson
import csv
import math
import time
import re
import pickle

####### slurm parameter parsing from job name ######
job_name = os.environ.get('SLURM_JOB_NAME', 'TPmeas1_11_4_Cg2')
pattern = r"(Reverse_?)?fixedBias_Cg(\d+)"
match = re.search(pattern, job_name)

if match:
    is_reverse = match.group(1) is not None
    Cg = int(match.group(2))
    gap_int = 0
    gap_tenth = 0
    gap_ratio = 0

    print(f"Parsed from Job Name '{job_name}': flip={is_reverse}, Cg={Cg}, D={gap_ratio}", flush=True)
else:
    raise NameError(f"Job Name is improperly formatted : {job_name}")

Cg_list = [2, 10]
Cg_list_gapped = [10]
gap_list = [2, 0.2]


##############################################################

def main(import_export: IMPORT_EXPORT, run_name, mean_Cg, is_resumed=False) -> None:
    # Set up dedicated checkpoint directory
    checkpoint_dir = import_export.results_dir_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # RUN TYPE
    flip = is_reverse
    rep_json = True if TASK_ID == 0 else False  # Only task 0 writes the init json
    periodic_y = True

    # NEW FIXED EXPERIMENT PARAMETERS
    FIXED_VOLTAGE = 0.0  # V=0 or any constant V you require for this sweep
    loop_count = 960  # Fixed at 12 tasks * 80 loops
    T0_unitless = 0.001
    mean_Rg = 100
    stdR = 2
    sig = 0.05

    # HARDCODED TEMPERATURE SWEEP TARGETS
    n_list_master = (list(range(1, 20)) +
                     list(range(20, 40, 2)) +
                     list(range(40, 80, 4)) +
                     list(range(80, 200, 16)) +
                     list(range(200, 500, 30)) +
                     list(range(500, 2001, 50))
                     )

    # MESSAGES
    print(f"############# MAIN PARAMETERS ##################")
    print(f"flip = {flip}", flush=True)
    print(f"Total Loops across Array: {loop_count}", flush=True)
    print(f"gap_ratio = {gap_ratio}", flush=True)
    print(f"Cg = {Cg}")
    print(f"stdR = {stdR}")
    print(f"sig = {sig}")
    print(f"Fixed Voltage Bias = {FIXED_VOLTAGE}", flush=True)
    print(f"Intended Gradients: min(n)={min(n_list_master)}, max(n)={max(n_list_master)}", flush=True)
    print(f"############# INITIALIZING GRID ##################")

    if is_resumed:
        run_to_get_init_from = run_name
        results_dir_of_past_run = import_export.results_dir_path
    else:
        run_to_get_init_from = "20260606_22h05m04s"
        results_dir_of_past_run = Path(__file__).parent.parent / f"results_{run_to_get_init_from}"

    infile = Path(results_dir_of_past_run / f"{run_to_get_init_from}.json")
    if infile.exists():
        json_txt = infile.read_text()
        raw_fields = orjson.loads(json_txt)
        init_str = ExperimentInitialState(**raw_fields)
        init = F.fix_types(init_str, loop_count)

        init = F.swap_in_init("flip", flip, init)
        if gap_ratio > 1e-3 and init.resolution != 1e-4:
            init = F.swap_in_init("resolution", 1e-4, init)

        if init.T0 != T0_unitless:
            init = F.swap_in_init("T0", T0_unitless, init)

        if init.Cg[0] != mean_Cg or init.Rg[0] != mean_Rg:
            init = update_init_Cg_Rg(init, mean_Cg, mean_Rg)

        if np.any(init.R_t_ij < 0.1):
            min_Rt = np.min(init.R_t_ij)
            shift_amount = 0.1 - min_Rt
            R_t_ij_shifted = init.R_t_ij + shift_amount
            init = F.swap_in_init("R_t_ij", R_t_ij_shifted, init)

        print(f"Success: Initialized state from {run_to_get_init_from}", flush=True)
    else:
        init = prepare_initial_state(loop_count=loop_count, unitless_T0=T0_unitless, flip=flip, periodic_y=periodic_y,
                                     Cg_C_ratio=mean_Cg, Rg_R_ratio=mean_Rg, stdR_R_ratio=stdR, sigC_C_ratio=sig)
        print("CREATED NEW INIT FILE", flush=True)

    # Task 0 writes the JSON config for the entire array
    if rep_json:
        outfile = Path(import_export.results_dir_path / f"{run_name}.json")
        raw_fields = asdict(init)
        serialized_init_data = orjson.dumps(raw_fields, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")
        outfile.write_text(serialized_init_data)
        print("STORED INIT IN JSON", flush=True)

    print("############# PRE-LOADING METADATA (NO TABLES) ##################", flush=True)
    Delta_0 = gap_ratio * init.Ec

    # Parse CSV Bounds Once
    bounds_dict = {}
    with open(import_export.csv_table_path) as f:
        for row in csv.reader(f):
            if row and row[0].strip().lstrip('-').isdigit():
                if len(row) >= 3 and row[1].strip() != "" and row[2].strip() != "":
                    rep_idx = int(row[0])
                    bounds_dict[rep_idx] = {"neg": float(row[1]), "pos": float(row[2])}

    # Build Memory-Resident Data Dictionary for all n
    simulation_data_dict = {}
    valid_n_list = []

    for n in n_list_master:
        if n not in bounds_dict:
            print(f"Warning: n={n} missing from bounds table. Skipping.")
            continue

        # Calculate strict T profile for this n
        T_std = n * init.T0 / 20
        T_list = [init.T0 + i * T_std for i in range(init.row_num)]
        if flip:
            T_list = np.flip(T_list).tolist()

        table_path = import_export.prepare_table_triplets_file_list[n]
        if not table_path.exists():
            print(f"Warning: Table file missing for n={n}. Skipping.")
            continue

        # PRE-CALCULATE PHYSICS, BUT DO NOT LOAD THE .NPZ
        gap_array = F.exact_bcs_gap(T_list, Delta_0)
        expected_err = F.calc_expected_dist_std(T_list, init.T0, gap_array, init.R_t_ij, init.Ec)

        # Populate Dictionary with Path instead of raw tables
        simulation_data_dict[n] = {
            "T": T_list,
            "T_std": T_std,
            "gap_array": gap_array,
            "expected_error": expected_err,
            "table_path": table_path.as_posix(),  # Pass the string path!
            "pos_energy_bound": bounds_dict[n]["pos"],
            "neg_energy_bound": bounds_dict[n]["neg"]
        }
        valid_n_list.append(n)

    print(f"Pre-loaded metadata for {len(valid_n_list)} temperature profiles.", flush=True)

    # 2. Print VERIFIED bounds
    if valid_n_list:
        print(f"---> Verified target gradients: min(n) = {min(valid_n_list)} | max(n) = {max(valid_n_list)}",
              flush=True)
    else:
        print("---> CRITICAL ERROR: No valid gradients found! Check your CSV bounds and .npz file paths.", flush=True)
        return

    print("############# RUNNING MULTIPROCESSING ARRAY ##################", flush=True)
    marker_file = import_export.results_dir_path / f".completed_task{TASK_ID}"
    if marker_file.exists():
        print(f"Task {TASK_ID} already completed. Exiting.", flush=True)
        return

    # CREATE THE I/O LOCK
    manager = multiprocessing.Manager()
    io_lock = manager.Lock()

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        print(f"Task {TASK_ID} running with {executor._max_workers} workers", flush=True)

        # Bind the unified parameters to our new fixed_bias steady state calculator
        loaded_state_function = partial(
            Get_Steady_State_fixed_bias,
            init=init,
            fixed_voltage=FIXED_VOLTAGE,
            valid_n_list=valid_n_list,
            sim_data=simulation_data_dict,
            io_lock=io_lock,  # <--- I/O LOCK ARGUMENT
            flip=flip,
            periodic_y=periodic_y,
            gap_ratio=gap_ratio
        )

        results = [None] * init.loop_count
        futures = {}

        # Safely cap bounds in case we requested fewer loops total
        safe_end_idx = min(END_LOOP_IDX, init.loop_count)

        for i in range(START_LOOP_IDX, safe_end_idx):
            ckpt_path = checkpoint_dir / f"ckpt_task{TASK_ID}_idx{i}.pkl"
            if ckpt_path.exists():
                try:
                    with open(ckpt_path, "rb") as f:
                        results[i] = pickle.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load {ckpt_path}, recomputing. Error: {e}")
                    ckpt_path.unlink(missing_ok=True)

            if results[i] is None:
                futures[executor.submit(loaded_state_function, i)] = i

        completed_count = (safe_end_idx - START_LOOP_IDX) - len(futures)
        if completed_count > 0:
            print(f"Checkpoint Resume: Task {TASK_ID} pre-loaded {completed_count} workers. {len(futures)} submitted.",
                  flush=True)

        for future in as_completed(futures):
            idx = futures[future]
            try:
                res = future.result()
                results[idx] = res

                # Save checkpoint instantly
                ckpt_path = checkpoint_dir / f"ckpt_task{TASK_ID}_idx{idx}.pkl"
                with open(ckpt_path, "wb") as f:
                    pickle.dump(res, f)

            except Exception as e:
                print(f"\nCRITICAL ERROR: Worker {idx} in Task {TASK_ID} crashed!", flush=True)
                for f_cancel in futures:
                    f_cancel.cancel()
                raise

        # ----------------- DATA AGGREGATION FOR THIS TASK -----------------
        valid_results = [res for res in results[START_LOOP_IDX:safe_end_idx] if res is not None]
        if valid_results:
            N_valid = len(valid_results)
            print(f"Task {TASK_ID} finished computing. Aggregating {N_valid} valid loops.", flush=True)

            # Stack all individual I_vec arrays into a 2D matrix of shape (N_valid, cycles)
            all_I_vecs = np.array([res.I_vec for res in valid_results])

            # Vectorized Mean: Calculate average across the 0th axis (columns/loops)
            avg_currents = np.mean(all_I_vecs, axis=0)

            # Vectorized Standard Deviation (ddof=1 gives unbiased sample variance)
            std_currents = np.std(all_I_vecs, axis=0, ddof=1)

            # Standard Error of the Mean (What you actually plot for error bars)
            err_currents = std_currents / np.sqrt(N_valid)

            # Output specific task CSV
            out_csv_path = import_export.results_dir_path / f"fixed_bias_task{TASK_ID}.csv"
            with open(out_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["n", "T_std", "delta_T", "I_avg", "I_err"])

                for idx, n in enumerate(valid_n_list):
                    T_std = simulation_data_dict[n]["T_std"]
                    delta_T = 0.001 + 0.006 * (n / 20)
                    writer.writerow([n, T_std, delta_T, avg_currents[idx], err_currents[idx]])

            print(f"Saved local task output to {out_csv_path.name}")

        # Output basic report for task
        report_path = import_export.results_dir_path / f"report_task{TASK_ID}.txt"
        with open(report_path, "w") as f:
            f.write(f"Task ID: {TASK_ID}\n")
            f.write(f"Run Time: {time.time() - t0} sec\n")

    # Mark as complete and clean up .pkl files
    marker_file.touch()
    for f in checkpoint_dir.glob(f"ckpt_task{TASK_ID}_idx*.pkl"):
        f.unlink(missing_ok=True)

    print(f"Task {TASK_ID} successfully completed all operations.", flush=True)


def find_resume_directory(base_dir: Path, current_job_name: str) -> Path:
    dirs = sorted([d for d in base_dir.glob("results_*") if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
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

    resume_dir = find_resume_directory(BASE_RESULTS_DIR, job_name)

    if resume_dir:
        RESULTS_DIR_PATH = resume_dir
        run_name_flat = resume_dir.name.replace("results_", "")
        if TASK_ID == 0:
            print(f"RESUMING existing run at {RESULTS_DIR_PATH} (Run Name: {run_name_flat})", flush=True)
        is_resumed = True
    else:
        # Prevent simultaneous directory creation from multiple array jobs causing race conditions
        date_ = datetime.datetime.now()
        run_name_flat = date_.strftime("%Y%m%d_%Hh%Mm%Ss")
        RESULTS_DIR_PATH = BASE_RESULTS_DIR / f"results_{run_name_flat}"

        try:
            RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=False)
            print(f"Created NEW results directory at {RESULTS_DIR_PATH}")
            is_resumed = False

            # Task 0 (or whichever reaches here first) saves metadata
            meta_data = {
                "slurm_job_name": job_name,
                "created_at": run_name_flat,
                "Cg": Cg,
                "gap_ratio": gap_ratio,
                "is_reverse": is_reverse
            }
            meta_file = RESULTS_DIR_PATH / "checkpoint_meta.json"
            meta_file.write_text(orjson.dumps(meta_data).decode("utf-8"))
        except FileExistsError:
            # Another array task beat this one to folder creation, treat as resumed
            is_resumed = True
            time.sleep(2)  # Give the creating task a moment to write checkpoint_meta.json

    if gap_ratio > 1e-3:
        tables_list = [EXPORT_PATH / f"64bit_GAP{gap_int}_{gap_tenth}_table_triplets_Tstd{n}_20_Cg_{Cg}.npz" for n in
                       range(501)]
        csv_table_path = MP_COMPUTE_PATH / f"gapped_table_Cg{Cg}_D{gap_int}_{gap_tenth}.csv"
    else:
        tables_list = [EXPORT_PATH / f"64bit_table_triplets_Tstd{n}_20_Cg_{Cg}.npz" for n in range(2001)]
        csv_table_path = MP_COMPUTE_PATH / f"table_Cg{Cg}.csv"

    main(
        IMPORT_EXPORT(
            plot_results=True,
            export_path=EXPORT_PATH,
            prepare_table_triplets_file_list=tables_list,
            csv_table_path=csv_table_path,
            results_dir_path=RESULTS_DIR_PATH
        ),
        run_name=run_name_flat,
        mean_Cg=Cg,
        is_resumed=is_resumed
    )
