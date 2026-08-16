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
from define_objects import IMPORT_EXPORT, ExperimentInitialState
# NOTE: We will assume a new return object and function in gamma_functions
from gamma_functions import Get_Steady_State_varyV
from preparation import (
    prepare_initial_state,
    validate_table_triplets_file,
    update_init_Cg_Rg
)
import curve_plotter
from dataclasses import asdict
import orjson
import csv
import time
import re
import pickle
from plot_graph_from_csv import plot_graph_from_csv

####### SLURM parameter parsing from job name ######
job_name = os.environ.get('SLURM_JOB_NAME', 'TPvaryV1_11_4_Cg2')
pattern = r"(Reverse_?)?TPvaryV(\d+)_(\d+)_(\d+)_Cg(\d+)_D(\d+)_(\d+)"
match = re.search(pattern, job_name)

if match:
    is_reverse = match.group(1) is not None
    x = int(match.group(2))
    last_rep = int(match.group(3))
    jumps = int(match.group(4))
    Cg = int(match.group(5))
    gap_int = int(match.group(6))
    gap_tenth = int(match.group(7))
    gap_ratio = gap_int + gap_tenth / 10

    repetition_list = list(range(x, last_rep, jumps))
    print(f"Parsed from Job Name '{job_name}': flip={is_reverse}, x={x}, last_rep={last_rep}, jumps={jumps}, Cg={Cg}",
          flush=True)
else:
    raise NameError(f"Job Name is improperly formatted : {job_name}")

Cg_list = [2, 5, 10, 20, 50]
Cg_list_gapped = [10]
gap_list = [2, 0.2]


##############################################################

def main(import_export: IMPORT_EXPORT, run_name, mean_Cg, first_rep, is_resumed=False) -> None:
    # Set up dedicated checkpoint directory
    checkpoint_dir = import_export.results_dir_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # RUN TYPE
    flip = is_reverse
    periodic_y = True
    plot_ongoing_voltage_map = False

    # EXPERIMENT PARAMETERS
    loop_count = max(num_workers, 960)
    repetition = 0
    last_repetition_to_do = 501
    repetition_list = list(range(first_rep, last_rep, jumps))
    T0_unitless = 0.001
    mean_Rg = 100
    stdR = 2
    sig = 0.05

    print(f"############# MAIN PARAMETERS ##################")
    print(f"flip = {flip}", flush=True)
    print(f"loop max: {loop_count}", flush=True)
    print(f"gap_ratio = {gap_ratio}", flush=True)
    print(f"Cg = {Cg}")
    print(f"repeating for dT=n*Tstd, n = {repetition_list}", flush=True)

    # ---------------------------------------------------------
    # STATE INITIALIZATION (Resuming or Creating New)
    # ---------------------------------------------------------
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
        if init.Cg[0] != mean_Cg or init.Rg[0] != mean_Rg:
            init = update_init_Cg_Rg(init, mean_Cg, mean_Rg)

        if np.any(init.R_t_ij < 0.1):
            min_Rt = np.min(init.R_t_ij)
            init = F.swap_in_init("R_t_ij", init.R_t_ij + (0.1 - min_Rt), init)
    else:
        init = prepare_initial_state(loop_count=loop_count, unitless_T0=T0_unitless, flip=flip, periodic_y=periodic_y,
                                     Cg_C_ratio=mean_Cg, Rg_R_ratio=mean_Rg, stdR_R_ratio=stdR, sigC_C_ratio=sig)

    # Output init to JSON
    outfile = Path(import_export.results_dir_path / f"{run_name}.json")
    outfile.write_text(orjson.dumps(asdict(init), option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8"))

    # ---------------------------------------------------------
    # DEFINE MACRO VOLTAGE SWEEP (Baseline Voltages)
    # ---------------------------------------------------------
    V_diff = 4
    steps = 100
    # Unlike TPmeas, we only need a one-way sweep of baseline voltages
    V_sweep = np.linspace(init.Vright * init.Volts, (init.Vright + V_diff) * init.Volts, num=steps)
    cycles = len(V_sweep)

    # Calculate uniform T0 baseline constants ONCE
    T_baseline = [init.T0] * init.row_num
    Delta_0 = gap_ratio * init.Ec
    gap_array_baseline = F.exact_bcs_gap(T_baseline, Delta_0)
    expected_err_baseline = F.calc_expected_dist_std(T_baseline, init.T0, gap_array_baseline, init.R_t_ij, init.Ec)

    print("############# RUN THERMOPOWER VARY-V EXPERIMENT ##################", flush=True)

    # ---------------------------------------------------------
    # MAIN EXPERIMENT LOOP (Iterating over dT gradients)
    # ---------------------------------------------------------
    increasing_T_gradient = True

    while increasing_T_gradient:
        repetition += 1
        if repetition > last_repetition_to_do:
            increasing_T_gradient = False
            continue
        if int(repetition) not in repetition_list:
            continue

        marker_file = import_export.results_dir_path / f".completed_rep{repetition}"
        if marker_file.exists():
            continue

        # 1. Define the specific temperature gradient for this run
        T_std = repetition * init.T0 / 20
        T_dT = [init.T0 + i * T_std for i in range(init.row_num)]
        if flip:
            T_dT = np.flip(T_dT)

        # 2. Pre-calculate Gradient Gaps and Errors ONCE (Problem 2 Solved)
        gap_array_dT = F.exact_bcs_gap(T_dT, Delta_0)
        expected_err_dT = F.calc_expected_dist_std(T_dT, init.T0, gap_array_dT, init.R_t_ij, init.Ec)

        # 3. Load the corresponding Integration Table (Problem 1 Solved)
        if not validate_table_triplets_file(import_export.prepare_table_triplets_file_list[repetition], init,
                                            np.array(T_dT)):
            warnings.warn(f"no validated table, skipped rep{T_std}")
            continue

        table_triplets = np.load(import_export.prepare_table_triplets_file_list[repetition].as_posix())
        table_val_dT = table_triplets["val"]
        table_prob_dT = table_triplets["prob"]
        table_T_dT = np.unique(table_triplets["temp"]).tolist()

        # 4. Dispatch to Physics Engine
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            t0 = time.time()

            # We pass BOTH the baseline and gradient parameters so the worker can seamlessly
            # toggle between step 0 and step 2 without recalculating matrices.
            loaded_state_function = partial(
                Get_Steady_State_varyV,
                init=init,
                V_sweep=V_sweep,
                cycles=cycles,
                table_val_dT=table_val_dT,
                table_prob_dT=table_prob_dT,
                table_T_dT=table_T_dT,
                flip=flip,
                T_baseline=T_baseline,
                T_dT=T_dT,
                expected_error_baseline=expected_err_baseline,
                expected_error_dT=expected_err_dT,
                periodic_y=periodic_y,
                plot_ongoing_voltage_map=plot_ongoing_voltage_map,
                gap_ratio=gap_ratio,
                gap_array_baseline=gap_array_baseline,
                gap_array_dT=gap_array_dT
            )

            # --- CHECKPOINT LOADING / PARTIAL SUBMISSION ---
            results = [None] * init.loop_count
            futures = {}

            for i in range(init.loop_count):
                ckpt_path = checkpoint_dir / f"ckpt_varyV_rep{repetition}_idx{i}.pkl"
                if ckpt_path.exists():
                    try:
                        with open(ckpt_path, "rb") as f:
                            results[i] = pickle.load(f)
                    except Exception:
                        ckpt_path.unlink(missing_ok=True)

                if results[i] is None:
                    futures[executor.submit(loaded_state_function, i)] = i

            # --- FAIL-FAST SUBMISSION WITH INCREMENTAL SAVING ---
            for future in as_completed(futures):
                idx = futures[future]
                res = future.result()
                results[idx] = res

                with open(checkpoint_dir / f"ckpt_varyV_rep{repetition}_idx{idx}.pkl", "wb") as f:
                    pickle.dump(res, f)

        # 5. Extract and Plot Data (Needs to be adapted for S(V) plotting in curve_plotter)
        # Instead of I(V), the results now contain DeltaV required at each V_baseline.
        curve_plotter.thermopower_curve_compute_and_save_csv(
            init=init,
            filename=run_name,
            results=results,
            V_sweep=V_sweep,
            repetition=repetition,
            results_path=import_export.results_dir_path
        )

        marker_file.touch()
        for f in checkpoint_dir.glob(f"ckpt_varyV_rep{repetition}_idx*.pkl"):
            f.unlink(missing_ok=True)

    print(f"plotting all new csv in {import_export.results_dir_path}")
    plot_graph_from_csv(run_names=[f"results_{run_name}"], directory=import_export.results_dir_path)


def find_resume_directory(base_dir: Path, current_job_name: str) -> Path:
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

    resume_dir = find_resume_directory(BASE_RESULTS_DIR, job_name)

    if resume_dir:
        RESULTS_DIR_PATH = resume_dir
        run_name_flat = resume_dir.name.replace("results_", "")
        print(f"RESUMING existing run at {RESULTS_DIR_PATH}", flush=True)
        is_resumed = True
    else:
        date_ = datetime.datetime.now()
        run_name_flat = date_.strftime("%Y%m%d_%Hh%Mm%Ss")
        RESULTS_DIR_PATH = BASE_RESULTS_DIR / f"results_{run_name_flat}"
        RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
        is_resumed = False

        meta_data = {
            "slurm_job_name": job_name,
            "created_at": run_name_flat,
            "Cg": Cg,
            "gap_ratio": gap_ratio,
        }
        (RESULTS_DIR_PATH / "checkpoint_meta.json").write_text(orjson.dumps(meta_data).decode("utf-8"))

    if gap_ratio > 1e-3:
        tables_list = [EXPORT_PATH / f"64bit_GAP{gap_int}_{gap_tenth}_table_triplets_Tstd{n}_20_Cg_{Cg}.npz" for n in
                       range(501)]
        csv_table_path = MP_COMPUTE_PATH / f"gapped_table_Cg{Cg}_D{gap_int}_{gap_tenth}.csv"
    else:
        tables_list = [EXPORT_PATH / f"64bit_table_triplets_Tstd{n}_20_Cg_{Cg}.npz" for n in range(501)]
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
        first_rep=x,
        is_resumed=is_resumed
    )