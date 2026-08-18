import os

LOOPS_PER_TASK = 80
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
num_workers = min(total_cpus - 5, LOOPS_PER_TASK)
print(f"worker number set to {num_workers} ; for {total_cpus} cpus", flush=True)

# -------------------------------------------------------------
# SLURM JOB ARRAY PARSING
# -------------------------------------------------------------
task_id_str = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
TASK_ID = int(task_id_str)
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
import math
from define_objects import IMPORT_EXPORT, ExperimentInitialState, SteadyStateVaryVResult
from gamma_functions import Get_Steady_State_varyV
from preparation import (
    prepare_initial_state,
    validate_table_triplets_file,
    update_init_Cg_Rg
)
from preparation import prepare_table_triplets_gapped, prepare_table_triplets, output_table_triplets, \
    prepare_table_triplets_NIS
from dataclasses import asdict
import orjson
import csv
import time
import re
import pickle

####### SLURM parameter parsing from job name ######
job_name = os.environ.get('SLURM_JOB_NAME', 'TPvaryV_rep40_Cg2_D0_0')
pattern = r"(Reverse_?)?TPvaryV_rep(\d+)_Cg(\d+)_D(\d+)_(\d+)_TmidNonT0"
match = re.search(pattern, job_name)

if match:
    is_reverse = match.group(1) is not None
    REPETITION = int(match.group(2))
    Cg = int(match.group(3))
    gap_int = int(match.group(4))
    gap_tenth = int(match.group(5))
    gap_ratio = gap_int + gap_tenth / 10

    print(f"Parsed from Job Name '{job_name}': flip={is_reverse}, rep={REPETITION}, Cg={Cg}, D={gap_ratio}",
          flush=True)
else:
    raise NameError(f"Job Name is improperly formatted : {job_name}")

Cg_list = [10]
Cg_list_gapped = [10]
gap_list = [2.0]


##############################################################

def main(import_export: IMPORT_EXPORT, run_name, mean_Cg, first_rep, is_resumed=False) -> None:
    checkpoint_dir = import_export.results_dir_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # RUN TYPE
    flip = is_reverse
    periodic_y = True

    # EXPERIMENT PARAMETERS
    loop_count = max(320, END_LOOP_IDX)  # Dynamically size for arrays
    T0_unitless = 0.001
    mean_Rg = 100
    stdR = 2
    sig = 0.05
    repetition = REPETITION

    if gap_ratio > 1e-3:
        if (Cg not in Cg_list_gapped) or (gap_ratio not in gap_list):
            raise ValueError(f"Cg must be in Cg_list_gapped, Cg = {Cg} ; "
                             f"gap ratio must be between in gap_list, D = {gap_ratio}")
        if Cg == 10 and gap_ratio == 2 and repetition == 40:
            pos_energy_boundT0 = 0.36 # FOR T0 0
            neg_energy_boundT0 = -0.461 # FOR T0 -0.30769
            pos_energy_bounddT = 0.62
            neg_energy_bounddT = -0.76
        else:
            raise ValueError("Unexpected Cg/Gap ratio combination.")

        if repetition not in [40, 96]:
            raise ValueError(
                f"CRITICAL: N-I-S Boundary tables are currently only generated for rep 40 and 96. "
                f"Requested rep={repetition}. Aborting to prevent invalid SC boundary"
            )

    else:
        if Cg not in Cg_list:
            raise ValueError(f"Cg must be in Cg_list, Cg = {Cg}")
        if Cg == 10:
            pos_energy_boundT0 = 0.32
            neg_energy_boundT0 = -0.42
            pos_energy_bounddT = 0.6
            neg_energy_bounddT = -0.71
        else:
            raise ValueError("what")

    print(f"############# MAIN PARAMETERS ##################")
    print(f"flip = {flip}", flush=True)
    print(f"Total Loops for this specific Array Worker: {loop_count}", flush=True)
    print(f"gap_ratio = {gap_ratio}", flush=True)
    print(f"Cg = {Cg}")

    # ---------------------------------------------------------
    # STATE INITIALIZATION & MIDFIX CALCULATION
    # ---------------------------------------------------------
    current_run_json = import_export.results_dir_path / f"{run_name}.json"
    sliced_baseline_npy = import_export.results_dir_path / "midfix_baseline_arrays.npy"

    # Pre-calculate T_mid locally for all tasks
    T_std = repetition * T0_unitless / 20
    center_idx = (7 - 1) / 2.0  # Hardcoded 7 row_num based on earlier scripts
    T_mid = T0_unitless + center_idx * T_std
    T_baseline = [T_mid] * 7

    if not is_resumed:
        # I AM THE CREATOR (Won the mkdir race for a brand new run)
        print(f"Task {TASK_ID} won the initialization race. Building state...", flush=True)

        run_to_get_init_from = "20260606_22h05m04s"
        results_dir_of_past_run = Path(__file__).resolve().parent.parent / f"results_{run_to_get_init_from}"
        past_infile = results_dir_of_past_run / f"{run_to_get_init_from}.json"

        if past_infile.exists():
            json_txt = past_infile.read_text()
            raw_fields = orjson.loads(json_txt)
            init_str = ExperimentInitialState(**raw_fields)
            init = F.fix_types(init_str, loop_count)

            init = F.swap_in_init("flip", flip, init)
            if init.Cg[0] != mean_Cg or init.Rg[0] != mean_Rg:
                init = update_init_Cg_Rg(init, mean_Cg, mean_Rg)

            if np.any(init.R_t_ij < 0.1):
                min_Rt = np.min(init.R_t_ij)
                init = F.swap_in_init("R_t_ij", init.R_t_ij + (0.1 - min_Rt), init)

            print(f"Successfully loaded and modified base state from {past_infile.name}", flush=True)
        else:
            init = prepare_initial_state(loop_count=loop_count, unitless_T0=T0_unitless, flip=flip,
                                         periodic_y=periodic_y,
                                         Cg_C_ratio=mean_Cg, Rg_R_ratio=mean_Rg, stdR_R_ratio=stdR, sigC_C_ratio=sig)
            print("!!!NEW INITIAL STATE CREATED FROM SCRATCH!!!", flush=True)

        # -- CREATOR EXTRACTS THE BASELINE ARRAYS ONCE --
        gradient_table_path = import_export.prepare_table_triplets_file_list[repetition]
        if not gradient_table_path.exists():
            raise FileNotFoundError(f"Missing required integration table: {gradient_table_path.name}")

        print(f"Slicing T_mid arrays from: {gradient_table_path.name}...", flush=True)
        big_table = np.load(gradient_table_path.as_posix())

        mask = np.isclose(big_table["temp"], T_mid, rtol=1e-5)
        if not np.any(mask):
            print(f"Available temperatures in table: {np.unique(big_table['temp'])}", flush=True)
            raise ValueError(f"CRITICAL: T_mid ({T_mid}) was not found inside the loaded table.")

        # Save the sliced baseline arrays for the followers
        with open(sliced_baseline_npy, 'wb') as f:
            np.save(f, big_table["val"][mask])
            np.save(f, big_table["prob"][mask])

        # ATOMIC WRITE FOR FOLLOWERS
        temp_json = import_export.results_dir_path / f"temp_init_{TASK_ID}.json"
        temp_json.write_text(orjson.dumps(asdict(init), option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8"))
        temp_json.replace(current_run_json)
        print(f"Creator Task {TASK_ID} successfully dropped {current_run_json.name} for followers.", flush=True)

    else:
        # I AM A FOLLOWER
        print(f"Task {TASK_ID} is a follower. Waiting for Creator to drop {current_run_json.name}...", flush=True)

        while not current_run_json.exists() or not sliced_baseline_npy.exists():
            time.sleep(2)

        while True:
            try:
                json_txt = current_run_json.read_text()
                raw_fields = orjson.loads(json_txt)
                break
            except ValueError:
                time.sleep(0.5)

        init_str = ExperimentInitialState(**raw_fields)
        init = F.fix_types(init_str, loop_count)
        print(f"Task {TASK_ID} successfully loaded synced state.", flush=True)

    # ---------------------------------------------------------
    # DEFINE MACRO VOLTAGE SWEEP
    # ---------------------------------------------------------
    V_diff = 4
    steps = 100
    V_sweep = np.linspace(init.Vright * init.Volts, (init.Vright + V_diff) * init.Volts, num=steps)
    cycles = len(V_sweep)

    # ---------------------------------------------------------
    # MIDFIX BASELINE SETUP
    # ---------------------------------------------------------
    Delta_0 = gap_ratio * init.Ec
    gap_array_baseline = F.exact_bcs_gap(T_baseline, Delta_0)
    expected_err_baseline = F.calc_expected_dist_std(T_baseline, init.T0, gap_array_baseline, init.R_t_ij, init.Ec)

    print(f"############# INITIALIZING MIDFIX BASELINE ##################")
    print(f"Target Midfix Temperature (T_mid) = {T_mid}", flush=True)

    # All tasks (Creator and Followers) now safely load the pre-sliced array
    with open(sliced_baseline_npy, 'rb') as f:
        table_val_baseline = np.load(f)
        table_prob_baseline = np.load(f)
    table_T_baseline = [T_mid]

    print(f"Successfully loaded {len(table_val_baseline)} integration steps for the baseline state.", flush=True)

    print("############# RUN THERMOPOWER VARY-V EXPERIMENT ##################", flush=True)

    # ---------------------------------------------------------
    # MAIN EXPERIMENT LOGIC
    # ---------------------------------------------------------
    marker_file = import_export.results_dir_path / f".completed_task{TASK_ID}"
    if marker_file.exists():
        exit(f"Task {TASK_ID} already completed in this path : {import_export.results_dir_path}")

    T_dT = [init.T0 + i * T_std for i in range(init.row_num)]
    if flip:
        T_dT = np.flip(T_dT)

    gap_array_dT = F.exact_bcs_gap(T_dT, Delta_0)
    expected_err_dT = F.calc_expected_dist_std(T_dT, init.T0, gap_array_dT, init.R_t_ij, init.Ec)

    # Tasks load the gradient table individually here. This is safer because they are just
    # reading the whole array, not slicing it, reducing memory mapping strain.
    if not validate_table_triplets_file(import_export.prepare_table_triplets_file_list[repetition], init,
                                        np.array(T_dT)):
        raise ImportError(f"no validated table, skipped rep{T_std}")

    table_triplets = np.load(import_export.prepare_table_triplets_file_list[repetition].as_posix())
    table_val_dT = table_triplets["val"]
    table_prob_dT = table_triplets["prob"]
    table_T_dT = np.unique(table_triplets["temp"]).tolist()

    # 3b. Load and Validate the matching N-I-S Boundary Integration Table
    if gap_ratio > 1e-3:
        if repetition == 40:
            nis_table_path = import_export.export_path / f"64bit_GAP2_0_NIS_table_triplets_Tmid7_0_Tstd20_20_Cg_10.npz"
        elif repetition == 96:
            nis_table_path = import_export.export_path / f"64bit_GAP2_0_NIS_table_triplets_Tmid15_4_Tstd20_20_Cg_10.npz"
        else:
            raise ValueError(f"invalid repetition {repetition}")

        print(f"Validating N-I-S Boundary Table: {nis_table_path.name}...", flush=True)
        if not nis_table_path.exists():
            raise FileNotFoundError(f"CRITICAL: Missing N-I-S table at {nis_table_path}.")

        HighT = init.T0 * (1 + (init.row_num - 1) * repetition / 20)
        if not validate_table_triplets_file(nis_table_path, init, [init.T0, HighT]):
            raise ImportError(
                f"CRITICAL: N-I-S table temperatures do not match current dT gradient for rep {repetition}.")

        nis_triplets = np.load(nis_table_path.as_posix())
        nis_table_val = nis_triplets["val"]
        nis_table_prob = nis_triplets["prob"]
        print("N-I-S Boundary Table Successfully Loaded and Validated.", flush=True)
    else:
        nis_table_val = None
        nis_table_prob = None

    # 4. Dispatch to Physics Engine
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        t0 = time.time()

        loaded_state_function = partial(
            Get_Steady_State_varyV,
            init=init,
            V_sweep=V_sweep,
            cycles=cycles,

            # Baseline
            table_val_baseline=table_val_baseline,
            table_prob_baseline=table_prob_baseline,
            table_T_baseline=table_T_baseline,
            T_baseline=np.array(T_baseline),
            expected_error_baseline=expected_err_baseline,
            gap_array_baseline=gap_array_baseline,

            # Gradient
            table_val_dT=table_val_dT,
            table_prob_dT=table_prob_dT,
            table_T_dT=table_T_dT,
            T_dT=np.array(T_dT),
            expected_error_dT=expected_err_dT,
            gap_array_dT=gap_array_dT,

            # Constants
            flip=flip,
            pos_energy_bound=pos_energy_boundT0,
            neg_energy_bound=neg_energy_boundT0,
            repetition=repetition,
            periodic_y=periodic_y,
            gap_ratio=gap_ratio,

            # Boundary N-I-S Tables
            nis_table_val=nis_table_val,
            nis_table_prob=nis_table_prob
        )

        results = [None] * init.loop_count
        futures = {}

        safe_end_idx = min(END_LOOP_IDX, init.loop_count)

        for i in range(START_LOOP_IDX, safe_end_idx):
            ckpt_path = checkpoint_dir / f"ckpt_varyV_rep{repetition}_idx{i}.pkl"
            if ckpt_path.exists():
                try:
                    with open(ckpt_path, "rb") as f:
                        results[i] = pickle.load(f)
                except Exception:
                    ckpt_path.unlink(missing_ok=True)

            if results[i] is None:
                futures[executor.submit(loaded_state_function, i)] = i

        for future in as_completed(futures):
            idx = futures[future]
            res = future.result()
            results[idx] = res

            with open(checkpoint_dir / f"ckpt_varyV_rep{repetition}_idx{idx}.pkl", "wb") as f:
                pickle.dump(res, f)

    valid_results = [res for res in results[START_LOOP_IDX:safe_end_idx] if res is not None]

    if valid_results:
        out_pkl_path = import_export.results_dir_path / f"varyV_raw_task{TASK_ID}.pkl"
        with open(out_pkl_path, "wb") as f:
            pickle.dump({'V_sweep': V_sweep, 'results': valid_results}, f)
        print(f"Task {TASK_ID} successfully saved raw results to {out_pkl_path.name}")

    marker_file.touch()

    for i in range(START_LOOP_IDX, safe_end_idx):
        f = checkpoint_dir / f"ckpt_varyV_rep{repetition}_idx{i}.pkl"
        f.unlink(missing_ok=True)

    print(f"Task {TASK_ID} completed.", flush=True)


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
        if TASK_ID == 0:
            print(f"RESUMING existing run at {RESULTS_DIR_PATH}", flush=True)
        is_resumed = True
    else:
        date_ = datetime.datetime.now()
        run_name_flat = date_.strftime("%Y%m%d_%Hh%Mm%Ss")
        RESULTS_DIR_PATH = BASE_RESULTS_DIR / f"results_{run_name_flat}"

        try:
            RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=False)
            print(f"Created NEW results directory at {RESULTS_DIR_PATH}")
            is_resumed = False

            meta_data = {
                "slurm_job_name": job_name,
                "created_at": run_name_flat,
                "Cg": Cg,
                "gap_ratio": gap_ratio,
                "repetition": REPETITION
            }
            (RESULTS_DIR_PATH / "checkpoint_meta.json").write_text(orjson.dumps(meta_data).decode("utf-8"))
        except FileExistsError:
            is_resumed = True
            time.sleep(2)

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
        first_rep=None,
        is_resumed=is_resumed
    )