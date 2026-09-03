import os
import csv
import re
import numpy as np
import matplotlib
import warnings
import decimal

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.signal import savgol_filter

poly_order = 4
max_grad = 0.025  # The absolute maximum gradient to include in the zoomed-in graphs
print(f"poly order = {poly_order}", flush=True)
print(f"max grad limit = {max_grad}", flush=True)


def calc_threshold_snr_interpolated(I, IErr, V):
    """
    Calculates Vth using the Continuous Signal-to-Noise Ratio (SNR) Breakout method.
    Finds the interpolated voltage where the signal dynamically breaks out of
    2x and 4x the simulation's specific error array (IErr).

    Returns:
        v_th: The midpoint voltage between the 2x and 4x thresholds.
        v_err: The uncertainty, defined as half the voltage gap between the thresholds.
    """
    if len(I) < 2:
        return np.nan, np.nan

    def find_crossing_V(multiplier):
        # We look for the exact point where I overtakes multiplier * IErr
        diff = I - (multiplier * IErr)
        mask = diff > 0

        if not mask.any():
            return np.nan

        idx = np.argmax(mask)  # First index where current breaks the noise multiplier

        if idx == 0:
            return V[0]

        # Linearly interpolate between the point just before and just after the crossing
        v_before, v_after = V[idx - 1], V[idx]
        diff_before, diff_after = diff[idx - 1], diff[idx]

        # 0 = diff_before + (diff_after - diff_before) * (v_exact - v_before) / (v_after - v_before)
        slope = (diff_after - diff_before) / (v_after - v_before)
        if slope == 0:
            return v_before

        v_exact = v_before - (diff_before / slope)
        return v_exact

    # Find the continuous small and big breakout voltages
    small_V = find_crossing_V(2.0)
    big_V = find_crossing_V(4.0)

    if np.isnan(small_V) or np.isnan(big_V):
        return np.nan, np.nan

    # Return the exact continuous midpoint threshold and its mathematical uncertainty
    v_th = (small_V + big_V) / 2.0
    v_err = abs(big_V - small_V) / 2.0

    return v_th, v_err


def parse_params(filepath):
    """
    Parses the parameters_{run_name}_rep{X}.txt files to establish the
    physical system constants and the specific temperature gradient.
    """
    params = {}
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract physical system parameters for grouping
    match_D = re.search(r'(?:gap ratio|D)\s*:\s*([\d.]+)', content, re.IGNORECASE)
    match_Cg = re.search(r'Cg\s*:\s*([\d.]+)', content)
    match_stdR = re.search(r'stdR \(exponent\)\s*:\s*([\d.]+)', content)
    match_sig = re.search(r'sig \(normal\)\s*:\s*([\d.]+)', content)
    match_T0 = re.search(r'T0\s*:\s*([\d.]+)', content)
    match_flip = re.search(r'flip\s*:\s*(True|False)', content)

    params['D'] = float(match_D.group(1)) if match_D else 0.0
    params['Cg'] = float(match_Cg.group(1)) if match_Cg else 0.0
    params['stdR'] = float(match_stdR.group(1)) if match_stdR else 0.0
    params['sig'] = float(match_sig.group(1)) if match_sig else 0.0
    params['T0'] = float(match_T0.group(1)) if match_T0 else 0.001
    params['flip'] = True if (match_flip and match_flip.group(1) == 'True') else False

    # Extract temperatures to find the total gradient delta T
    match_T = re.search(r'T\s*:\s*\[(.*?)\]', content)
    if match_T:
        # Replaced commas with spaces to securely split by whitespace
        t_vals = [float(x) for x in match_T.group(1).replace(',', ' ').split()]
        params['T_left'] = t_vals[0]
        params['T_right'] = t_vals[-1]
        params['T_mid'] = t_vals[len(t_vals) // 2]

        # Exact directional gradient (will naturally be negative if flip=True)
        params['dT'] = params['T_right'] - params['T_left']
    else:
        raise NameError(f"{filepath} has a corrupted T list")

    return params


def get_folder_midfix_info(directory, run_name):
    """
    Evaluates up to two parameter files in a folder to determine if the run is
    a midfix physical setup or a standard T0-anchored setup.
    Returns: (midfix: bool, Tmid: float/None, is_corrupted: bool)
    """
    param_files = list(directory.glob(f"parameters_{run_name}_rep*.txt"))
    if not param_files:
        param_files = [f for f in directory.glob("*.txt") if "parameters" in f.name and "rep" in f.name]

    if not param_files:
        return False, None, False

    parsed_params = [parse_params(f) for f in param_files[:2]]

    if len(parsed_params) == 1:
        p = parsed_params[0]
        if abs(p['T_left'] - p['T0']) < 1e-6 or abs(p['T_right'] - p['T0']) < 1e-6:
            return False, None, False
        else:
            return True, p['T_mid'], False

    p1, p2 = parsed_params[0], parsed_params[1]

    # Condition 1: Tleft=T0 in both OR Tright=T0 in both (Standard)
    if (abs(p1['T_left'] - p1['T0']) < 1e-6 and abs(p2['T_left'] - p2['T0']) < 1e-6) or \
            (abs(p1['T_right'] - p1['T0']) < 1e-6 and abs(p2['T_right'] - p2['T0']) < 1e-6):
        return False, None, False

    # Condition 2: Middle entry is identical in both (Midfix)
    if abs(p1['T_mid'] - p2['T_mid']) < 1e-6:
        return True, p1['T_mid'], False

    # ELSE corrupted
    return False, None, True


def read_sweeps(csv_path):
    """
    Reads V, I, and I_err (row[2]) vectors, isolating both the forward (Up)
    and backward (Down) sweeps. Reverses the Down sweep so the SNR
    algorithm evaluates it forward from 0 -> Vmax.
    """
    v_col, i_col, ierr_col = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                v_col.append(float(row[0]))
                i_col.append(float(row[1]))
                ierr_col.append(float(row[2]))
            except ValueError:
                pass

    if not v_col:
        empty = np.array([])
        return empty, empty, empty, empty, empty, empty

    # Find the peak voltage index to split the arrays
    idx_max = v_col.index(max(v_col))

    # Isolate forward sweep (0 -> Vmax)
    v_up = np.array(v_col[:idx_max + 1])
    i_up = np.array(i_col[:idx_max + 1])
    ierr_up = np.array(ierr_col[:idx_max + 1])

    # Isolate backward sweep (Vmax -> 0) and flip to (0 -> Vmax) for SNR math
    v_down = np.flip(np.array(v_col[idx_max:]))
    i_down = np.flip(np.array(i_col[idx_max:]))
    ierr_down = np.flip(np.array(ierr_col[idx_max:]))

    return v_up, i_up, ierr_up, v_down, i_down, ierr_down


# Folders that should be completely skipped by the script
ignored_folders = {
    "20251207_17h43m26s",
    "20260605_19h36m00s",
    "20260606_22h05m04s"
}


def run_scanner_mode(base_dir, ivs_txt_path):
    """
    MODE 1: scans folders for duplicates and maps the physical runs.
    Outputs the log to IVs.txt.
    """
    print(f"[{ivs_txt_path.name} NOT FOUND] -> Initializing Scanner Mode...", flush=True)

    # catalog structure: catalog[(D, stdR, sig, T0, midfix, Tmid)][Cg][flip][rep] = [folder1, folder2, ...]
    catalog = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    corrupted_folders = []

    for directory in base_dir.glob("results_*"):
        if not directory.is_dir(): continue
        run_name = directory.name.replace("results_", "")

        # Skip explicitly ignored folders
        if run_name in ignored_folders:
            continue

        midfix, Tmid, is_corrupted = get_folder_midfix_info(directory, run_name)
        if is_corrupted:
            corrupted_folders.append(run_name)
            continue

        for csv_path in directory.glob("*.csv"):
            name = csv_path.name
            match_rep = re.search(r"rep(\d+)", name)
            if not match_rep: continue
            rep = int(match_rep.group(1))

            param_file = directory / f"parameters_{run_name}_rep{rep}.txt"
            if not param_file.exists():
                param_files = list(directory.glob(f"*rep{rep}*.txt"))
                if param_files:
                    param_file = param_files[0]
                else:
                    continue

            params = parse_params(param_file)
            D, stdR, sig, T0, Cg, flip = params['D'], params['stdR'], params['sig'], params['T0'], params['Cg'], params['flip']

            # Track folder strictly grouped by physics parameters and separated internally by flip
            catalog[(D, stdR, sig, T0, midfix, Tmid)][Cg][flip][rep].append(f"results_{run_name}")

    # Generate the IVs.txt report
    with open(ivs_txt_path, 'w') as f:
        # Log corrupted runs clearly at the top of the IVs.txt output
        if corrupted_folders:
            f.write("===== CORRUPTED FOLDERS (INSPECT THESE) =====\n")
            for cf in corrupted_folders:
                f.write(f"results_{cf}\n")
            f.write("=============================================\n\n")

        for (D, stdR, sig, T0, midfix, Tmid), cg_data in catalog.items():
            f.write(f"----- | D={D} | StdR={stdR} ; sig={sig} ; T0={T0} ; midfix={midfix} ; Tmid={Tmid} ---------\n")

            for Cg, flip_dict in cg_data.items():
                multiples = []
                relevant_folders = set()

                for flip_val in [True, False]:
                    if flip_val in flip_dict:
                        rep_dict = flip_dict[flip_val]
                        all_runs = sorted(list(rep_dict.keys()))
                        f.write(f"Cg : {Cg}, flip={flip_val}, with runs {all_runs}\n")

                        for rep, folders in rep_dict.items():
                            if len(folders) > 1:
                                multiples.append(f"rep{rep}(flip={flip_val})")
                                relevant_folders.update(folders)

                multiples = sorted(multiples)
                folders_sorted = sorted(list(relevant_folders))
                folders_str = ', '.join(folders_sorted) if folders_sorted else 'None'

                f.write(f"runs with multiples : {multiples}\n")
                f.write(f"folders relevant : {folders_str}\n\n")

            f.write("-------------------------------------------\n")

    print(f"\n[DONE] Scan complete. Diagnostic file created at:\n{ivs_txt_path.absolute()}")
    print("Please review it, delete multiple runs/folders as needed, delete IVs.txt, and run this script again.")


def run_analysis_mode(base_dir):
    """
    MODE 2: Full physical extraction, plotting, and S(T) differentiation.
    """
    print("[IVs.txt FOUND] -> Clean data assumed. Initializing Analysis Mode...")

    output_dir = base_dir / f"Thermopower_Analysis_Normal_Metal_savgol"
    output_dir.mkdir(exist_ok=True)

    # Key: (D, Cg, stdR, sig, T0, midfix, Tmid) -> Value: list of dictionaries
    system_groups = defaultdict(list)

    print("Scanning for results directories...", flush=True)
    for directory in base_dir.glob("results_*"):
        if not directory.is_dir():
            continue

        run_name = directory.name.replace("results_", "")

        # Skip explicitly ignored folders
        if run_name in ignored_folders:
            continue

        midfix, Tmid, is_corrupted = get_folder_midfix_info(directory, run_name)
        if is_corrupted:
            print(f"Skipping corrupted folder: {run_name}", flush=True)
            continue

        print(f"Processing run batch: {run_name}", flush=True)

        for csv_path in directory.glob("*.csv"):
            name = csv_path.name

            # repetition index matches the gradient size
            match_rep = re.search(r"rep(\d+)", name)
            if not match_rep:
                continue
            rep = int(match_rep.group(1))

            param_file = directory / f"parameters_{run_name}_rep{rep}.txt"
            if not param_file.exists():
                print(f"Warning: No param file found for {name}. Skipping...")
                continue

            params = parse_params(param_file)
            sys_key = (params['D'], params['Cg'], params['stdR'], params['sig'], params['T0'], midfix, Tmid)

            # Extract both sweeps including the dynamic IErr array
            v_up, i_up, ierr_up, v_down, i_down, ierr_down = read_sweeps(csv_path)
            if len(v_up) == 0:
                continue

            # Apply dynamic SNR Algorithm to both sweeps utilizing the specific IErr
            # Extract both Vth AND the dynamic uncertainty margin (err)
            vth_up, err_up = calc_threshold_snr_interpolated(i_up, ierr_up, v_up)
            vth_down, err_down = calc_threshold_snr_interpolated(i_down, ierr_down, v_down)

            system_groups[sys_key].append({
                'Run_Name': run_name,
                'Repetition': rep,
                'Delta_T': params['dT'],
                'Vth_up': vth_up,
                'err_up': err_up,
                'Vth_down': vth_down,
                'err_down': err_down
            })

    # Process and Plot for each unique physical system
    for sys_key, data in system_groups.items():
        if len(data) < 2:
            continue

        D, Cg, stdR, sig, T0, midfix, Tmid = sys_key

        sys_folder_name = f"System_D{D}_Cg{Cg}_stdR{stdR}_sig{sig}_T0_{T0}"
        if midfix:
            Tmid_ratio = Tmid / T0
            Tmid_units = int(Tmid_ratio)
            decimal_part = decimal.Decimal(str(Tmid_ratio - Tmid_units))
            decimal_str = str(decimal_part).replace('0.', '')
            Tmid_pastdigits = int(decimal_str) if decimal_str else 0
            sys_folder_name += f"_Tmid{Tmid_units}_{Tmid_pastdigits}"

        sys_dir = output_dir / sys_folder_name
        sys_dir.mkdir(exist_ok=True)

        # Sort dynamically maps data from most negative (flip=True) to most positive (flip=False)
        data = sorted(data, key=lambda x: x['Delta_T'])

        dTs = np.array([d['Delta_T'] for d in data])
        Vth_up = np.array([d['Vth_up'] for d in data])
        Vth_down = np.array([d['Vth_down'] for d in data])
        err_up = np.array([d['err_up'] for d in data])
        err_down = np.array([d['err_down'] for d in data])

        # Filter NaNs ensuring arrays stay parallel
        valid_mask = ~np.isnan(Vth_up) & ~np.isnan(Vth_down)
        dTs = dTs[valid_mask]
        Vth_up = Vth_up[valid_mask]
        Vth_down = Vth_down[valid_mask]
        err_up = err_up[valid_mask]
        err_down = err_down[valid_mask]

        if len(dTs) < 4:
            print(f"Skipping {sys_folder_name} - Not enough valid threshold data.", flush=True)
            continue

        # -----------------------------------------------------
        # Dynamic Thermopower Derivative using Savitzky-Golay
        # -----------------------------------------------------
        min_window = poly_order + 1
        if min_window % 2 == 0:
            min_window += 1

        # need odd number for savgol
        max_window = 7
        window_length = min(max_window, len(dTs) if len(dTs) % 2 != 0 else len(dTs) - 1)
        if window_length < min_window:
            window_length = min_window if min_window <= len(dTs) else (len(dTs) if len(dTs) % 2 != 0 else len(dTs) - 1)

        # Only perform the fit if we have enough points and a valid odd window
        if window_length > poly_order and window_length % 2 != 0:
            avg_dx = np.mean(np.diff(dTs))
            # Fallback if step is perfectly 0 to avoid division by zero
            if avg_dx == 0:
                avg_dx = 1e-6

            S_up = -savgol_filter(Vth_up, window_length=window_length, polyorder=poly_order, deriv=1, delta=avg_dx)
            S_down = -savgol_filter(Vth_down, window_length=window_length, polyorder=poly_order, deriv=1, delta=avg_dx)
            S_dT = dTs

            # Standard error propagation for subtraction across standard temperature steps
            dT_steps = np.diff(dTs)
            dT_steps = np.append(dT_steps, dT_steps[-1])
            S_err_up = np.sqrt(err_up ** 2 + np.roll(err_up, shift=1) ** 2) / dT_steps
            S_err_down = np.sqrt(err_down ** 2 + np.roll(err_down, shift=1) ** 2) / dT_steps
            S_err_up[0], S_err_down[0] = S_err_up[1], S_err_down[1]  # fallback for the first point boundary
        else:
            # Fallback to standard gradient if there are too few points
            S_up = -np.gradient(Vth_up, dTs)
            S_down = -np.gradient(Vth_down, dTs)
            S_dT = dTs

            dT_steps = np.diff(dTs)
            dT_steps = np.append(dT_steps, dT_steps[-1])
            S_err_up = np.sqrt(err_up ** 2 + np.roll(err_up, shift=1) ** 2) / dT_steps
            S_err_down = np.sqrt(err_down ** 2 + np.roll(err_down, shift=1) ** 2) / dT_steps
            S_err_up[0], S_err_down[0] = S_err_up[1], S_err_down[1]

        # REMOVE THE NOISY TAILS AFTER SAVGOL CALCULATION
        # Because we merged the flips, extreme positive gradient is at index [-1],
        # and extreme negative gradient is at index [0]. We drop them both to be safe.
        if len(dTs) > 0:
            if dTs[-1] > 0:
                dTs = dTs[:-1]
                Vth_up = Vth_up[:-1]
                Vth_down = Vth_down[:-1]
                err_up = err_up[:-1]
                err_down = err_down[:-1]
                S_dT = S_dT[:-1]
                S_up = S_up[:-1]
                S_down = S_down[:-1]
                S_err_up = S_err_up[:-1]
                S_err_down = S_err_down[:-1]
            if len(dTs) > 0 and dTs[0] < 0:
                dTs = dTs[1:]
                Vth_up = Vth_up[1:]
                Vth_down = Vth_down[1:]
                err_up = err_up[1:]
                err_down = err_down[1:]
                S_dT = S_dT[1:]
                S_up = S_up[1:]
                S_down = S_down[1:]
                S_err_up = S_err_up[1:]
                S_err_down = S_err_down[1:]

        export_csv_path = sys_dir / "aggregated_thermopower_results.csv"
        with open(export_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Delta_T", "Vth_Up", "Vth_Down", "S_Up", "S_Down", "S_err_up", "S_err_down"])
            for idx, dt_val in enumerate(dTs):
                writer.writerow(
                    [dt_val, Vth_up[idx], Vth_down[idx], S_up[idx], S_down[idx], S_err_up[idx], S_err_down[idx]])

        # Create unified title without 'flip' label
        title_str = f"D={D}, Cg={Cg}, stdR={stdR}, $\\sigma$={sig}, T0={T0}"
        if midfix:
            title_str += f", Tmid={Tmid}"

        # ==========================================================
        # GRAPH 1A: Threshold Voltage (Up vs Down) - WITH ERROR BARS
        # ==========================================================
        plt.figure(figsize=(10, 6))
        plt.errorbar(dTs, Vth_up, yerr=err_up, marker='o', linestyle='-', color='dodgerblue', linewidth=2,
                     label='Sweep Up', capsize=3)
        plt.errorbar(dTs, Vth_down, yerr=err_down, marker='s', linestyle='--', color='crimson', linewidth=2,
                     label='Sweep Down', capsize=3)

        plt.xlabel(r'Total Temperature Gradient $\Delta T = T_{right} - T_{left}$ ($e^2 / (k_B \langle C \rangle)$)',
                   fontsize=12)
        plt.ylabel(r'Threshold Voltage $V_{th}$ ($e / \langle C \rangle$) [SNR Breakout]', fontsize=12)
        plt.title(f'Threshold Voltage Hysteresis vs. Gradient\n{title_str}', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(sys_dir / 'Vth_vs_Gradient_Hysteresis_with_error.png', dpi=300)
        plt.close()

        # ==========================================================
        # GRAPH 1B: Threshold Voltage (Up vs Down) - NO ERROR BARS
        # ==========================================================
        plt.figure(figsize=(10, 6))
        plt.plot(dTs, Vth_up, marker='o', linestyle='-', color='dodgerblue', linewidth=2, label='Sweep Up')
        plt.plot(dTs, Vth_down, marker='s', linestyle='--', color='crimson', linewidth=2, label='Sweep Down')

        plt.xlabel(r'Total Temperature Gradient $\Delta T = T_{right} - T_{left}$ ($e^2 / (k_B \langle C \rangle)$)',
                   fontsize=12)
        plt.ylabel(r'Threshold Voltage $V_{th}$ ($e / \langle C \rangle$) [SNR Breakout]', fontsize=12)
        plt.title(f'Threshold Voltage Hysteresis vs. Gradient\n{title_str}', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(sys_dir / 'Vth_vs_Gradient_Hysteresis_no_error.png', dpi=300)
        plt.close()

        # ==========================================================
        # GRAPH 2A: Thermopower S(T) (Up vs Down) - WITH ERROR BANDS
        # ==========================================================
        plt.figure(figsize=(10, 6))
        plt.plot(S_dT, S_up, marker='o', linestyle='-', color='dodgerblue', linewidth=2, label='S(T) Up')
        plt.fill_between(S_dT, S_up - S_err_up, S_up + S_err_up, color='dodgerblue', alpha=0.2)

        plt.plot(S_dT, S_down, marker='s', linestyle='--', color='crimson', linewidth=2, label='S(T) Down')
        plt.fill_between(S_dT, S_down - S_err_down, S_down + S_err_down, color='crimson', alpha=0.2)

        plt.axhline(0, color='black', linestyle='-', alpha=0.8)
        plt.xlabel(r'Total Temperature Gradient $\Delta T = T_{right} - T_{left}$ ($e^2 / (k_B \langle C \rangle)$)',
                   fontsize=12)
        plt.ylabel(r'Thermopower $S(T) = -dV_{th} / d(\Delta T)$ ($k_B / e$)', fontsize=12)
        plt.title(f'Thermopower Hysteresis vs. Gradient\n{title_str}', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(sys_dir / 'Thermopower_S_vs_Gradient_Hysteresis_with_error.png', dpi=300)
        plt.close()

        # ==========================================================
        # GRAPH 2B: Thermopower S(T) (Up vs Down) - NO ERROR BANDS
        # ==========================================================
        plt.figure(figsize=(10, 6))
        plt.plot(S_dT, S_up, marker='o', linestyle='-', color='dodgerblue', linewidth=2, label='S(T) Up')
        plt.plot(S_dT, S_down, marker='s', linestyle='--', color='crimson', linewidth=2, label='S(T) Down')

        plt.axhline(0, color='black', linestyle='-', alpha=0.8)
        plt.xlabel(r'Total Temperature Gradient $\Delta T = T_{right} - T_{left}$ ($e^2 / (k_B \langle C \rangle)$)',
                   fontsize=12)
        plt.ylabel(r'Thermopower $S(T) = -dV_{th} / d(\Delta T)$ ($k_B / e$)', fontsize=12)
        plt.title(f'Thermopower Hysteresis vs. Gradient\n{title_str}', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(sys_dir / 'Thermopower_S_vs_Gradient_Hysteresis_no_error.png', dpi=300)
        plt.close()

        # ==========================================================
        # ADDITIONAL GRAPHS: Truncated by max_grad
        # ==========================================================
        grad_mask = np.abs(dTs) <= max_grad

        if np.sum(grad_mask) >= 2:
            dTs_z = dTs[grad_mask]
            Vth_up_z = Vth_up[grad_mask]
            Vth_down_z = Vth_down[grad_mask]
            err_up_z = err_up[grad_mask]
            err_down_z = err_down[grad_mask]

            S_dT_z = S_dT[grad_mask]
            S_up_z = S_up[grad_mask]
            S_down_z = S_down[grad_mask]
            S_err_up_z = S_err_up[grad_mask]
            S_err_down_z = S_err_down[grad_mask]

            title_str_zoomed = title_str + f"\n(|$\\Delta T| \\leq {max_grad}$)"

            # --- Zoomed GRAPH 1A ---
            plt.figure(figsize=(10, 6))
            plt.errorbar(dTs_z, Vth_up_z, yerr=err_up_z, marker='o', linestyle='-', color='dodgerblue', linewidth=2,
                         label='Sweep Up', capsize=3)
            plt.errorbar(dTs_z, Vth_down_z, yerr=err_down_z, marker='s', linestyle='--', color='crimson', linewidth=2,
                         label='Sweep Down', capsize=3)

            plt.xlabel(
                r'Total Temperature Gradient $\Delta T = T_{right} - T_{left}$ ($e^2 / (k_B \langle C \rangle)$)',
                fontsize=12)
            plt.ylabel(r'Threshold Voltage $V_{th}$ ($e / \langle C \rangle$) [SNR Breakout]', fontsize=12)
            plt.title(f'Threshold Voltage Hysteresis vs. Gradient\n{title_str_zoomed}', fontsize=14)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(sys_dir / 'Vth_vs_Gradient_Hysteresis_with_error_zoomed.png', dpi=300)
            plt.close()

            # --- Zoomed GRAPH 1B ---
            plt.figure(figsize=(10, 6))
            plt.plot(dTs_z, Vth_up_z, marker='o', linestyle='-', color='dodgerblue', linewidth=2, label='Sweep Up')
            plt.plot(dTs_z, Vth_down_z, marker='s', linestyle='--', color='crimson', linewidth=2, label='Sweep Down')

            plt.xlabel(
                r'Total Temperature Gradient $\Delta T = T_{right} - T_{left}$ ($e^2 / (k_B \langle C \rangle)$)',
                fontsize=12)
            plt.ylabel(r'Threshold Voltage $V_{th}$ ($e / \langle C \rangle$) [SNR Breakout]', fontsize=12)
            plt.title(f'Threshold Voltage Hysteresis vs. Gradient\n{title_str_zoomed}', fontsize=14)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(sys_dir / 'Vth_vs_Gradient_Hysteresis_no_error_zoomed.png', dpi=300)
            plt.close()

            # --- Zoomed GRAPH 2A ---
            plt.figure(figsize=(10, 6))
            plt.plot(S_dT_z, S_up_z, marker='o', linestyle='-', color='dodgerblue', linewidth=2, label='S(T) Up')
            plt.fill_between(S_dT_z, S_up_z - S_err_up_z, S_up_z + S_err_up_z, color='dodgerblue', alpha=0.2)

            plt.plot(S_dT_z, S_down_z, marker='s', linestyle='--', color='crimson', linewidth=2, label='S(T) Down')
            plt.fill_between(S_dT_z, S_down_z - S_err_down_z, S_down_z + S_err_down_z, color='crimson', alpha=0.2)

            plt.axhline(0, color='black', linestyle='-', alpha=0.8)
            plt.xlabel(
                r'Total Temperature Gradient $\Delta T = T_{right} - T_{left}$ ($e^2 / (k_B \langle C \rangle)$)',
                fontsize=12)
            plt.ylabel(r'Thermopower $S(T) = -dV_{th} / d(\Delta T)$ ($k_B / e$)', fontsize=12)
            plt.title(f'Thermopower Hysteresis vs. Gradient\n{title_str_zoomed}', fontsize=14)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(sys_dir / 'Thermopower_S_vs_Gradient_Hysteresis_with_error_zoomed.png', dpi=300)
            plt.close()

            # --- Zoomed GRAPH 2B ---
            plt.figure(figsize=(10, 6))
            plt.plot(S_dT_z, S_up_z, marker='o', linestyle='-', color='dodgerblue', linewidth=2, label='S(T) Up')
            plt.plot(S_dT_z, S_down_z, marker='s', linestyle='--', color='crimson', linewidth=2, label='S(T) Down')

            plt.axhline(0, color='black', linestyle='-', alpha=0.8)
            plt.xlabel(
                r'Total Temperature Gradient $\Delta T = T_{right} - T_{left}$ ($e^2 / (k_B \langle C \rangle)$)',
                fontsize=12)
            plt.ylabel(r'Thermopower $S(T) = -dV_{th} / d(\Delta T)$ ($k_B / e$)', fontsize=12)
            plt.title(f'Thermopower Hysteresis vs. Gradient\n{title_str_zoomed}', fontsize=14)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(sys_dir / 'Thermopower_S_vs_Gradient_Hysteresis_no_error_zoomed.png', dpi=300)
            plt.close()

        print(f"Exported data and generated dual-sweep graphs in: {sys_dir}", flush=True)

    print("\nBatch Thermopower processing complete.")


def main():
    base_dir = Path(__file__).parent.absolute()
    ivs_txt_path = base_dir / "IVs.txt"

    if not ivs_txt_path.exists():
        run_scanner_mode(base_dir, ivs_txt_path)
    else:
        run_analysis_mode(base_dir)


if __name__ == '__main__':
    main()