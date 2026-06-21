import os
import csv
import re
import numpy as np
import matplotlib
import warnings

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.signal import savgol_filter

poly_order = 4
print(f"poly order = {poly_order}", flush=True)


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
    v_err = abs(big_V - small_V) / 2.0  # Error is half the gap width

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
    match_Cg = re.search(r'Cg\s*:\s*([\d.]+)', content)
    match_stdR = re.search(r'stdR \(exponent\)\s*:\s*([\d.]+)', content)
    match_sig = re.search(r'sig \(normal\)\s*:\s*([\d.]+)', content)
    match_T0 = re.search(r'T0\s*:\s*([\d.]+)', content)

    params['Cg'] = float(match_Cg.group(1)) if match_Cg else 0.0
    params['stdR'] = float(match_stdR.group(1)) if match_stdR else 0.0
    params['sig'] = float(match_sig.group(1)) if match_sig else 0.0
    params['T0'] = float(match_T0.group(1)) if match_T0 else 0.001

    # Extract temperatures to find the total gradient delta T
    match_T = re.search(r'T\s*:\s*\[(.*?)\]', content)
    if match_T:
        t_vals = [float(x) for x in match_T.group(1).split(',')]
        params['T_left'] = t_vals[0]
        params['T_right'] = t_vals[-1]
        params['dT'] = params['T_right'] - params['T_left']
    else:
        params['dT'] = 0.0

    return params


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
    MODE 1: Rapidly scans folders for duplicates and maps the physical runs.
    Outputs the log to IVs.txt.
    """
    print(f"[{ivs_txt_path.name} NOT FOUND] -> Initializing Scanner Mode...", flush=True)

    # catalog structure: catalog[(stdR, sig, T0)][Cg][rep] = [folder1, folder2, ...]
    catalog = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for directory in base_dir.glob("results_*"):
        if not directory.is_dir(): continue
        run_name = directory.name.replace("results_", "")

        # Skip explicitly ignored folders
        if run_name in ignored_folders:
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
            stdR, sig, T0, Cg = params['stdR'], params['sig'], params['T0'], params['Cg']

            # Track exactly which folder this rep came from
            catalog[(stdR, sig, T0)][Cg][rep].append(f"results_{run_name}")

    # Generate the IVs.txt report
    with open(ivs_txt_path, 'w') as f:
        for (stdR, sig, T0), cg_data in catalog.items():
            f.write(f"----- StdR={stdR} ; sig={sig} ; T0={T0} ---------\n")

            for Cg, rep_dict in cg_data.items():
                all_runs = sorted(list(rep_dict.keys()))

                multiples = []
                relevant_folders = set()

                # Check for reps that exist in more than one folder
                for rep, folders in rep_dict.items():
                    if len(folders) > 1:
                        multiples.append(rep)
                        relevant_folders.update(folders)

                multiples = sorted(multiples)
                folders_sorted = sorted(list(relevant_folders))
                folders_str = ', '.join(folders_sorted) if folders_sorted else 'None'

                f.write(f"Cg : {Cg} with runs {all_runs}\n")
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

    output_dir = base_dir / f"Thermopower_Analysis_Normal_Metal_weighted_deg{poly_order}"
    output_dir.mkdir(exist_ok=True)

    # Key: (Cg, stdR, sig, T0) -> Value: list of dictionaries
    system_groups = defaultdict(list)

    print("Scanning for results directories...", flush=True)
    for directory in base_dir.glob("results_*"):
        if not directory.is_dir():
            continue

        run_name = directory.name.replace("results_", "")

        # Skip explicitly ignored folders
        if run_name in ignored_folders:
            continue

        print(f"Processing run batch: {run_name}", flush=True)

        for csv_path in directory.glob("*.csv"):
            name = csv_path.name

            # Extract repetition index
            match_rep = re.search(r"rep(\d+)", name)
            if not match_rep:
                continue
            rep = int(match_rep.group(1))

            param_file = directory / f"parameters_{run_name}_rep{rep}.txt"
            if not param_file.exists():
                # Fallback search if exact name slightly varies
                param_files = list(directory.glob(f"*rep{rep}*.txt"))
                if param_files:
                    param_file = param_files[0]
                else:
                    print(f"Warning: No param file found for {name}. Skipping...")
                    continue

            params = parse_params(param_file)
            sys_key = (params['Cg'], params['stdR'], params['sig'], params['T0'])

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

        Cg, stdR, sig, T0 = sys_key
        sys_folder_name = f"System_Cg{Cg}_stdR{stdR}_sig{sig}_T0_{T0}"
        sys_dir = output_dir / sys_folder_name
        sys_dir.mkdir(exist_ok=True)

        # Sort by actual physical gradient (dT) to handle uneven rep jumps safely
        data = sorted(data, key=lambda x: x['Delta_T'])

        dTs = np.array([d['Delta_T'] for d in data])
        Vth_up = np.array([d['Vth_up'] for d in data])
        Vth_down = np.array([d['Vth_down'] for d in data])
        err_up = np.array([d['err_up'] for d in data])
        err_down = np.array([d['err_down'] for d in data])

        # Filter NaNs ensuring arrays stay perfectly parallel
        valid_mask = ~np.isnan(Vth_up) & ~np.isnan(Vth_down)
        dTs = dTs[valid_mask]
        Vth_up = Vth_up[valid_mask]
        Vth_down = Vth_down[valid_mask]
        err_up = err_up[valid_mask]
        err_down = err_down[valid_mask]

        # --- RE-INTEGRATED: Remove the last noisy point ---
        if len(dTs) > 0:
            dTs = dTs[:-1]
            Vth_up = Vth_up[:-1]
            Vth_down = Vth_down[:-1]
            err_up = err_up[:-1]
            err_down = err_down[:-1]

        if len(dTs) < 4:  # Sav-Gol needs at least a few points
            print(f"Skipping {sys_folder_name} - Not enough valid threshold data.", flush=True)
            continue

        # -----------------------------------------------------
        # Dynamic Thermopower Derivative using Savitzky-Golay
        # -----------------------------------------------------
        min_window = poly_order + 1
        if min_window % 2 == 0: min_window += 1

        # 5 or 7 are safe odd maximums.
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

        # Export Unified CSV
        export_csv_path = sys_dir / "aggregated_thermopower_results.csv"
        with open(export_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Delta_T", "Vth_Up", "Vth_Down", "S_Up", "S_Down", "S_err_up", "S_err_down"])
            for idx, dt_val in enumerate(dTs):
                writer.writerow(
                    [dt_val, Vth_up[idx], Vth_down[idx], S_up[idx], S_down[idx], S_err_up[idx], S_err_down[idx]])

        # ==========================================================
        # GRAPH 1A: Threshold Voltage (Up vs Down) - WITH ERROR BARS
        # ==========================================================
        plt.figure(figsize=(10, 6))
        plt.errorbar(dTs, Vth_up, yerr=err_up, marker='o', linestyle='-', color='dodgerblue', linewidth=2,
                     label='Sweep Up', capsize=3)
        plt.errorbar(dTs, Vth_down, yerr=err_down, marker='s', linestyle='--', color='crimson', linewidth=2,
                     label='Sweep Down', capsize=3)

        plt.xlabel(r'Total Temperature Gradient $\Delta T = T_{right} - T_{left}$ ($e^2 / k_B \langle C \rangle$)',
                   fontsize=12)
        plt.ylabel(r'Threshold Voltage $V_{th}$ ($e / \langle C \rangle$) [SNR Breakout]', fontsize=12)
        plt.title(f'Threshold Voltage Hysteresis vs. Gradient\n$C_g={Cg}$, $stdR={stdR}$, $\\sigma={sig}$, $T_0={T0}$',
                  fontsize=14)
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

        plt.xlabel(r'Total Temperature Gradient $\Delta T = T_{right} - T_{left}$ ($e^2 / k_B \langle C \rangle$)',
                   fontsize=12)
        plt.ylabel(r'Threshold Voltage $V_{th}$ ($e / \langle C \rangle$) [SNR Breakout]', fontsize=12)
        plt.title(f'Threshold Voltage Hysteresis vs. Gradient\n$C_g={Cg}$, $stdR={stdR}$, $\\sigma={sig}$, $T_0={T0}$',
                  fontsize=14)
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
        plt.xlabel(r'Total Temperature Gradient $\Delta T$ ($e^2 / k_B \langle C \rangle$)', fontsize=12)
        plt.ylabel(r'Thermopower $S(T) = -dV_{th} / d(\Delta T)$ ($k_B / e$)', fontsize=12)
        plt.title(f'Thermopower Hysteresis vs. Gradient\n$C_g={Cg}$, $stdR={stdR}$, $\\sigma={sig}$, $T_0={T0}$',
                  fontsize=14)
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
        plt.xlabel(r'Total Temperature Gradient $\Delta T$ ($e^2 / k_B \langle C \rangle$)', fontsize=12)
        plt.ylabel(r'Thermopower $S(T) = -dV_{th} / d(\Delta T)$ ($k_B / e$)', fontsize=12)
        plt.title(f'Thermopower Hysteresis vs. Gradient\n$C_g={Cg}$, $stdR={stdR}$, $\\sigma={sig}$, $T_0={T0}$',
                  fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(sys_dir / 'Thermopower_S_vs_Gradient_Hysteresis_no_error.png', dpi=300)
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