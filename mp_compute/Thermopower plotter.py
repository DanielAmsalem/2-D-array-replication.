import os
import csv
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def calc_threshold_snr_interpolated(I, IErr, V):
    """
    Calculates Vth using the Continuous Signal-to-Noise Ratio (SNR) Breakout method.
    Finds the interpolated voltage where the signal dynamically breaks out of
    2x and 4x the simulation's specific error array (IErr).
    """
    if len(I) < 2:
        return np.nan

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
        return np.nan

    # Return the exact continuous midpoint threshold
    return (small_V + big_V) / 2.0


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


def run_scanner_mode(base_dir, ivs_txt_path):
    """
    MODE 1: Rapidly scans folders for duplicates and maps the physical runs.
    Outputs the log to IVs.txt.
    """
    print(f"[{ivs_txt_path.name} NOT FOUND] -> Initializing Scanner Mode...")

    # catalog structure: catalog[(stdR, sig, T0)][Cg] = {'counts': {rep: count}, 'folders': set()}
    catalog = defaultdict(lambda: defaultdict(lambda: {'counts': defaultdict(int), 'folders': set()}))

    for directory in base_dir.glob("results_*"):
        if not directory.is_dir(): continue
        run_name = directory.name.replace("results_", "")

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

            catalog[(stdR, sig, T0)][Cg]['counts'][rep] += 1
            catalog[(stdR, sig, T0)][Cg]['folders'].add(f"results_{run_name}")

    # Generate the IVs.txt report
    with open(ivs_txt_path, 'w') as f:
        for (stdR, sig, T0), cg_data in catalog.items():
            f.write(f"----- StdR={stdR} ; sig={sig} ; T0={T0} ---------\n")

            for Cg, stats in cg_data.items():
                rep_counts = stats['counts']
                all_runs = sorted(list(rep_counts.keys()))
                multiples = sorted([r for r, count in rep_counts.items() if count > 1])
                folders = sorted(list(stats['folders']))

                f.write(f"Cg : {Cg} with runs {all_runs}\n")
                f.write(f"runs with multiples : {multiples}\n")
                f.write(f"folders relevant : {', '.join(folders)}\n\n")

            f.write("-------------------------------------------\n")

    print(f"\n[DONE] Scan complete. Diagnostic file created at:\n{ivs_txt_path.absolute()}")
    print("Please review it, delete multiple runs/folders as needed, delete IVs.txt, and run this script again.")


def run_analysis_mode(base_dir):
    """
    MODE 2: Full physical extraction, plotting, and S(T) differentiation.
    """
    print("[IVs.txt FOUND] -> Clean data assumed. Initializing Analysis Mode...")

    output_dir = base_dir / "Thermopower_Analysis"
    output_dir.mkdir(exist_ok=True)

    # Key: (Cg, stdR, sig, T0) -> Value: list of dictionaries
    system_groups = defaultdict(list)

    print("Scanning for results directories...")
    for directory in base_dir.glob("results_*"):
        if not directory.is_dir():
            continue

        run_name = directory.name.replace("results_", "")
        print(f"Processing run batch: {run_name}")

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
            vth_up = calc_threshold_snr_interpolated(i_up, ierr_up, v_up)
            vth_down = calc_threshold_snr_interpolated(i_down, ierr_down, v_down)

            system_groups[sys_key].append({
                'Run_Name': run_name,
                'Repetition': rep,
                'Delta_T': params['dT'],
                'Vth_up': vth_up,
                'Vth_down': vth_down
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

        # Filter NaNs ensuring arrays stay perfectly parallel
        valid_mask = ~np.isnan(Vth_up) & ~np.isnan(Vth_down)
        dTs = dTs[valid_mask]
        Vth_up = Vth_up[valid_mask]
        Vth_down = Vth_down[valid_mask]

        if len(dTs) < 2:
            print(f"Skipping {sys_folder_name} - Not enough valid threshold data.")
            continue

        # -----------------------------------------------------
        # Dynamic Thermopower Derivative: S = -dVth / d(DeltaT)
        # -----------------------------------------------------
        dVth_up = np.diff(Vth_up)
        dVth_down = np.diff(Vth_down)
        ddTs = np.diff(dTs)

        # Protect against duplicate gradients dividing by zero
        nonzero_dT = ddTs != 0
        S_up = -dVth_up[nonzero_dT] / ddTs[nonzero_dT]
        S_down = -dVth_down[nonzero_dT] / ddTs[nonzero_dT]

        S_dT = dTs[1:][nonzero_dT]

        # Export Unified CSV
        export_csv_path = sys_dir / "aggregated_thermopower_results.csv"
        with open(export_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Delta_T", "Vth_Up", "Vth_Down", "S_Up", "S_Down"])
            for idx, dt_val in enumerate(dTs):
                s_u = S_up[idx - 1] if idx > 0 and nonzero_dT[idx - 1] else np.nan
                s_d = S_down[idx - 1] if idx > 0 and nonzero_dT[idx - 1] else np.nan
                writer.writerow([dt_val, Vth_up[idx], Vth_down[idx], s_u, s_d])

        # --- Graph 1: Threshold Voltage (Up vs Down) ---
        plt.figure(figsize=(10, 6))
        plt.plot(dTs, Vth_up, marker='o', linestyle='-', color='dodgerblue', linewidth=2, label='Sweep Up')
        plt.plot(dTs, Vth_down, marker='s', linestyle='--', color='crimson', linewidth=2, label='Sweep Down')
        plt.xlabel('Total Temperature Gradient $\\Delta T = T_{right} - T_{left}$ (K)', fontsize=12)
        plt.ylabel('Threshold Voltage $V_{th}$ (V) [SNR Breakout]', fontsize=12)
        plt.title(f'Threshold Voltage Hysteresis vs. Gradient\n$C_g={Cg}$, $stdR={stdR}$, $\\sigma={sig}$, $T_0={T0}$',
                  fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(sys_dir / 'Vth_vs_Gradient_Hysteresis.png', dpi=300)
        plt.close()

        # --- Graph 2: Thermopower S(T) (Up vs Down) ---
        plt.figure(figsize=(10, 6))
        plt.plot(S_dT, S_up, marker='o', linestyle='-', color='dodgerblue', linewidth=2, label='S(T) Up')
        plt.plot(S_dT, S_down, marker='s', linestyle='--', color='crimson', linewidth=2, label='S(T) Down')
        plt.axhline(0, color='black', linestyle='-', alpha=0.8)
        plt.xlabel('Total Temperature Gradient $\\Delta T$ (K)', fontsize=12)
        plt.ylabel('Thermopower $S(T) = -dV_{th} / d(\\Delta T)$ (V/K)', fontsize=12)
        plt.title(f'Thermopower Hysteresis vs. Gradient\n$C_g={Cg}$, $stdR={stdR}$, $\\sigma={sig}$, $T_0={T0}$',
                  fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(sys_dir / 'Thermopower_S_vs_Gradient_Hysteresis.png', dpi=300)
        plt.close()

        print(f"Exported data and generated dual-sweep graphs in: {sys_dir}")

    print("\nBatch Thermopower processing complete.")


def main():
    base_dir = Path(__file__).parent.parent
    ivs_txt_path = base_dir / "IVs.txt"

    if not ivs_txt_path.exists():
        run_scanner_mode(base_dir, ivs_txt_path)
    else:
        run_analysis_mode(base_dir)


if __name__ == '__main__':
    main()