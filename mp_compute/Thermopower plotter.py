import os
import csv
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def exact_poly_vth(v_arr, i_arr, degree=3, threshold=1e-6):
    """
    Extracts the highly precise Vth using an exact polynomial interpolation
    around the first point that breaches the threshold.
    """
    mask = i_arr > threshold
    if not mask.any():
        return np.nan
    idx = np.argmax(mask)

    # Define exact nodes based on polynomial degree
    if degree == 1:
        indices = [idx - 1, idx]
    elif degree == 3:
        indices = [idx - 1, idx, idx + 1, idx + 2]
    else:
        return np.nan

    # Ensure indices stay within array bounds
    indices = [i for i in indices if 0 <= i < len(i_arr)]
    if len(indices) != degree + 1:
        return np.nan

    v_sub = v_arr[indices]
    i_sub = i_arr[indices]

    # Exact fit mapping I to Vl, extracting the root at I=0
    coeffs = np.polyfit(i_sub, v_sub, degree)
    poly = np.poly1d(coeffs)
    return poly(0)


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

        # Define gradT strictly as T_right - T_left
        params['dT'] = params['T_right'] - params['T_left']
    else:
        params['dT'] = 0.0

    return params


def read_forward_sweep(csv_path):
    """
    Safely reads V and I vectors from the CSV file and isolates the forward sweep.
    """
    v_col, i_col = [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                v = float(row[0])
                i = float(row[1])
                v_col.append(v)
                i_col.append(i)
            except ValueError:
                pass  # Skip non-numeric headers

    if not v_col:
        return np.array([]), np.array([])

    # Isolate forward sweep (up to max voltage)
    idx_max = v_col.index(max(v_col))
    v_forward = np.array(v_col[:idx_max + 1])
    i_forward = np.array(i_col[:idx_max + 1])

    return v_forward, i_forward


def main():
    base_dir = Path(__file__).parent.parent

    output_dir = base_dir / "Thermopower_Analysis"
    output_dir.mkdir(exist_ok=True)

    # Dictionary to hold grouped data based on physical parameters
    # Key: (Cg, stdR, sig, T0) -> Value: list of dictionaries mapping rep, dT, Vth
    system_groups = defaultdict(list)

    print("Scanning for results directories...")
    # Find all generated results folders
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

            # Locate the exact corresponding parameter file
            param_file = directory / f"parameters_{run_name}_rep{rep}.txt"
            if not param_file.exists():
                # Fallback search if exact name slightly varies
                param_files = list(directory.glob(f"*rep{rep}*.txt"))
                if param_files:
                    param_file = param_files[0]
                else:
                    print(f"Warning: No parameter file found for {name}. Skipping...")
                    continue

            # Parse parameters to establish physical properties and dT
            params = parse_params(param_file)
            sys_key = (params['Cg'], params['stdR'], params['sig'], params['T0'])

            # Retrieve the forward IV trace
            v_forward, i_forward = read_forward_sweep(csv_path)
            if len(v_forward) == 0:
                continue

            # Extract high-precision Vth
            # Fallback to linear if cubic throws NaN (for very sharp features)
            vth_cubic = exact_poly_vth(v_forward, i_forward, degree=3, threshold=1e-6)
            if np.isnan(vth_cubic):
                vth = exact_poly_vth(v_forward, i_forward, degree=1, threshold=1e-6)
            else:
                vth = vth_cubic

            system_groups[sys_key].append({
                'Run_Name': run_name,
                'Repetition': rep,
                'Delta_T': params['dT'],
                'Vth': vth
            })

    # Consolidate and evaluate logic for each uniquely identified physical system
    for sys_key, data in system_groups.items():
        if len(data) < 2:
            print(f"Skipping System {sys_key} (Insufficient data points)")
            continue

        Cg, stdR, sig, T0 = sys_key
        sys_folder_name = f"System_Cg{Cg}_stdR{stdR}_sig{sig}_T0_{T0}"
        sys_dir = output_dir / sys_folder_name
        sys_dir.mkdir(exist_ok=True)

        # Sort entirely by the actual physical gradient Delta T (cross-directory safe)
        data = sorted(data, key=lambda x: x['Delta_T'])

        # Aggregate arrays
        dTs = np.array([d['Delta_T'] for d in data])
        Vths = np.array([d['Vth'] for d in data])

        # Remove stray NaNs
        valid_mask = ~np.isnan(Vths)
        dTs = dTs[valid_mask]
        Vths = Vths[valid_mask]

        if len(dTs) < 2:
            continue

        # -----------------------------------------------------
        # Dynamic Thermopower Derivative: S(T) = -dVth / d(dT)
        # -----------------------------------------------------
        dVths = np.diff(Vths)
        ddTs = np.diff(dTs)

        # Protect against duplicate gradients dividing by zero
        nonzero_dT = ddTs != 0
        S = -dVths[nonzero_dT] / ddTs[nonzero_dT]

        # Anchor the derivative securely to the step
        S_dT = dTs[1:][nonzero_dT]

        # Export unified CSV log of the calculations
        export_csv_path = sys_dir / "aggregated_thermopower_results.csv"
        with open(export_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Delta_T", "Vth", "S(T)"])
            for idx, dt_val in enumerate(dTs):
                s_val = S[idx - 1] if idx > 0 and nonzero_dT[idx - 1] else np.nan
                writer.writerow([dt_val, Vths[idx], s_val])

        # --- Graph 1: Threshold Voltage (Vth) vs Gradient ---
        plt.figure(figsize=(10, 6))
        plt.plot(dTs, Vths, marker='o', linestyle='-', color='dodgerblue', linewidth=2, markersize=7)
        plt.xlabel('Total Temperature Gradient $\\Delta T = T_{right} - T_{left}$ (K)', fontsize=12)
        plt.ylabel('Extrapolated Threshold Voltage $V_{th}$ (V)', fontsize=12)
        plt.title(f'Threshold Voltage vs. Gradient\n$C_g={Cg}$, $stdR={stdR}$, $sig={sig}$, $T_0={T0}$', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(sys_dir / 'Vth_vs_Gradient.png', dpi=300)
        plt.close()

        # --- Graph 2: Thermopower S(T) vs Gradient ---
        plt.figure(figsize=(10, 6))
        plt.plot(S_dT, S, marker='s', linestyle='-', color='crimson', linewidth=2, markersize=7)
        plt.axhline(0, color='black', linestyle='--', alpha=0.7)
        plt.xlabel('Total Temperature Gradient $\\Delta T$ (K)', fontsize=12)
        plt.ylabel('Thermopower $S(T) = -dV_{th} / d(\\Delta T)$ (V/K)', fontsize=12)
        plt.title(f'Thermopower $S(T)$ vs. Gradient\n$C_g={Cg}$, $stdR={stdR}$, $sig={sig}$, $T_0={T0}$', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(sys_dir / 'Thermopower_S_vs_Gradient.png', dpi=300)
        plt.close()

        print(f"Exported data and generated graphs in: {sys_dir}")

    print("\nBatch Thermopower processing complete.")


if __name__ == '__main__':
    main()