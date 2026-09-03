import os
import sys
import csv
import re
import numpy as np
import pandas as pd
import matplotlib
import warnings
import decimal

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from collections import defaultdict
from matplotlib import rcParams
import matplotlib.colorbar as colorbar

# ==========================================
# 1. KC's Strict Formatting Guidelines
# ==========================================
# Use Myriad Pro (Ensure it is installed on your OS)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Myriad Pro', 'Arial']
# Ensure true vector font rendering in PDF
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# Global Axis and Tick Formatting (strictly 0.5 pt for boxes/axes)
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5

# Font Sizes (Updated to strictly align with guidelines)
rcParams['axes.titlesize'] = 25
rcParams['axes.labelsize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20

# --- Configuration ---
# Filter specific repetitions. Leave empty [] to plot ALL repetitions.
REP_LIST = [0, 17, 200]
list_for_string = "_".join(str(x) for x in REP_LIST)

# Voltage Plotting Window (Vmin < V < Vmax)
V_MIN = 0
V_MAX = 2

# Statistical Info for SEM Calculation
N_LOOPS = 960


# ---------------------

# ==========================================
# Data Processing & Math Functions
# ==========================================
def calculate_mad_threshold(y_array, k_factor=2.0):
    """
    Calculates the dynamic threshold using Median Absolute Deviation (MAD).
    Highly robust against large outlier spikes (true physical steps) because
    the median is anchored firmly at the baseline noise level.
    """
    diff_array = np.diff(y_array)
    abs_diff = np.abs(diff_array)

    # Calculate Median and MAD
    M = np.median(abs_diff)
    mad = np.median(np.abs(abs_diff - M))

    # Define the threshold
    return M + (k_factor * mad)


def filter_true_steps(I_array, I_err_array, n_loops, threshold):
    """
    Finds true physical Coulomb Blockade steps by checking all points.
    A point is a step if the jump size exceeds the dynamic MAD threshold AND
    the second derivative significantly exceeds the quadrature error.
    """
    valid_steps = []

    # Iterate through the array from idx = 1 to len - 2
    for idx in range(1, len(I_array) - 1):
        # Gate 1 (The Trigger): Calculate the simple jump size
        jump_size = abs(I_array[idx + 1] - I_array[idx])

        if jump_size > threshold:
            # Gate 2 (The Second Derivative): Calculate the discrete change in slope
            delta_S = abs(I_array[idx + 1] - I_array[idx]) - abs(I_array[idx] - I_array[idx - 1])

            # Gate 3 (Quadrature Error Propagation): Combine the error for those exact three points
            err_combined = np.sqrt(I_err_array[idx + 1] ** 2 + 4 * I_err_array[idx] ** 2 + I_err_array[idx - 1] ** 2
            ) / np.sqrt(n_loops)

            # Gate 4 (Statistical Confirmation): If step difference is greater than combined error
            if abs(delta_S) > err_combined:
                valid_steps.append(idx)

    return valid_steps


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
    and backward (Down) sweeps. Reverses the Down sweep so it is mapped from 0 -> Vmax
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

    # Isolate backward sweep (Vmax -> 0) and flip to (0 -> Vmax) for hysteresis math
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
    print(f"[{ivs_txt_path.name} NOT FOUND] -> Initializing Scanner Mode for {base_dir.name}...", flush=True)

    data_dir = base_dir
    if not data_dir.exists():
        print(f"ERROR: Could not find the results directory at {data_dir}")
        return

    # catalog structure: catalog[(D, stdR, sig, T0, midfix, Tmid)][Cg][flip][rep] = [folder1, folder2, ...]
    catalog = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    corrupted_folders = []

    # Search specifically inside the results/ folder
    for directory in data_dir.glob("results_*"):
        if not directory.is_dir(): continue
        run_name = directory.name.replace("results_", "")

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

            is_flip_csv = 'flip' in name.lower()
            txt_files = list(directory.glob("*.txt"))
            param_files = []

            for f in txt_files:
                # (?!\d) strictly asserts that the repetition number is NOT immediately followed by another digit
                if re.search(rf"rep{rep}(?!\d)", f.name):
                    if is_flip_csv and 'flip' in f.name.lower():
                        param_files.append(f)
                    elif not is_flip_csv and 'flip' not in f.name.lower():
                        param_files.append(f)

            if param_files:
                param_file = param_files[0]
            else:
                continue

            params = parse_params(param_file)
            D, stdR, sig, T0, Cg, flip = params['D'], params['stdR'], params['sig'], params['T0'], params['Cg'], params[
                'flip']

            # Track folder strictly grouped by physics parameters and separated internally by flip
            catalog[(D, stdR, sig, T0, midfix, Tmid)][Cg][flip][rep].append(f"results_{run_name}")

    # Generate the IVs.txt report
    with open(ivs_txt_path, 'w') as f:
        if corrupted_folders:
            f.write("===== CORRUPTED FOLDERS (INSPECT THESE) =====\n")
            for cf in corrupted_folders:
                f.write(f"results_{cf}\n")
            f.write("=============================================\n\n")

        for (D, stdR, sig, T0, midfix, Tmid), cg_data in catalog.items():
            if D != 0.0:
                f.write(
                    f"----- | D={D} | StdR={stdR} ; sig={sig} ; T0={T0} ; midfix={midfix} ; Tmid={Tmid} ---------\n")
            else:
                f.write(f"----- StdR={stdR} ; sig={sig} ; T0={T0} ; midfix={midfix} ; Tmid={Tmid} ---------\n")

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
    print(
        f"Please review it, delete multiple runs/folders as needed, delete {ivs_txt_path.name}, and run this script again.")


def run_analysis_mode(base_dir):
    """
    MODE 2: Full physical extraction, plotting of Bare I-V curves.
    """
    print(f"[IVs.txt FOUND in {base_dir.name}] -> Clean data assumed. Initializing Analysis Mode...")

    data_dir = base_dir
    if not data_dir.exists():
        print(f"ERROR: Could not find the results directory at {data_dir}")
        return

    # UNIFIED OUTPUT DIRECTORY FOR ALL COMPENDIUMS
    output_dir = Path("/home/amsalda/IV_curves/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Key: (D, Cg, stdR, sig, T0, midfix, Tmid) -> Value: list of dictionaries
    system_groups = defaultdict(list)

    # ==========================================================
    # PHASE 1: Initialize Global Aggregator for Cg Scaling (Graph 5 & 6 Set)
    # ==========================================================
    cg_scaling_data = defaultdict(list)

    print("Scanning for results directories...", flush=True)
    for directory in data_dir.glob("results_*"):
        if not directory.is_dir():
            continue

        run_name = directory.name.replace("results_", "")

        if run_name in ignored_folders:
            continue

        midfix, Tmid, is_corrupted = get_folder_midfix_info(directory, run_name)
        if is_corrupted:
            print(f"Skipping corrupted folder: {run_name}", flush=True)
            continue

        for csv_path in directory.glob("*.csv"):
            name = csv_path.name

            match_rep = re.search(r"rep(\d+)", name)
            if not match_rep:
                continue
            rep = int(match_rep.group(1))

            is_flip_csv = 'flip' in name.lower()
            txt_files = list(directory.glob("*.txt"))
            param_files = []

            for f in txt_files:
                if re.search(rf"rep{rep}(?!\d)", f.name):
                    if is_flip_csv and 'flip' in f.name.lower():
                        param_files.append(f)
                    elif not is_flip_csv and 'flip' not in f.name.lower():
                        param_files.append(f)

            if param_files:
                param_file = param_files[0]
            else:
                continue

            params = parse_params(param_file)
            sys_key = (params['D'], params['Cg'], params['stdR'], params['sig'], params['T0'], midfix, Tmid)

            # Extract both sweeps
            v_up, i_up, ierr_up, v_down, i_down, ierr_down = read_sweeps(csv_path)
            if len(v_up) == 0:
                continue

            system_groups[sys_key].append({
                'Run_Name': run_name,
                'Repetition': rep,
                'Delta_T': params['dT'],
                'Flip': params['flip'],
                'V_up': v_up,
                'I_up': i_up,
                'Ierr_up': ierr_up,
                'V_down': v_down,
                'I_down': i_down,
                'Ierr_down': ierr_down,
                'Num_Steps_Up': 0,    # Step counting deferred to Phase 2
                'Num_Steps_Down': 0   # Step counting deferred to Phase 2
            })

    # Process and Plot for each unique physical system
    for sys_key, data in system_groups.items():

        D, Cg, stdR, sig, T0, midfix, Tmid = sys_key

        # --- DYNAMIC FOLDER AND TITLE FORMATTING ---
        if D != 0.0:
            d_val = int(D) if float(D).is_integer() else D
            sys_folder_name = f"System_SC_Delta{d_val}Ec_Cg{Cg}_stdR{stdR}_sig{sig}"
            title_str = rf"$\Delta={d_val} E_c$, Cg={Cg}, stdR={stdR}, $\sigma$={sig}"
        else:
            sys_folder_name = f"System_Metal_Cg{Cg}_stdR{stdR}_sig{sig}"
            title_str = rf"Metal ($\Delta=0$), Cg={Cg}, stdR={stdR}, $\sigma$={sig}"

        if midfix:
            Tmid_ratio = Tmid / T0
            sys_folder_name += f"_Tmid{int(Tmid_ratio)}_{int(str(decimal.Decimal(str(Tmid_ratio - int(Tmid_ratio)))).replace('0.', '')) or 0}"
            title_str += f", Tmid={Tmid}"

        sys_dir = output_dir / sys_folder_name
        sys_dir.mkdir(exist_ok=True)

        # ==========================================================
        # PHASE 1: DYNAMIC CSV PATHING AND DATA LOADING (Vth Analysis)
        # ==========================================================
        vth_base_dir = Path("/home/amsalda/Vth_vs_dT_graphs/")
        vth_csv_path = vth_base_dir / sys_folder_name / "aggregated_Vth_results.csv"
        df_vth = None

        if vth_csv_path.exists():
            df_vth = pd.read_csv(vth_csv_path)
        else:
            print(f"Warning: Vth threshold data not found at {vth_csv_path}. Skipping Graph 4 (Density) for this system.")

        # PRE-FILTER DATA BEFORE COLORMAP & PLOTTING LOGIC
        if REP_LIST:
            filtered_data = [d for d in data if d['Repetition'] in REP_LIST]
        else:
            filtered_data = data

        if not filtered_data:
            continue

        # Sort dynamically maps data from most negative (flip=True) to most positive (flip=False)
        filtered_data = sorted(filtered_data, key=lambda x: x['Delta_T'])

        # Prepare colormap properties strictly based on pre-filtered Delta T ranges
        dTs = [d['Delta_T'] for d in filtered_data]
        min_dT, max_dT = min(dTs), max(dTs)
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=min_dT, vmax=max_dT)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # ==========================================================
        # MULTIPLIER LOOP: Generates graphs with and without error bars
        # ==========================================================
        for with_errors in [False, True]:
            err_suffix = "WithErr" if with_errors else "NoErr"

            # ----------------------------------------------------------
            # GRAPH 1: I-V Sweeps Up
            # ----------------------------------------------------------
            fig_up, ax_up = plt.subplots(figsize=(10, 8))

            for d in filtered_data:
                color = cmap(norm(d['Delta_T']))

                # Apply Boolean Mask for Voltage bounds V_MIN < V < V_MAX
                mask_up = (d['V_up'] > V_MIN) & (d['V_up'] < V_MAX)

                v_plot = d['V_up'][mask_up]
                i_plot = d['I_up'][mask_up]

                # Differentiate Flip runs explicitly using linestyle and label
                ls = '--' if d['Flip'] else '-'
                flip_str = " (Flip)" if d['Flip'] else ""
                lbl = rf"$\Delta T = {d['Delta_T']:.4g}${flip_str}" if REP_LIST else None

                if with_errors:
                    # Calculate true SEM from the standard deviation
                    sem_up = d['Ierr_up'][mask_up] / np.sqrt(N_LOOPS)
                    ax_up.errorbar(v_plot, i_plot, yerr=sem_up, color=color,
                                   linestyle=ls, linewidth=1.5, marker='None',
                                   capsize=2, elinewidth=0.8, label=lbl)
                else:
                    ax_up.plot(v_plot, i_plot, color=color, linewidth=1.5, linestyle=ls, label=lbl)

            ax_up.set_xlabel(r'Voltage $\left[ \frac{e}{\langle C \rangle} \right]$', labelpad=15)
            ax_up.set_ylabel(r'Current $\left[ \frac{e}{\langle R \rangle \langle C \rangle} \right]$', labelpad=15)
            ax_up.set_title(f'I-V Characteristics (Sweep Up)\n{title_str}', pad=20)
            ax_up.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

            if not REP_LIST:
                # Add Native Inset Colorbar (If no specific rep is filtered)
                axins_up = ax_up.inset_axes([0.05, 0.85, 0.35, 0.03])
                cb_up = fig_up.colorbar(sm, cax=axins_up, orientation="horizontal")
                cb_up.ax.xaxis.set_ticks_position('bottom')
                cb_up.ax.set_title(r'$\Delta T \left[ \frac{e^2}{k_B \langle C \rangle} \right]$', size=16, pad=10)
                cb_up.ax.tick_params(labelsize=14, width=0.5)
                cb_up.outline.set_linewidth(0.5)
            else:
                # Use a standard Legend instead of a colorbar for small curve counts
                handles, labels = ax_up.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_up.legend(by_label.values(), by_label.keys(),
                             title=r'$\Delta T \left[ \frac{e^2}{k_B \langle C \rangle} \right]$',
                             loc='upper left', fontsize=12, title_fontsize=14,
                             framealpha=0.9, edgecolor='gray')

            plt.tight_layout()
            out_name_up = f'IV_Curves_Up_{err_suffix}_Reps{list_for_string}.pdf'
            plt.savefig(sys_dir / out_name_up, format='pdf', bbox_inches='tight')
            plt.close(fig_up)

            # ----------------------------------------------------------
            # GRAPH 2: I-V Sweeps Down
            # ----------------------------------------------------------
            fig_down, ax_down = plt.subplots(figsize=(10, 8))

            for d in filtered_data:
                color = cmap(norm(d['Delta_T']))

                # Apply Boolean Mask for Voltage bounds V_MIN < V < V_MAX
                mask_down = (d['V_down'] > V_MIN) & (d['V_down'] < V_MAX)

                v_plot = d['V_down'][mask_down]
                i_plot = d['I_down'][mask_down]

                # Differentiate Flip runs explicitly using linestyle and label
                ls = '--' if d['Flip'] else '-'
                flip_str = " (Flip)" if d['Flip'] else ""
                lbl = rf"$\Delta T = {d['Delta_T']:.4g}${flip_str}" if REP_LIST else None

                if with_errors:
                    # Calculate true SEM from the standard deviation
                    sem_down = d['Ierr_down'][mask_down] / np.sqrt(N_LOOPS)
                    ax_down.errorbar(v_plot, i_plot, yerr=sem_down, color=color,
                                     linestyle=ls, linewidth=1.5, marker='None',
                                     capsize=2, elinewidth=0.8, label=lbl)
                else:
                    ax_down.plot(v_plot, i_plot, color=color, linewidth=1.5, linestyle=ls, label=lbl)

            ax_down.set_xlabel(r'Voltage $\left[ \frac{e}{\langle C \rangle} \right]$', labelpad=15)
            ax_down.set_ylabel(r'Current $\left[ \frac{e}{\langle R \rangle \langle C \rangle} \right]$', labelpad=15)
            ax_down.set_title(f'I-V Characteristics (Sweep Down)\n{title_str}', pad=20)
            ax_down.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

            if not REP_LIST:
                # Add Native Inset Colorbar
                axins_down = ax_down.inset_axes([0.05, 0.85, 0.35, 0.03])
                cb_down = fig_down.colorbar(sm, cax=axins_down, orientation="horizontal")
                cb_down.ax.xaxis.set_ticks_position('bottom')
                cb_down.ax.set_title(r'$\Delta T \left[ \frac{e^2}{k_B \langle C \rangle} \right]$', size=16, pad=10)
                cb_down.ax.tick_params(labelsize=14, width=0.5)
                cb_down.outline.set_linewidth(0.5)
            else:
                # Use a standard Legend instead of a colorbar for small curve counts
                handles, labels = ax_down.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_down.legend(by_label.values(), by_label.keys(),
                               title=r'$\Delta T \left[ \frac{e^2}{k_B \langle C \rangle} \right]$',
                               loc='upper left', fontsize=12, title_fontsize=14,
                               framealpha=0.9, edgecolor='gray')

            plt.tight_layout()
            out_name_down = f'IV_Curves_Down_{err_suffix}_Reps{list_for_string}.pdf'
            plt.savefig(sys_dir / out_name_down, format='pdf', bbox_inches='tight')
            plt.close(fig_down)

        # ==========================================================
        # PHASE 2: DATA MERGING, STEP COUNTING, AND ERROR PROPAGATION
        # ==========================================================
        rho_up, rho_down = [], []
        rho_err_up, rho_err_down = [], []
        dT_density = []
        Ec = 0.5/Cg

        for d in data:
            has_vth = False
            vth_up = 0.0
            vth_down = 0.0

            if df_vth is not None:
                # Find matching row in Vth dataframe
                diffs = (df_vth['Delta_T'] - d['Delta_T']).abs()
                closest_idx = diffs.idxmin()

                # Safety Check: only proceed if we found a valid dT match
                if diffs[closest_idx] < 1e-4:
                    vth_row = df_vth.loc[closest_idx]

                    vth_up = vth_row['Vth_Up']
                    vth_err_up = vth_row['Vth_err_up']
                    vth_down = vth_row['Vth_Down']
                    vth_err_down = vth_row['Vth_err_down']
                    has_vth = True

            # --- STEP COUNTING LOGIC (VTH MASKING) ---
            if has_vth:
                # Mask out sub-threshold current to isolate true plateau noise for the MAD algorithm
                mask_up = d['V_up'] >= vth_up
                mask_down = d['V_down'] >= vth_down

                active_I_up = d['I_up'][mask_up]
                active_I_down = d['I_down'][mask_down]

                # Fallback safeguard in case threshold bounds are anomalously close to max sweep
                if len(active_I_up) < 2: active_I_up = d['I_up']
                if len(active_I_down) < 2: active_I_down = d['I_down']

                thresh_up = calculate_mad_threshold(active_I_up, k_factor=2.0)
                thresh_down = calculate_mad_threshold(active_I_down, k_factor=2.0)
            else:
                # Fallback: Compute standard threshold if Vth data isn't present
                thresh_up = calculate_mad_threshold(d['I_up'], k_factor=2.0)
                thresh_down = calculate_mad_threshold(d['I_down'], k_factor=2.0)

            # Once the accurate threshold is established, verify steps on the full array
            d['Num_Steps_Up'] = len(filter_true_steps(d['I_up'], d['Ierr_up'], N_LOOPS, thresh_up)) + 1
            d['Num_Steps_Down'] = len(filter_true_steps(d['I_down'], d['Ierr_down'], N_LOOPS, thresh_down)) + 1
            # -----------------------------------------

            if has_vth:
                n_up = d['Num_Steps_Up']
                n_down = d['Num_Steps_Down']

                denom_up = 4.0 - vth_up
                denom_down = 4.0 - vth_down

                # Compute Density and Error
                if denom_up > 0 and denom_down > 0:
                    r_up = n_up / denom_up
                    r_down = n_down / denom_down

                    r_err_up = (r_up / denom_up) * vth_err_up
                    r_err_down = (r_down / denom_down) * vth_err_down

                    rho_up.append(r_up)
                    rho_down.append(r_down)
                    rho_err_up.append(r_err_up)
                    rho_err_down.append(r_err_down)
                    dT_density.append(d['Delta_T'] / Ec)

                    # ==========================================================
                    # PHASE 2: HARVEST DATA FOR Cg SCALING COMPENDIUM (Graph 5-10)
                    # Gate 1: Metal only (Delta=0)
                    # Gate 2: Repetition 0 only
                    # Gate 3: Isothermal only (Delta_T = 0)
                    # ==========================================================
                    if D == 0.0 and d['Repetition'] == 0 and abs(d['Delta_T']) < 1e-6:
                        thermal_key = f"Midfix (Tmid={Tmid})" if midfix else f"Standard (T0={T0})"
                        cg_scaling_data[thermal_key].append({
                            'Cg': Cg,
                            'Steps_Up': n_up,
                            'Steps_Down': n_down,
                            'Rho_Up': r_up,
                            'Rho_Down': r_down,
                            'Rho_Err_Up': r_err_up,
                            'Rho_Err_Down': r_err_down
                        })

        # ==========================================================
        # GRAPH 3: Number of Steps vs. Temperature Gradient (Delta T)
        # ==========================================================
        # Note: This is drawn outside the 'with_error' multiplier loop using raw 'data'
        fig_steps, ax_steps = plt.subplots(figsize=(10, 8))

        dT_vals = [d['Delta_T']/Ec for d in data]
        steps_up = [d['Num_Steps_Up'] for d in data]
        steps_down = [d['Num_Steps_Down'] for d in data]

        # Raw scatter over all data (No grouping or averaging, no alpha transparency per specs)
        ax_steps.scatter(dT_vals, steps_up, marker='o', color='crimson',
                         s=50, edgecolors='k', label='Up Sweep', zorder=2)
        ax_steps.scatter(dT_vals, steps_down, marker='s', color='dodgerblue',
                         s=50, edgecolors='k', label='Down Sweep', zorder=2)

        ax_steps.set_xlabel(r'Temperature Gradient $\Delta T/E_c$',
                            labelpad=15)
        ax_steps.set_ylabel(r'Number of Steps', labelpad=15)
        ax_steps.yaxis.set_major_locator(MaxNLocator(integer=True))

        # New specific title formatting
        steps_title = rf"Number of Steps as a Function of $\Delta T$" + f"\n{title_str}"
        ax_steps.set_title(steps_title, pad=20)

        ax_steps.grid(True, linestyle='--', linewidth=0.5, color='lightgray', zorder=1)
        ax_steps.legend(loc='upper left', fontsize=16, framealpha=0.9, edgecolor='gray')

        plt.tight_layout()
        out_name_steps = 'Steps_vs_DeltaT.pdf'
        plt.savefig(sys_dir / out_name_steps, format='pdf', bbox_inches='tight')
        plt.close(fig_steps)

        # ==========================================================
        # PHASE 3: GRAPH 4 - Density of Steps vs. Temperature Gradient
        # ==========================================================
        if df_vth is not None and len(dT_density) > 0:
            fig_density, ax_density = plt.subplots(figsize=(10, 8))

            ax_density.errorbar(dT_density, rho_up, yerr=rho_err_up, fmt='o',
                                color='crimson', markeredgecolor='k', markersize=7,
                                capsize=2, elinewidth=1.5, linestyle='None',
                                label='Up Sweep', zorder=2)

            ax_density.errorbar(dT_density, rho_down, yerr=rho_err_down, fmt='s',
                                color='dodgerblue', markeredgecolor='k', markersize=7,
                                capsize=2, elinewidth=1.5, linestyle='None',
                                label='Down Sweep', zorder=2)

            ax_density.set_xlabel(r'Temperature Gradient $\Delta T/E_c$',
                                  labelpad=15)
            ax_density.set_ylabel(r'Density of Steps $\left[ \frac{\langle C \rangle}{e} \right]$', labelpad=15)

            density_title = rf"Density of Steps as a Function of $\Delta T$" + f"\n{title_str}"
            ax_density.set_title(density_title, pad=20)

            ax_density.grid(True, linestyle='--', linewidth=0.5, color='lightgray', zorder=1)
            ax_density.legend(loc='upper left', fontsize=16, framealpha=0.9, edgecolor='gray')

            plt.tight_layout()
            out_name_density = 'Density_vs_DeltaT.pdf'
            plt.savefig(sys_dir / out_name_density, format='pdf', bbox_inches='tight')
            plt.close(fig_density)

        print(f"Generated Vector PDFs in: {sys_dir}", flush=True)

    print("\nBatch I-V Curve processing complete.")

    # ==========================================================
    # PHASE 3: GENERATE THE 6 GLOBAL Cg SCALING COMPENDIUM GRAPHS
    # ==========================================================
    if cg_scaling_data:
        print("\nGenerating Global Cg Scaling Graphs...")
        vth_base_dir = Path("/home/amsalda/Vth_vs_dT_graphs/")
        vth_base_dir.mkdir(parents=True, exist_ok=True)

        # We pre-calculate the totals and sort the data for sequential line drawing
        processed_cg_data = {}
        for t_key, items in cg_scaling_data.items():
            # Sort exactly by Cg to make plotting continuous lines work properly
            sorted_items = sorted(items, key=lambda x: x['Cg'])
            for item in sorted_items:
                # Calculating mathematical totals
                item['Steps_Total'] = item['Steps_Up'] + item['Steps_Down']
                item['Rho_Total'] = item['Rho_Up'] + item['Rho_Down']
                item['Rho_Err_Total'] = np.sqrt(item['Rho_Err_Up'] ** 2 + item['Rho_Err_Down'] ** 2)
            processed_cg_data[t_key] = sorted_items

        # Graph configurations containing strictly formatted labels
        plot_configs = [
            {'name': 'Steps_Up', 'y_key': 'Steps_Up', 'err_key': None, 'title': 'Number of Steps (Up Sweep) vs. $C_g$',
             'ylabel': 'Number of Steps', 'is_density': False},
            {'name': 'Steps_Down', 'y_key': 'Steps_Down', 'err_key': None,
             'title': 'Number of Steps (Down Sweep) vs. $C_g$', 'ylabel': 'Number of Steps', 'is_density': False},
            {'name': 'Steps_Total', 'y_key': 'Steps_Total', 'err_key': None, 'title': 'Total Number of Steps vs. $C_g$',
             'ylabel': 'Total Steps', 'is_density': False},
            {'name': 'Density_Up', 'y_key': 'Rho_Up', 'err_key': 'Rho_Err_Up',
             'title': r'Density of Steps (Up Sweep) vs. $C_g$',
             'ylabel': r'Density of Steps $\left[ \frac{\langle C \rangle}{e} \right]$', 'is_density': True},
            {'name': 'Density_Down', 'y_key': 'Rho_Down', 'err_key': 'Rho_Err_Down',
             'title': r'Density of Steps (Down Sweep) vs. $C_g$',
             'ylabel': r'Density of Steps $\left[ \frac{\langle C \rangle}{e} \right]$', 'is_density': True},
            {'name': 'Density_Total', 'y_key': 'Rho_Total', 'err_key': 'Rho_Err_Total',
             'title': r'Total Density of Steps vs. $C_g$',
             'ylabel': r'Total Step Density $\left[ \frac{\langle C \rangle}{e} \right]$', 'is_density': True}
        ]

        # Distinct markers to visually separate the different Tmid/T0 thermal profiles
        markers = ['o', 's', '^', 'D', 'v', 'p']

        for config in plot_configs:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Sort the thermal keys to ensure 'Standard' and specific 'Midfix' lines are consistently styled
            sorted_keys = sorted(processed_cg_data.keys())
            for idx, t_key in enumerate(sorted_keys):
                items = processed_cg_data[t_key]
                if not items:
                    continue

                cg_vals = [x['Cg'] for x in items]
                y_vals = [x[config['y_key']] for x in items]
                marker = markers[idx % len(markers)]

                if config['is_density']:
                    y_errs = [x[config['err_key']] for x in items]
                    ax.errorbar(cg_vals, y_vals, yerr=y_errs, fmt=f'-{marker}',
                                markersize=8, capsize=4, elinewidth=1.5, linewidth=1.5,
                                alpha=0.9, markeredgecolor='k', label=t_key)
                else:
                    ax.plot(cg_vals, y_vals, f'-{marker}', markersize=8,
                            linewidth=1.5, alpha=0.9, markeredgecolor='k', label=t_key)

            ax.set_xlabel(r'Ground Capacitance $C_g \left[ C \right]$', labelpad=15)
            ax.set_ylabel(config['ylabel'], labelpad=15)
            # Inject strict Title requirement (Metal, Delta=0, Delta T=0)
            ax.set_title(config['title'] + r' | Metal ($\Delta=0$), Isothermal ($\Delta T=0$)', pad=20)

            # Restrict step-count graphs to integer limits
            if not config['is_density']:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray', zorder=1)
            ax.legend(loc='best', fontsize=14, framealpha=0.9, edgecolor='gray')

            plt.tight_layout()
            out_filename = f"Scaling_{config['name']}_vs_Cg.pdf"
            plt.savefig(vth_base_dir / out_filename, format='pdf', bbox_inches='tight')
            plt.close(fig)

        print(f"6 Cg Scaling graphs generated successfully in: {vth_base_dir.absolute()}")


def main():
    # If run with a specific directory argument, process only that.
    # Otherwise, queue both the Normal Metal and SC compendiums automatically.
    if len(sys.argv) > 1:
        base_dirs = [Path(arg) for arg in sys.argv[1:]]
    else:
        base_dirs = [
            Path("/home/amsalda/SITresults/Thermopower_NormalMetal_compendium/results/"),
            Path("/home/amsalda/SITresults/Thermopower_SC_compendium/results/")
        ]

    for base_dir in base_dirs:
        if not base_dir.exists():
            print(f"Warning: Directory not found, skipping {base_dir}")
            continue

        print(f"\n{'=' * 60}\nProcessing Base Directory: {base_dir.name}\n{'=' * 60}")
        ivs_txt_path = base_dir / "IVs.txt"

        if not ivs_txt_path.exists():
            run_scanner_mode(base_dir, ivs_txt_path)
        else:
            run_analysis_mode(base_dir)


if __name__ == '__main__':
    main()