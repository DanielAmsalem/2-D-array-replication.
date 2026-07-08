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
from scipy.signal import find_peaks


def average_diff(x, window_size=5):
    if window_size < 5:
        window_size = 5
    if 2 * window_size > len(x):
        window_size = len(x) // 2
    if window_size == 0:
        return np.diff(x)

    def preper_kernel(X, window_size_padding):
        kernel_padding = (1 / window_size_padding) * np.ones((2 * window_size_padding,))
        kernel_padding[window_size_padding:] *= -1
        padder = np.pad(X, window_size_padding - 1, 'edge')
        return padder, kernel_padding

    padded, kernel = preper_kernel(x, window_size)
    result = np.convolve(padded, kernel, mode='valid')
    return result


def calculate_dynamic_threshold(diff_array, factor, absolute):
    """
    Calculates the adaptive noise threshold based on the standard deviation
    of the smaller fluctuations in the derivative.
    """
    # Isolate 'small' noise (below the average derivative magnitude)
    small_noise = diff_array[np.logical_and(diff_array > 0, diff_array < np.average(diff_array))]

    if len(small_noise) < 1:
        return absolute

    return max(factor * np.std(small_noise), absolute)


def find_jump_positions_and_first_jump_size(I_diff, threshold, diff_err):
    """
    Uses find_peaks to identify the first significant instability and returns
    the magnitude of that jump, its error, and all positive peak indices.
    """
    peaks, properties = find_peaks(I_diff, height=threshold, prominence=0.5 * threshold)

    if len(peaks) > 0:
        # Filter the peaks array to keep ONLY the indices where I_diff is positive
        # This acts exactly like your for-loop but is executed instantly in C
        positive_peaks_mask = I_diff[peaks] > 0

        positive_peaks = peaks[positive_peaks_mask]
        positive_heights = properties["peak_heights"][positive_peaks_mask]

        if len(positive_peaks) > 0:
            first_peak_idx = positive_peaks[0]
            jump_size_error = diff_err[first_peak_idx]

            return positive_heights[0], jump_size_error, positive_peaks, positive_heights

    return np.nan, np.nan, np.array([])


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
    match_flip = re.search(r'flip\s*:\s*(True|False)', content)

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
    and backward (Down) sweeps. Reverses the Down sweep so it is mappedn from 0 -> Vmax
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
    print(f"[{ivs_txt_path.name} NOT FOUND] -> Initializing Scanner Mode...", flush=True)

    # Point to the actual results folder securely
    data_dir = base_dir / "results"
    if not data_dir.exists():
        print(f"ERROR: Could not find the results directory at {data_dir}")
        return

    # catalog structure: catalog[(stdR, sig, T0, midfix, Tmid)][Cg][flip][rep] = [folder1, folder2, ...]
    catalog = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    corrupted_folders = []

    # Search specifically inside the results/ folder
    for directory in data_dir.glob("results_*"):
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
            stdR, sig, T0, Cg, flip = params['stdR'], params['sig'], params['T0'], params['Cg'], params['flip']

            # Track folder strictly grouped by physics parameters and separated internally by flip
            catalog[(stdR, sig, T0, midfix, Tmid)][Cg][flip][rep].append(f"results_{run_name}")

    # Generate the IVs.txt report in the base_dir
    with open(ivs_txt_path, 'w') as f:
        if corrupted_folders:
            f.write("===== CORRUPTED FOLDERS (INSPECT THESE) =====\n")
            for cf in corrupted_folders:
                f.write(f"results_{cf}\n")
            f.write("=============================================\n\n")

        for (stdR, sig, T0, midfix, Tmid), cg_data in catalog.items():
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
    print("Please review it, delete multiple runs/folders as needed, delete IVs.txt, and run this script again.")


def run_analysis_mode(base_dir):
    """
    MODE 2: Full physical extraction and grouping of Hysteresis Curves.
    """
    print("[IVs.txt FOUND] -> Clean data assumed. Initializing Analysis Mode...")

    # Point to the actual results folder for reading
    data_dir = base_dir / "results"
    if not data_dir.exists():
        print(f"ERROR: Could not find the results directory at {data_dir}")
        return

    # Set exact target output folder as requested
    output_dir = base_dir / "IV_curve_Analysis_Normal_Metal"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Key: (Cg, stdR, sig, T0, midfix, Tmid) -> Value: list of dictionaries
    system_groups = defaultdict(list)

    print("Scanning for results directories...", flush=True)
    for directory in data_dir.glob("results_*"):
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

            match_rep = re.search(r"rep(\d+)", name)
            if not match_rep:
                continue
            rep = int(match_rep.group(1))

            param_file = directory / f"parameters_{run_name}_rep{rep}.txt"
            if not param_file.exists():
                param_files = list(directory.glob(f"*rep{rep}*.txt"))
                if param_files:
                    param_file = param_files[0]
                else:
                    print(f"Warning: No param file found for {name}. Skipping...")
                    continue

            params = parse_params(param_file)
            sys_key = (params['Cg'], params['stdR'], params['sig'], params['T0'], midfix, Tmid)

            # Extract both sweeps
            v_up, i_up, ierr_up, v_down, i_down, ierr_down = read_sweeps(csv_path)
            if len(v_up) == 0:
                continue

            # We store the raw curves for Hysteresis calculations
            system_groups[sys_key].append({
                'Run_Name': run_name,
                'Repetition': rep,
                'Delta_T': params['dT'],
                'V_up': v_up,
                'I_up': i_up,
                'IErr_up': ierr_up,
                'V_down': v_down,
                'I_down': i_down,
                'IErr_down': ierr_down
            })

    # Number of repetitions to put on a single output graph to avoid crowding
    CHUNK_SIZE = 6

    # Process and Plot for each unique physical system
    for sys_key, data in system_groups.items():
        if len(data) == 0:
            continue

        Cg, stdR, sig, T0, midfix, Tmid = sys_key

        # Construct specific output folder names
        sys_folder_name = f"System_Cg{Cg}_stdR{stdR}_sig{sig}_T0_{T0}"
        if midfix:
            Tmid_ratio = Tmid / T0
            Tmid_units = int(Tmid_ratio)
            decimal_part = decimal.Decimal(str(Tmid_ratio - Tmid_units))
            decimal_str = str(decimal_part).replace('0.', '')
            Tmid_pastdigits = int(decimal_str) if decimal_str else 0
            sys_folder_name += f"_Tmid{Tmid_units}_{Tmid_pastdigits}"

        sys_dir = output_dir / sys_folder_name
        sys_dir.mkdir(exist_ok=True)

        # Sort the data dynamically by repetition to easily group sequential sweeps
        data = sorted(data, key=lambda x: x['Repetition'])

        # Create Unified physical Title String
        title_str = f"Cg={Cg}, stdR={stdR}, $\\sigma$={sig}, T0={T0}"
        if midfix:
            title_str += f", Tmid={Tmid}"

        Volt_str = r"$\left [ \frac{e}{\langle C \rangle } \right ]$"
        Amp_str = r"$\left [ \frac{e}{\langle R \rangle \langle C \rangle} \right ]$"
        VdotA_str = r"$\left [ \frac{e^2}{\langle R \rangle \langle C \rangle ^2} \right ]$"
        T_str = r"$\left [ \frac{e^2}{k_B \langle C \rangle} \right ]$"

        # Initialize the CSV writers for the extracted data points
        loop_area_path = sys_dir / "Loop_area.csv"
        first_jump_path = sys_dir / "First_jump.csv"

        with open(loop_area_path, 'w', newline='') as f_area, open(first_jump_path, 'w', newline='') as f_jump:
            writer_area = csv.writer(f_area)
            writer_jump = csv.writer(f_jump)
            writer_area.writerow(["Repetition", "Delta_T", "Loop_Area", "Error"])
            writer_jump.writerow(["Repetition", "Delta_T", "First_Jump_Size", "Err"])

            # Break the massive lists of curves into legible chunks of CHUNK_SIZE
            chunks = [data[i:i + CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]

            for chunk in chunks:
                k = 0
                min_rep = min(d['Repetition'] for d in chunk)
                max_rep = max(d['Repetition'] for d in chunk)

                plt.figure(figsize=(10, 6))
                colormap = plt.cm.plasma

                for idx, d in enumerate(chunk):
                    vertical_offset = 0.3
                    # Direct array extraction. V_up and V_down are identical per our parsing.
                    # We implement a safe truncation just in case an edge-case file dropped a single row.
                    min_len = min(len(d['V_up']), len(d['V_down']))

                    v_u = d['V_up'][:min_len]
                    i_u = d['I_up'][:min_len]
                    ierr_u = d['IErr_up'][:min_len]

                    i_d = d['I_down'][:min_len]
                    ierr_d = d['IErr_down'][:min_len]

                    rep = d['Repetition']
                    dT = d['Delta_T']

                    # Calculate precise Hysteresis Delta I through exact subtraction
                    i_diff = i_d - i_u
                    joint_error = np.sqrt(ierr_u ** 2 + ierr_d ** 2)
                    dV = np.abs(np.mean(np.diff(v_u)))

                    # Extract the Loop Area (integration of the hysteresis via trapezoidal rule)
                    area = np.trapz(i_diff, v_u)
                    dA = np.mean(joint_error) * dV
                    writer_area.writerow([rep, dT, area, dA])

                    # Calculate first jump size
                    # Kasirer uses this to identify 'spikes' in current change
                    diff = average_diff(i_diff, window_size=5)

                    # noise threshold for this specific run
                    # 0.003 threshold is from thesis calc for single Isle
                    # we have multiple isles which increases average noise, I went with 0.01
                    thresh = calculate_dynamic_threshold(diff, factor=10, absolute=0.01)

                    # Extract the size of the first jump
                    jump_size, jump_err, jump_positions, jump_heights = find_jump_positions_and_first_jump_size(diff, thresh,
                                                                                                  joint_error)

                    # 4. Save to your First_jump.csv
                    writer_jump.writerow([rep, dT, jump_size, jump_err])

                    # Plot this specific loop onto the chunked graph
                    color = colormap(idx / max(1, len(chunk) - 1))
                    i_diff_corrected = i_diff + k * vertical_offset
                    plt.errorbar(v_u, i_diff_corrected, yerr=joint_error, fmt='-',
                                 label=f"Rep={rep}, $\\Delta T$={dT:.4g}", linewidth=1.5, capsize=3, elinewidth=1,
                                 alpha=0.8)
                    k += 1

                # Format the graph
                plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
                plt.xlabel('Voltage ' + Volt_str, fontsize=12)
                plt.ylabel(r'Hysteresis $\Delta I = I_{dec} - I_{inc}$ ' + Amp_str, fontsize=12)
                plt.title(f'Hysteresis vs. Voltage\n{title_str}', fontsize=14)

                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize='small')
                plt.grid(True, linestyle='--', alpha=0.6)

                # Output Chunked Image
                plt.savefig(sys_dir / f'Hyst_V_Curve_rep{min_rep}_{max_rep}.png', dpi=300, bbox_inches='tight')
                plt.close()

        # 1. Read the data back from the CSV we just populated
        area_data = []
        jump_data = []
        with open(loop_area_path, 'r') as f_read, open(first_jump_path, 'r') as f_jump:
            reader_loop = csv.reader(f_read)
            reader_jump = csv.reader(f_jump)
            next(reader_loop)  # Skip headers
            next(reader_jump)
            for row in reader_loop:
                # row: [Repetition, Delta_T, Loop_Area, Area_Error]
                area_data.append([float(row[1]), float(row[2]), float(row[3])])
            for row in reader_jump:
                # row: [Repetition, Delta_T, JumpSize, JumpError]
                jump_data.append([float(row[1]), float(row[2]), float(row[3])])
        area_data = np.array(area_data)
        jump_data = np.array(jump_data)

        # Sort by Delta_T (column 0) to ensure the line plot connects correctly
        sort_idx_area = np.argsort(area_data[:, 0])
        sorted_dT_area = area_data[sort_idx_area, 0]
        sorted_area = area_data[sort_idx_area, 1]
        sorted_dA = area_data[sort_idx_area, 2]

        # Same for jump data
        sort_idx_jump = np.argsort(jump_data[:, 0])
        sorted_dT_jump = jump_data[sort_idx_jump, 0]
        sorted_jump = jump_data[sort_idx_jump, 1]
        sorted_dJ = jump_data[sort_idx_jump, 2]

        # Plot Loop Area vs Delta_T
        plt.figure(figsize=(10, 6))
        plt.errorbar(sorted_dT_area, sorted_area, yerr=sorted_dA, fmt='-o',
                     label='Loop Area', linewidth=1.5, capsize=3, elinewidth=1, alpha=0.8)

        plt.xlabel(r'Temperature Gradient $\Delta T$ ' + T_str, fontsize=12)
        plt.ylabel(r'Hysteresis Loop Area ' + VdotA_str, fontsize=12)
        plt.title(f'Hysteresis Loop Area vs. Gradient\n{title_str}', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(sys_dir / 'Loop_Area_vs_Gradient.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot Jump Size vs Delta_T
        plt.figure(figsize=(10, 6))
        plt.errorbar(sorted_dT_jump, sorted_jump, yerr=sorted_dJ, fmt='-o',
                     label='Jump Size', linewidth=1.5, capsize=3, elinewidth=1, alpha=0.8)
        plt.xlabel(r'Temperature Gradient $\Delta T$ ' + T_str, fontsize=12)
        plt.ylabel(r'Jump Size ' + Amp_str, fontsize=12)
        plt.title(f'Jump Size vs. Gradient\n{title_str}', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(sys_dir / 'Jump_Size_vs_Gradient.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Generated Hysteresis chunked graphs and CSV extracts in: {sys_dir}", flush=True)

    print("\nBatch Hysteresis processing complete.")


def main():
    # Explicitly set the base directory so it works securely regardless of where it is executed from
    base_dir = Path("/home/amsalda/SITresults/Thermopower_NormalMetal_compendium")
    ivs_txt_path = Path("/home/amsalda/IVs.txt")

    if not ivs_txt_path.exists():
        run_scanner_mode(base_dir, ivs_txt_path)
    else:
        run_analysis_mode(base_dir)


if __name__ == '__main__':
    main()
