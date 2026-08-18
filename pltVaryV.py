import numpy as np
# NEVER IMPORT import matplotlib.subplots
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import json
import pickle
import sys
from pathlib import Path

# ==========================================================
# 0. PATH RESOLUTION FOR PICKLE
# ==========================================================
current_dir = Path(__file__).resolve().parent
compute_dir = current_dir / "mp_compute"
sys.path.append(str(compute_dir))

from define_objects import SteadyStateVaryVResult

# ==========================================================
# KC'S FIGURE GUIDELINES ENFORCEMENT
# ==========================================================
rcParams['font.family'] = 'sans-serif'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5

rcParams['axes.titlesize'] = 25
rcParams['axes.labelsize'] = 22
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 16

debug = True
S_min = 0.25

def discover_latest_varyV_runs(base_dir="."):
    base_path = Path(base_dir)
    runs_by_config = {}

    directories = sorted(
        [d for d in base_path.glob("results_*") if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    if debug: print(f"{len(directories)} directories found", flush=True)

    for folder in directories:
        run_name = str(folder).split("results_")[1]
        meta_file = folder / "checkpoint_meta.json"
        init_file = folder / f"{run_name}.json"

        if not meta_file.exists() or not init_file.exists():
            continue

        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            with open(init_file, 'r', encoding='utf-8') as q:
                init = json.load(q)

            job_name = meta.get("slurm_job_name", "")
            if "TPvaryV" not in job_name:
                continue

            gap_ratio = float(meta.get("gap_ratio", 0.0))
            Cg = int(meta.get("Cg", init.get("Cg", [2])[0]))
            is_midfix = meta.get("is_midfix", False) or ("TmidNonT0" in job_name) or ("Tmid" in job_name)
            is_flip = str(init.get("flip", "false")).strip().lower() == "true"

            config_key = (Cg, gap_ratio, is_midfix)

            if config_key not in runs_by_config:
                runs_by_config[config_key] = {'fwd': None, 'rev': None}

            if is_flip and runs_by_config[config_key]['rev'] is None:
                runs_by_config[config_key]['rev'] = folder
                print(f"--> Auto-detected Reverse Run {config_key}: {folder.name}")
            elif not is_flip and runs_by_config[config_key]['fwd'] is None:
                runs_by_config[config_key]['fwd'] = folder
                print(f"--> Auto-detected Forward Run {config_key}: {folder.name}")

        except Exception as a:
            if debug: print(f"Exception {a} while processing {folder.name}", flush=True)
            pass

    return runs_by_config


def aggregate_task_data(folder_path, output_dir):
    if folder_path is None:
        return None

    folder = Path(folder_path)
    pkl_files = list(folder.glob("varyV_raw_task*.pkl"))

    if not pkl_files:
        print(f"Warning: No varyV_raw_task PKL files found in {folder.name}")
        return None

    print(f"Aggregating {len(pkl_files)} array tasks from {folder.name}...")

    global_DeltaV = []
    global_I_base = []
    V_sweep = None
    successful_loops = 0
    failed_loops = 0

    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            if V_sweep is None:
                V_sweep = data['V_sweep']

            for res in data['results']:
                if getattr(res, 'error_count', 0) > 10:
                    failed_loops += 1
                    continue

                if np.isnan(res.DeltaV_vec).any():
                    failed_loops += 1
                    continue

                global_DeltaV.append(res.DeltaV_vec)
                global_I_base.append(res.I_baseline_vec)
                successful_loops += 1

        except Exception as e:
            print(f"Error loading {pkl_file.name}: {e}")

    print(f"Aggregation complete: {successful_loops} successful iterations, {failed_loops} dropped.")

    if successful_loops == 0:
        return None

    DeltaV_matrix = np.array(global_DeltaV)
    I_base_matrix = np.array(global_I_base)

    DeltaV_avg = np.mean(DeltaV_matrix, axis=0)
    DeltaV_err = np.std(DeltaV_matrix, axis=0) / np.sqrt(successful_loops)
    I_baseline_avg = np.mean(I_base_matrix, axis=0)

    meta_file = folder / "checkpoint_meta.json"
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    repetition = meta.get("repetition", 40)
    Cg = meta.get("Cg", 2)
    gap_ratio = float(meta.get("gap_ratio", 0.0))

    run_name = folder.name.replace("results_", "")
    init_file = folder / f"{run_name}.json"
    with open(init_file, 'r', encoding='utf-8') as q:
        init = json.load(q)

    is_flip = str(init.get("flip", "false")).strip().lower() == "true"

    T0 = 0.001
    T_std = repetition * T0 / 20
    dT_total = (7 - 1) * T_std

    # flip has dT_total -> -dT_total
    if is_flip:
        dT_total = -dT_total

    Seebeck_coeff = DeltaV_avg / dT_total
    Seebeck_err = DeltaV_err / abs(dT_total)

    df = pd.DataFrame({
        "V_baseline_(V)": V_sweep,
        "DeltaV_avg_(V)": DeltaV_avg,
        "DeltaV_err_(V)": DeltaV_err,
        "I_baseline_avg_(e/s)": I_baseline_avg,
        "Thermopower_S(V)": Seebeck_coeff,
        "Thermopower_err": Seebeck_err
    })

    direction = "Reverse" if is_flip else "Forward"
    csv_filename = output_dir / f"VaryV_rep{repetition}_Cg{Cg}_D{gap_ratio}_{direction}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved aggregated data to {csv_filename.name}")

    return {"df": df, "Cg": Cg, "dT": dT_total, "rep": repetition, "D": gap_ratio}


def plot_thermopower_varyV(fwd_folder, rev_folder, base_output_path, config_key):
    Cg, gap_ratio, is_midfix = config_key

    baseline_str = "Tmid_Baseline" if is_midfix else "T0_Baseline"
    sys_dir = base_output_path / f"Cg{Cg}_D{gap_ratio}_{baseline_str}"
    sys_dir.mkdir(parents=True, exist_ok=True)

    fwd_data = aggregate_task_data(fwd_folder, sys_dir)
    rev_data = aggregate_task_data(rev_folder, sys_dir)

    if fwd_data is None and rev_data is None:
        print(f"No valid data aggregated for {config_key}. Skipping plot.")
        return

    base_data = fwd_data if fwd_data is not None else rev_data
    rep = base_data["rep"]

    # --- Nested plotting function allowing flexible styling via `dasher` ---
    def generate_plot(plot_fwd, plot_rev, file_suffix, dasher, invert_midfix_rev=False):
        fig, ax1 = plt.subplots(figsize=(10, 7))

        color_s = 'tab:red'
        color_s_rev = 'darkorange'

        ax1.set_xlabel(r'Voltage Bias $V$ $\left[ \frac{e}{\langle C \rangle} \right]$', labelpad=15)
        ax1.set_ylabel(r'Thermopower |S(V)| $\left[ \frac{k_B}{e} \right]$', color='black', labelpad=15)

        # -----------------------------------------------
        # FORWARD SWEEP
        # -----------------------------------------------
        if plot_fwd is not None:
            df_f = plot_fwd["df"]
            dT_val_f = plot_fwd["dT"]

            if is_midfix:
                s_vals_f = df_f["Thermopower_S(V)"]
                s_label_f = r'$S(V)$ ($\Delta T > 0$)' if invert_midfix_rev else r'$S(V)$'
            else:
                s_vals_f = -df_f["Thermopower_S(V)"]
                s_label_f = r'$-S(V)$'

            V_f = df_f["V_baseline_(V)"]
            err_f = df_f["Thermopower_err"]

            if dasher:
                # Filter strictly for absolute S(V) > 0.3
                mask_f = np.abs(s_vals_f) > S_min
                V_plot_f = V_f[mask_f]
                S_plot_f = s_vals_f[mask_f]
                err_plot_f = err_f[mask_f]
                fmt_f = 'o'  # Scatter only, no connecting lines

                # Add vertical dashed lines indicating peak alignment
                for x_val in V_plot_f:
                    ax1.axvline(x=x_val, color='gray', linestyle='--', linewidth=1.0, alpha=0.5, zorder=0)
            else:
                V_plot_f = V_f
                S_plot_f = s_vals_f
                err_plot_f = err_f
                fmt_f = '-o'

            line1 = ax1.errorbar(
                V_plot_f, S_plot_f, yerr=err_plot_f,
                fmt=fmt_f, color=color_s, linewidth=1.5, elinewidth=1.5, markersize=6, capsize=3,
                label=s_label_f
            )

        # -----------------------------------------------
        # REVERSE SWEEP
        # -----------------------------------------------
        if plot_rev is not None:
            df_r = plot_rev["df"]
            dT_val_r = plot_rev["dT"]

            if is_midfix:
                if invert_midfix_rev:
                    s_vals_r = -df_r["Thermopower_S(V)"]
                    s_label_r = r'$-S(V)$ ($\Delta T < 0$)'
                else:
                    s_vals_r = df_r["Thermopower_S(V)"]
                    s_label_r = r'$S(V)$'
            else:
                s_vals_r = -df_r["Thermopower_S(V)"]
                s_label_r = r'$-S(V)$'

            V_r = df_r["V_baseline_(V)"]
            err_r = df_r["Thermopower_err"]

            if dasher:
                # Filter strictly for absolute S(V) > 0.3
                mask_r = np.abs(s_vals_r) > S_min
                V_plot_r = V_r[mask_r]
                S_plot_r = s_vals_r[mask_r]
                err_plot_r = err_r[mask_r]
                fmt_r = 's'  # Scatter only, no connecting lines

                # Add vertical dashed lines indicating peak alignment
                for x_val in V_plot_r:
                    ax1.axvline(x=x_val, color='gray', linestyle='--', linewidth=1.0, alpha=0.5, zorder=0)
            else:
                V_plot_r = V_r
                S_plot_r = s_vals_r
                err_plot_r = err_r
                fmt_r = '-s'

            line_r = ax1.errorbar(
                V_plot_r, S_plot_r, yerr=err_plot_r,
                fmt=fmt_r, color=color_s_rev, linewidth=1.5, elinewidth=1.5, markersize=6, capsize=3,
                label=s_label_r
            )

        # Apply tight X-Axis for Midfix (Metal vs SC splitting)
        if is_midfix:
            if gap_ratio == 0.0:
                ax1.set_xlim(0.8, 4)
            else:
                ax1.set_xlim(1.5, 3)

        ax1.tick_params(axis='y', length=6, width=0.5)
        ax1.tick_params(axis='x', length=6, width=0.5)

        # -----------------------------------------------
        # CURRENT I(V) AXIS
        # -----------------------------------------------
        ax2 = ax1.twinx()
        color_i = 'tab:blue'
        color_i_rev = 'purple'
        ax2.set_ylabel(r'Current $I(V)$ $\left[ \frac{e}{\langle R \rangle \langle C \rangle} \right]$', color='black',
                       labelpad=15)

        # Non-dashed line, width 2
        if plot_fwd is not None:
            df_f = plot_fwd["df"]
            label_i_f = r'$I(V)$ ($\Delta T > 0$)' if invert_midfix_rev else r'$I(V)$'
            line2, = ax2.plot(df_f["V_baseline_(V)"], df_f["I_baseline_avg_(e/s)"], linestyle='-',
                              color=color_i, linewidth=2, alpha=0.8, label=label_i_f)

        if plot_rev is not None:
            df_r = plot_rev["df"]
            label_i_r = r'$I(V)$ ($\Delta T < 0$)' if invert_midfix_rev else r'$I(V)$'
            line2_r, = ax2.plot(df_r["V_baseline_(V)"], df_r["I_baseline_avg_(e/s)"], linestyle='-',
                                color=color_i_rev, linewidth=2, alpha=0.8, label=label_i_r)

        ax2.tick_params(axis='y', length=6, width=0.5)
        for spine in ax2.spines.values():
            spine.set_linewidth(0.5)

        # -----------------------------------------------
        # TITLE
        # -----------------------------------------------
        Ec = 0.5 / Cg

        # Calculate proper title depending on if it's combined or separated
        if "Combined" in file_suffix and plot_fwd is not None:
            title_str = 'Thermopower & Current as a Function of Voltage\n' rf'$|\Delta T| = {abs(plot_fwd["dT"]) / Ec:.2f}E_c$'
        elif plot_fwd is not None:
            title_str = 'Thermopower & Current as a Function of Voltage\n' rf'$\Delta T = {plot_fwd["dT"] / Ec:.2f}E_c$'
        elif plot_rev is not None:
            title_str = 'Thermopower & Current as a Function of Voltage\n' rf'$\Delta T = {plot_rev["dT"] / Ec:.2f}E_c$'

        ax1.set_title(title_str, pad=20)

        fig.tight_layout()

        # -----------------------------------------------
        # COMBINED LEGEND EXPORT
        # -----------------------------------------------
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        # Shift the legend to the upper right for combined plots
        if "Combined" in file_suffix:
            legend_loc = 'upper right'
            bbox_anchor = (0.98, 0.98)
        else:
            legend_loc = 'upper left'
            bbox_anchor = (0.02, 0.98)

        legend_with_iv = ax1.legend(lines + lines2, labels + labels2,
                                    loc=legend_loc,
                                    bbox_to_anchor=bbox_anchor,
                                    frameon=True, edgecolor='black')
        legend_with_iv.get_frame().set_linewidth(0.5)

        dasher_suffix = "Dasher" if dasher else "Standard"
        out_file = sys_dir / f"Publication_VaryV_rep{rep}_Cg{Cg}_D{gap_ratio}_{file_suffix}_{dasher_suffix}.pdf"
        plt.savefig(out_file, format='pdf', bbox_inches='tight')
        print(f"Publication vector PDF saved to: {sys_dir.name}/{out_file.name}")
        plt.close()

    # ==========================================================
    # EXECUTION: Loop through True/False dasher configurations
    # ==========================================================
    for use_dasher in [False, True]:
        if is_midfix:
            # Generate entirely separate plots for forward and reverse sweeps
            if fwd_data is not None:
                generate_plot(plot_fwd=fwd_data, plot_rev=None, file_suffix="Forward", dasher=use_dasher)
            if rev_data is not None:
                generate_plot(plot_fwd=None, plot_rev=rev_data, file_suffix="Reverse", dasher=use_dasher)

            # Requirement 2: Midfix + SC regime (D!=0) fifth combined graph logic
            if gap_ratio != 0.0 and fwd_data is not None and rev_data is not None:
                generate_plot(
                    plot_fwd=fwd_data,
                    plot_rev=rev_data,
                    file_suffix="Combined_InvertedRev",
                    dasher=use_dasher,
                    invert_midfix_rev=True
                )
        else:
            # Legacy behavior: overlay them on a single combined plot
            generate_plot(plot_fwd=fwd_data, plot_rev=rev_data, file_suffix="Combined", dasher=use_dasher)


if __name__ == "__main__":
    print("Scanning directory for recent VaryV simulation runs...")

    runs_by_config = discover_latest_varyV_runs("./")
    OUTPUT_DIRECTORY = Path("./S_of_V_results")
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    if not runs_by_config:
        print("No valid TPvaryV runs discovered. Exiting.")
    else:
        for config_key, folders in runs_by_config.items():
            print(f"\n{'=' * 60}")
            print(f"Processing Data for: Cg={config_key[0]}, D={config_key[1]}, Midfix={config_key[2]}")
            print(f"{'=' * 60}")

            plot_thermopower_varyV(folders['fwd'], folders['rev'], OUTPUT_DIRECTORY, config_key)

        print("\nAll regimes processed and filed into /S_of_V_results successfully!")