import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path

# ==========================================================
# KC'S FIGURE GUIDELINES ENFORCEMENT (Strict Parameterization)
# ==========================================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'

# "I like to have the axes and boxes surrounding a plot NOT stand out... I use 0.5 pt"
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.minor.width'] = 0.5

# "20 pt for the numbers, 25 pt for the axis labels"
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 16


def discover_latest_runs(base_dir="."):
    """
    Scans the base directory for 'results_*' folders.
    Sorts them by newest first. Reads 'checkpoint_meta.json' to ensure it is a
    fixed bias run, and determines if it's the forward or reverse sweep.
    """
    base_path = Path(base_dir)
    fwd_folder = None
    rev_folder = None

    # Sort folders by modification time (newest first)
    directories = sorted(
        [d for d in base_path.glob("results_*") if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    for folder in directories:
        # Stop searching if we already found the latest of both
        if fwd_folder is not None and rev_folder is not None:
            break

        # ONLY look at the explicitly created checkpoint metadata
        meta_file = folder / "checkpoint_meta.json"

        if not meta_file.exists():
            continue  # Skip folders without metadata

        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Filter out standard I(V) curve results that don't have 'fixedBias' in the job name
            job_name = data.get("slurm_job_name", "")
            if "fixedBias" not in job_name:
                continue

            # Check if it's the forward or reverse run based on our metadata format
            if "is_reverse" in data:
                is_flip = str(data["is_reverse"]).strip().lower() == "true"

                if is_flip and rev_folder is None:
                    rev_folder = folder
                    print(f"--> Auto-detected Reverse Run (is_reverse=True): {folder.name}")
                elif not is_flip and fwd_folder is None:
                    fwd_folder = folder
                    print(f"--> Auto-detected Forward Run (is_reverse=False): {folder.name}")

        except Exception:
            # Silently pass over unreadable or corrupted JSON files
            pass

    return fwd_folder, rev_folder


def load_and_split_hysteresis_data(folder_path):
    """
    Reads CSV files, aggregates by step_idx to preserve chronology,
    and splits the data into chronological Up-Sweep and Down-Sweep DataFrames.
    """
    if folder_path is None:
        return None, None

    folder = Path(folder_path)
    csv_files = list(folder.glob("fixed_bias_task*.csv"))

    if not csv_files:
        print(f"Warning: No fixed_bias_task CSV files found in {folder.name}")
        return None, None

    print(f"Aggregating {len(csv_files)} batch files from {folder.name}...")

    df_list = [pd.read_csv(f) for f in csv_files]
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"there are {len(df_list)} csv's found", flush=True)

    if "step_idx" not in full_df.columns:
        raise ValueError(
            "CRITICAL: 'step_idx' column missing. This script requires the updated up-and-down sweep sequence.")

    # ==========================================================
    # FIX 1: Group strictly by step_idx to perfectly merge all 48 tasks
    # ==========================================================
    grouped = full_df.groupby("step_idx").agg({
        "delta_T": "mean",
        "I_avg": ["mean", "std", "count"]
    })

    # Flatten multi-level column names from the agg function
    grouped.columns = ["delta_T", "mean", "std", "count"]
    grouped = grouped.reset_index()

    # Calculate Standard Error of the Mean (SEM) = std / sqrt(N)
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['sem'] = grouped['sem'].fillna(0)

    # 2. Sort chronologically by step
    grouped = grouped.sort_values("step_idx").reset_index(drop=True)

    # ==========================================================
    # FIX 2: Find peak magnitude using absolute value so negative sweeps split properly
    # ==========================================================
    peak_idx = grouped['delta_T'].abs().idxmax()

    # Slicing with .loc safely includes the peak in BOTH arrays so the plotted line connects flawlessly
    sweep_up = grouped.loc[:peak_idx]
    sweep_down = grouped.loc[peak_idx:]

    return sweep_up, sweep_down


def plot_thermopower_current(fwd_folder, rev_folder, output_path):
    # Unpack the Up and Down sweeps for both physical gradients
    fwd_up, fwd_down = load_and_split_hysteresis_data(fwd_folder)
    rev_up, rev_down = load_and_split_hysteresis_data(rev_folder)

    if fwd_up is None and rev_up is None:
        print("CRITICAL ERROR: No data found to plot. Exiting.")
        return

    Ec = 0.05
    line_width = 2.0  # Slightly thicker line since markers are removed

    # Base title and y-axis strings
    base_title = r"$I(\Delta T)$ Current Under Temperature Gradient at $V=0$"
    y_label = r"Current $\left[ \frac{e}{\langle R \rangle \langle C \rangle} \right]$"

    # ==========================================================
    # GRAPH 1: Positive Gradient Alone (\Delta T > 0)
    # ==========================================================
    if fwd_up is not None:
        fig1, ax1 = plt.subplots(figsize=(12, 8))

        ax1.errorbar(
            fwd_up["delta_T"] / Ec, fwd_up["mean"], yerr=fwd_up["sem"],
            marker='None', linestyle='-', color='crimson', linewidth=line_width,
            capsize=4, elinewidth=1.5, capthick=1.0, label=r'$\Delta T > 0$ (Increasing)'
        )
        ax1.errorbar(
            fwd_down["delta_T"] / Ec, fwd_down["mean"], yerr=fwd_down["sem"],
            marker='None', linestyle='--', color='dodgerblue', linewidth=line_width,
            capsize=4, elinewidth=1.5, capthick=1.0, label=r'$\Delta T > 0$ (Decreasing)'
        )

        ax1.set_title(base_title, fontsize=28, pad=20)
        ax1.set_xlabel(r"$\Delta T/E_c$", labelpad=15)
        ax1.set_ylabel(y_label, labelpad=15)
        ax1.legend(loc="best")
        ax1.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)

        plt.tight_layout()
        out_file1 = Path(output_path) / "Thermopower_I_vs_deltaT_fixedV0_pos.pdf"
        plt.savefig(out_file1, format='pdf', bbox_inches='tight')
        plt.close(fig1)
        print(f"Generated: {out_file1.name}")

    # ==========================================================
    # GRAPH 2: Negative Gradient Alone (\Delta T < 0) mapped to +X
    # ==========================================================
    if rev_up is not None:
        fig2, ax2 = plt.subplots(figsize=(12, 8))

        ax2.errorbar(
            rev_up["delta_T"] / Ec, rev_up["mean"], yerr=rev_up["sem"],
            marker='None', linestyle='-', color='darkorange', linewidth=line_width,
            capsize=4, elinewidth=1.5, capthick=1.0, label=r'$\Delta T < 0$ (Increasing magnitude)'
        )
        ax2.errorbar(
            rev_down["delta_T"] / Ec, rev_down["mean"], yerr=rev_down["sem"],
            marker='None', linestyle='--', color='purple', linewidth=line_width,
            capsize=4, elinewidth=1.5, capthick=1.0, label=r'$\Delta T < 0$ (Decreasing magnitude)'
        )

        ax2.set_title(base_title, fontsize=28, pad=20)
        ax2.set_xlabel(r"$|\Delta T|/E_c$", labelpad=15)
        ax2.set_ylabel(y_label, labelpad=15)
        ax2.legend(loc="best")
        ax2.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)

        plt.tight_layout()
        out_file2 = Path(output_path) / "Thermopower_I_vs_deltaT_fixedV0_neg.pdf"
        plt.savefig(out_file2, format='pdf', bbox_inches='tight')
        plt.close(fig2)
        print(f"Generated: {out_file2.name}")

    # ==========================================================
    # GRAPH 3: Combined Overlay (Mapped to Absolute Magnitude)
    # ==========================================================
    if fwd_up is not None and rev_up is not None:
        fig3, ax3 = plt.subplots(figsize=(12, 8))

        # Plot Forward
        ax3.errorbar(
            fwd_up["delta_T"] / Ec, fwd_up["mean"], yerr=fwd_up["sem"],
            marker='None', linestyle='-', color='crimson', linewidth=line_width,
            capsize=4, elinewidth=1.5, capthick=1.0, label=r'$\Delta T > 0$ (Increasing)'
        )
        ax3.errorbar(
            fwd_down["delta_T"] / Ec, fwd_down["mean"], yerr=fwd_down["sem"],
            marker='None', linestyle='--', color='dodgerblue', linewidth=line_width,
            capsize=4, elinewidth=1.5, capthick=1.0, label=r'$\Delta T > 0$ (Decreasing)'
        )

        # Plot Reverse mapped to positive X
        ax3.errorbar(
            rev_up["delta_T"] / Ec, rev_up["mean"], yerr=rev_up["sem"],
            marker='None', linestyle='-', color='darkorange', linewidth=line_width,
            capsize=4, elinewidth=1.5, capthick=1.0, label=r'$\Delta T < 0$ (Increasing magnitude)'
        )
        ax3.errorbar(
            rev_down["delta_T"] / Ec, rev_down["mean"], yerr=rev_down["sem"],
            marker='None', linestyle='--', color='purple', linewidth=line_width,
            capsize=4, elinewidth=1.5, capthick=1.0, label=r'$\Delta T < 0$ (Decreasing magnitude)'
        )

        ax3.set_title(base_title, fontsize=28, pad=20)
        ax3.set_xlabel(r"$|\Delta T|/E_c$", labelpad=15)
        ax3.set_ylabel(y_label, labelpad=15)

        # Explicitly enforce Top-Left positioning for the combined legend
        ax3.legend(loc="upper left")
        ax3.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)

        plt.tight_layout()
        out_file3 = Path(output_path) / "Thermopower_I_vs_deltaT_fixedV0_combined.pdf"
        plt.savefig(out_file3, format='pdf', bbox_inches='tight')
        plt.close(fig3)
        print(f"Generated: {out_file3.name}")


if __name__ == "__main__":
    # ==========================================================
    # AUTOMATIC RUN DISCOVERY
    # ==========================================================
    print("Scanning directory for recent simulation runs...")

    FORWARD_RUN_DIR, REVERSE_RUN_DIR = discover_latest_runs("./")

    OUTPUT_DIRECTORY = "./"

    plot_thermopower_current(FORWARD_RUN_DIR, REVERSE_RUN_DIR, OUTPUT_DIRECTORY)