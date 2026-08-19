import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
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

# Global Axis and Tick Formatting (0.5 pt for boxes/axes)
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5

# Font Sizes (Keeping KC's 20-25% larger ratio)
rcParams['axes.titlesize'] = 25
rcParams['axes.labelsize'] = 22
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18


# ==========================================
# 2. Main Plotting Function (Standard Setup)
# ==========================================
def plot_standard_IV(filename):
    """
    Loads the CSV for the non-midfix case, plots the IV curves (Linear and Semi-Log)
    with a color gradient based on delta_T, and exports KC-compliant PDFs.
    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Warning: Could not find {filename}.")
        return

    # 1. Enforce Voltage Cutoff V <= 1
    df = df[df.iloc[:, 0] <= 1]

    # Extract Voltage (x-axis)
    V = df.iloc[:, 0].values

    # Pre-calculate all delta_T values to establish the color normalization range
    dTs = []
    columns_to_plot = []

    # Flag to check if we hit the baseline noise floor in this specific dataset
    min_I_val = float('inf')

    for col in df.columns[1:]:
        # Allow negative 'n' values for symmetrical sweeps
        match = re.search(r'Grad_(-?\d+)_I', col)
        if match:
            n = int(match.group(1))

            # CORRECTION 1: Filter exactly for -8 < n < 8
            if -8 < n < 8:
                delta_T = n * 0.02
                dTs.append(delta_T)
                columns_to_plot.append((col, delta_T))

                # Track the absolute minimum positive current to trigger the baseline line
                positive_vals = df[col].values[df[col].values > 0]
                if len(positive_vals) > 0:
                    current_min = np.min(positive_vals)
                    if current_min < min_I_val:
                        min_I_val = current_min

    # Sort columns by Delta T to ensure proper rendering order
    columns_to_plot.sort(key=lambda x: x[1])

    # Set up the colormap (Viridis: dark purple to bright yellow)
    min_dT, max_dT = min(dTs), max(dTs)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min_dT, vmax=max_dT)

    # Create a ScalarMappable for the colorbars
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # ---------------------------------------------------------
    # PLOT 1: LINEAR I-V
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    for col_name, dT in columns_to_plot:
        I = df[col_name].values
        color = cmap(norm(dT))
        ax1.plot(V, I, color=color, linewidth=1.5, alpha=0.9)

    ax1.set_title(r'Single Island I-V Under Thermal Gradient', pad=20)
    ax1.set_xlabel(r'Voltage $\left[ \frac{e}{\langle C \rangle} \right]$', labelpad=15)
    ax1.set_ylabel(r'Current $\left[ \frac{e}{\langle R \rangle \langle C \rangle} \right]$', labelpad=15)
    ax1.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

    # Add Native Inset Colorbar (Upper Left)
    axins1 = ax1.inset_axes([0.05, 0.85, 0.35, 0.03])
    cb1 = fig1.colorbar(sm, cax=axins1, orientation="horizontal")

    # Shift ticks to top to fix collision
    cb1.ax.xaxis.set_ticks_position('top')
    # Bypassing set_label to place the text precisely to the right of the colorbar
    # Increased text size to 22 to match axis labels
    cb1.ax.text(1.05, 0.5, r'$\Delta T \left[ \frac{e^2}{k_B \langle C \rangle} \right]$',
                transform=cb1.ax.transAxes, va='center', ha='left', size=22)

    # Increased labelsize to 20 for readability and added slight padding
    cb1.ax.tick_params(labelsize=20, width=0.5, pad=5)
    cb1.outline.set_linewidth(0.5)

    plt.tight_layout()
    out1 = '2D_IV_Standard_Linear.pdf'
    plt.savefig(out1, format='pdf', bbox_inches='tight')
    plt.close(fig1)
    print(f"Generated: {out1}")

    # ---------------------------------------------------------
    # PLOT 2: SEMI-LOG I-V
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    I_0 = 0.1  # Fixed baseline to prevent sub-zero log noise

    for col_name, dT in columns_to_plot:
        I = df[col_name].values
        color = cmap(norm(dT))

        # Filter for strictly positive values to allow log calculation
        valid_mask = I > 0
        V_valid = V[valid_mask]
        I_log = np.log(I[valid_mask] / I_0)

        ax2.plot(V_valid, I_log, color=color, linewidth=1.5, alpha=0.9)

    # Baseline Accuracy Line Logic
    baseline_threshold = 10e-9
    if min_I_val <= baseline_threshold:
        # Calculate the log-scaled position of the baseline, accounting for I_0
        log_baseline = np.log(baseline_threshold / I_0)
        ax2.axhline(log_baseline, color='darkred', linestyle='--', linewidth=1.5, alpha=0.8)

    ax2.set_title(r'Single Island Semi-Log I-V Under Thermal Gradient', pad=20)
    ax2.set_xlabel(r'Voltage $\left[ \frac{e}{\langle C \rangle} \right]$', labelpad=15)
    ax2.set_ylabel(r'$\ln(I / I_0)$', labelpad=15)
    ax2.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

    # CORRECTION 2: Cut the lower bound of the Y-axis at -20
    ax2.set_ylim(bottom=-20)

    # Add Native Inset Colorbar (Upper Left)
    axins2 = ax2.inset_axes([0.45, 0.88, 0.35, 0.03])
    cb2 = fig2.colorbar(sm, cax=axins2, orientation="horizontal")

    # Shift ticks to top to fix collision
    cb2.ax.xaxis.set_ticks_position('top')
    # Bypassing set_label to place the text precisely to the right of the colorbar
    # Increased text size to 22 to match axis labels
    cb2.ax.text(1.05, 0.5, r'$\Delta T \left[ \frac{e^2}{k_B \langle C \rangle} \right]$',
                transform=cb2.ax.transAxes, va='center', ha='left', size=22)

    # Increased labelsize to 20 for readability and added slight padding
    cb2.ax.tick_params(labelsize=20, width=0.5, pad=5)
    cb2.outline.set_linewidth(0.5)

    plt.tight_layout()
    out2 = '2D_IV_SemiLog.pdf'
    plt.savefig(out2, format='pdf', bbox_inches='tight')
    plt.close(fig2)
    print(f"Generated: {out2}")


# ==========================================
# 3. Execution
# ==========================================
if __name__ == '__main__':
    # Update this variable with the exact name of your non-midfix CSV file
    target_csv = 'NEW_IV_data_reintegrated_metal_a.csv'
    plot_standard_IV(target_csv)