import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg for standard local preview
import matplotlib.pyplot as plt
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

# 1. Pathing and Loading Data
sys_dir = Path('.')  # Change this to your target directory if needed
file_name = "aggregated_Vth_results_Cg10_nonfix.csv"
df = pd.read_csv(file_name)

dTs = df['Delta_T']
Vth_up = df['Vth_Up']
Vth_down = df['Vth_Down']

# Convert Standard Deviations to SEM
N = 960
err_up = df['Vth_err_up'] / np.sqrt(N)
err_down = df['Vth_err_down'] / np.sqrt(N)

# 2. Define the theoretical model (Vectorized Piecewise)
def model_func(dT, a, b, c , q):
    return np.where(
        dT > 0,
        a * (np.exp(-np.abs(dT+0.001) / (2*b)) + c * dT),
        a * (np.exp(-np.abs(dT+0.001) / (2*b)) + q * dT)
    )

# 3. Setting Manual Parameters
Ec =0.025
Vth_0 = 1.39
Fwd = 4.1
Bck = 0.4

# Generate a dense array for the theoretical smooth curve
dTs_dense = np.linspace(dTs.min(), dTs.max(), 500)
Vth_model = model_func(dTs_dense, Vth_0, Ec, Fwd, Bck)

# ==========================================================
# 4. Plotting Execution
# ==========================================================
plt.figure(figsize=(10, 8))

# Up Sweep: Crimson Red, Solid Line
plt.errorbar(dTs, Vth_up, yerr=err_up, marker='o', markersize=8, color='crimson',
             markerfacecolor='crimson', linestyle='-', linewidth=1.5,
             capsize=4, elinewidth=1.5, alpha=0.9, label='Sweep Up')

# Down Sweep: Dodgerblue, Dashed Line
plt.errorbar(dTs, Vth_down, yerr=err_down, marker='s', markersize=8, color='dodgerblue',
             markerfacecolor='dodgerblue', linestyle='--', linewidth=1.5,
             capsize=4, elinewidth=1.5, alpha=0.9, label='Sweep Down')

# Theoretical Model: Black, Dash-dot Line
label_model = rf'Model : $E_c={Ec}$, $V_0={Vth_0}$, $s^+={Fwd}$, $s^-={Bck}$'
plt.plot(dTs_dense, Vth_model, color='black', linestyle='-.', linewidth=2, zorder=3, label=label_model)

# Formatting Labels and Titles (using standard TeX strings)
plt.xlabel(r'Total Temperature Gradient $\Delta T = T_{right} - T_{left}$ ($e^2 / (k_B \langle C \rangle)$)', labelpad=15, fontsize=16)
plt.ylabel(r'Threshold Voltage $V_{th}$ ($e / \langle C \rangle$) [SNR Breakout]', labelpad=15, fontsize=16)

# Title String Configuration
title_str = "$C_g = 10$"  # Update this to match your actual variable if looped
plt.title(f'Threshold voltage as a function of $\\Delta T$\n{title_str}', pad=20, fontsize=18)

# Legend Formatting
legend = plt.legend(fontsize=16, loc='best')
legend.get_frame().set_linewidth(0.5)

# Grid and Layout (matching KC's 0.5pt standard)
plt.grid(True, linestyle='--', linewidth=0.5, color='lightgray')
plt.tight_layout()

# Export as PDF (Vector Graphic)
out_path = sys_dir / 'Vth_vs_Gradient_with_error.pdf'
plt.savefig(out_path, format='pdf', bbox_inches='tight')
print(f"Graph successfully saved to: {out_path.absolute()}")

plt.show()