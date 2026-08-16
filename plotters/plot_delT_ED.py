import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib import rcParams

# ==========================================
# 1. KC's Strict Formatting Guidelines
# ==========================================
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Myriad Pro', 'Arial']
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

# Unified Unit Strings
Volt_str = r"$\left[ \frac{e}{\langle C \rangle} \right]$"
T_str = r"$\left[ \frac{e^2}{k_B \langle C \rangle} \right]$"
S_str = r"$\left[ \frac{k_B}{e} \right]$"

# --- Configuration ---
FILE_NAME = 'NEW_IV_data_reintegrated_metal_a.csv'
E_c = 0.05  # Charging energy constant
m_values = [1, 3, 10, 19]  # Gradient multipliers

# Define distinct styling for the 4 loops to maintain a clean aesthetic
colors = ['dodgerblue', 'mediumseagreen', 'tomato', 'mediumpurple']
markers = ['o', 's', '^', 'D']


# ==========================================
# 2. Math Functions: Vertical Feedback Walk
# ==========================================
def walk_down_for_voltage(V_array, I_target_col, I_0, start_v_idx):
    """
    Walks downward (increasing voltage) in the target gradient column to find
    the exact voltage required to restore the original current (I_0).

    Returns:
        delta_V: The required voltage bias to counteract the gradient.
    """
    # Slice arrays to only search forward (downward in the table) from the current voltage
    search_I = I_target_col[start_v_idx:]
    search_V = V_array[start_v_idx:]

    # Find the first index where the new current crosses/exceeds I_0
    cross_indices = np.where(search_I >= I_0)[0]

    if len(cross_indices) == 0:
        return np.nan  # Array ended before current recovered to I_0

    idx_cross = cross_indices[0]

    if idx_cross == 0:
        # If it's already higher at the exact same voltage, delta_V is ~0
        # (or the threshold criteria wasn't strictly broken)
        if abs(search_I[0] - I_0) < 1e-6:
            return 0.0
        return np.nan

    # Sub-grid linear interpolation to find the exact V where I == I_0
    V_before = search_V[idx_cross - 1]
    V_after = search_V[idx_cross]
    I_before = search_I[idx_cross - 1]
    I_after = search_I[idx_cross]

    if I_after == I_before:
        V_exact = V_before
    else:
        # y = mx + b inversion to find exact x (Voltage)
        V_exact = V_before + (V_after - V_before) * ((I_0 - I_before) / (I_after - I_before))

    # delta V = V_new - V_original
    delta_V = V_exact - V_array[start_v_idx]

    return delta_V


# ==========================================
# 3. Data Processing & 4. Plotting (Combined)
# ==========================================
print("Loading and grouping raw data...", flush=True)
df = pd.read_csv(FILE_NAME)
V_array = df['Vl (V)'].values

# Extract and average gradients
grad_data = {}
for col in df.columns:
    if 'Grad_' in col and '_I' in col:
        n = int(col.split('_')[1])
        dt_val = round(n * 0.02, 2)  # Rounded to prevent Python floating-point key mismatch
        if dt_val not in grad_data:
            grad_data[dt_val] = []
        grad_data[dt_val].append(df[col].values)

for dt in grad_data:
    grad_data[dt] = np.mean(grad_data[dt], axis=0)

try:
    I_col_0 = grad_data[0.0]
except KeyError:
    raise ValueError("Could not find baseline gradient column (dT=0.0) in the CSV.")

# Initialize the Figures before the loop
fig1, ax1 = plt.subplots(figsize=(10, 8))
fig2, ax2 = plt.subplots(figsize=(10, 8))

# Draw the zero-lines once
ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)

all_results_dfs = []

print("Starting iteration over specified gradients...", flush=True)

# Loop over the user-defined m values
for idx, m in enumerate(m_values):
    DT_STEP = round(m * 0.02, 2)

    if DT_STEP not in grad_data:
        print(f"Warning: Gradient dT={DT_STEP} not found. Skipping m={m}.")
        continue

    I_col_target = grad_data[DT_STEP]

    print(f"  --> Calculating for m={m} (dT={DT_STEP})...", flush=True)
    results = []

    # Iterate over every starting voltage
    for v_idx, v_val in enumerate(V_array):
        I_0 = I_col_0[v_idx]
        delta_V = walk_down_for_voltage(V_array, I_col_target, I_0, v_idx)
        S_V = -delta_V / DT_STEP if not np.isnan(delta_V) else np.nan

        results.append({
            'm_value': m,
            'dT_step': DT_STEP,
            'Vl': v_val,
            'I_0': I_0,
            'delta_V': delta_V,
            'S(V)': S_V
        })

    res_df = pd.DataFrame(results)
    plot_df = res_df.dropna(subset=['S(V)']).copy()
    all_results_dfs.append(plot_df)

    # Calculate dynamic ratio for the legend
    grad_ratio = DT_STEP / E_c
    legend_label = rf'$\Delta T / E_c = {grad_ratio:g}$'  # ':g' removes trailing zeros intelligently

    # Add to GRAPH 1 (Delta V)
    ax1.plot(plot_df['Vl'], plot_df['delta_V'], marker=markers[idx], markersize=6, color='black',
             markerfacecolor=colors[idx], linestyle='-', linewidth=1.5, alpha=0.9,
             label=legend_label)

    # Add to GRAPH 2 (S(V))
    ax2.plot(plot_df['Vl'], plot_df['S(V)'], marker=markers[idx], markersize=6, color='black',
             markerfacecolor=colors[idx], linestyle='-', linewidth=1.5, alpha=0.9,
             label=legend_label)

# Concatenate all results and save them out
final_data_export = pd.concat(all_results_dfs, ignore_index=True)
final_data_export.to_csv('thermopower_results_active_feedback_all.csv', index=False)
print("All data exported. Finalizing plots...", flush=True)

# ----------------------------------------------------
# Finalize GRAPH 1: Delta V
# ----------------------------------------------------
ax1.set_xlabel(r'Left Electrode $V_{left}$ ' + Volt_str, labelpad=15)
ax1.set_ylabel(r'Required Bias Shift $\Delta V$ ' + Volt_str, labelpad=15)
ax1.set_title(r'Counteracting Bias $\Delta V$ to Maintain $I(V_{left},\Delta T=0)$', pad=20)

legend1 = ax1.legend(fontsize=18, loc='best')
legend1.get_frame().set_linewidth(0.5)
ax1.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

fig1.tight_layout()
fig1.savefig('Delta_V_vs_Voltage_Active_MultiGrad.pdf', format='pdf', bbox_inches='tight')
plt.close(fig1)

# ----------------------------------------------------
# Finalize GRAPH 2: Thermopower S(V)
# ----------------------------------------------------
ax2.set_xlabel(r'Baseline Voltage $V_{left}$ ' + Volt_str, labelpad=15)
ax2.set_ylabel(r'Thermopower $S(V)$ ' + S_str, labelpad=15)
ax2.set_title(r'Thermopower as a function of $V_{left}$', pad=20)

legend2 = ax2.legend(fontsize=18, loc='best')
legend2.get_frame().set_linewidth(0.5)
ax2.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

fig2.tight_layout()
fig2.savefig('Thermopower_S_vs_Voltage_Active_MultiGrad.pdf', format='pdf', bbox_inches='tight')
plt.close(fig2)

print("Batch processing complete: Multi-gradient active feedback loop exported to vector PDFs.")