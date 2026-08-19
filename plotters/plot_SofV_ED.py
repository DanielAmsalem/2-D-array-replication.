import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# ==========================================
# 2. Configuration & File Mapping
# ==========================================
E_c = 0.025  # Charging energy constant
M_VAL = 1  # Target gradient multiplier
DT_STEP = round(M_VAL * 0.02, 2)  # Specific delta T to evaluate
V_MAX = 2.0  # Maximum voltage to calculate and plot

# Map the theoretical states to their respective filenames and plot styles
FILES_CONFIG = {
    r'$\Delta = 0$ (Metallic)': {
        'file': 'NEW_METAL_OLDCONFIG_IV_data_reintegrated.csv',
        'color': 'tomato',
        'marker': 's'
    },
    r'$\Delta = 0.2E_c$': {
        'file': 'IV_data_reintegrated_D0_2_oldconfig.csv',
        'color': 'mediumseagreen',
        'marker': '^'
    },
    r'$\Delta = 2E_c$': {
        'file': 'NEW_IV_data_reintegrated_D2_0_OLDCONFIG.csv',
        'color': 'dodgerblue',
        'marker': 'o'
    }
}


# ==========================================
# 3. Math Functions: Vertical Feedback Walk
# ==========================================
def walk_bidirectional_for_voltage(V_array, I_target_col, I_0, start_v_idx):
    """
    Checks the initial current state. If the gradient caused current to drop,
    walks downward (increases voltage) to restore I_0. If the gradient caused
    current to spike (e.g. gap suppression), walks upward (decreases voltage).

    Returns:
        delta_V: The required voltage bias shift to counteract the gradient.
    """
    I_initial = I_target_col[start_v_idx]

    # If already within threshold, no bias shift is needed
    if abs(I_initial - I_0) < 1e-6:
        return 0.0

    if I_initial < I_0:
        # CASE 1: Current dropped. We need MORE voltage. (Walk down the table)
        search_I = I_target_col[start_v_idx:]
        search_V = V_array[start_v_idx:]

        cross_indices = np.where(search_I >= I_0)[0]
        if len(cross_indices) == 0:
            return np.nan  # Array ended before current recovered

        idx_cross = cross_indices[0]
        if idx_cross == 0:
            return 0.0

        V_before = search_V[idx_cross - 1]
        V_after = search_V[idx_cross]
        I_before = search_I[idx_cross - 1]
        I_after = search_I[idx_cross]

    else:
        # CASE 2: Current spiked (Gap suppression). We need LESS voltage. (Walk up the table)
        search_I = I_target_col[:start_v_idx + 1]
        search_V = V_array[:start_v_idx + 1]

        # Since we are walking backwards, we want to find where the current drops back below I_0
        cross_indices = np.where(search_I <= I_0)[0]
        if len(cross_indices) == 0:
            return np.nan  # Reached 0V without dropping enough

        idx_cross = cross_indices[-1]  # The last index (highest voltage) where I <= I_0

        if idx_cross == len(search_I) - 1:
            return 0.0

        V_before = search_V[idx_cross]
        V_after = search_V[idx_cross + 1]
        I_before = search_I[idx_cross]
        I_after = search_I[idx_cross + 1]

    # Sub-grid linear interpolation to find the exact V where I == I_0
    if I_after == I_before:
        V_exact = V_before
    else:
        # y = mx + b inversion to find exact x (Voltage)
        V_exact = V_before + (V_after - V_before) * ((I_0 - I_before) / (I_after - I_before))

    # delta V = V_new - V_original. (Will naturally be negative for Case 2)
    delta_V = V_exact - V_array[start_v_idx]

    return delta_V


# ==========================================
# 4. Data Processing & 5. Plotting
# ==========================================

# Initialize the 2 Figures before the loop
fig1, ax1 = plt.subplots(figsize=(10, 8))  # Linear Delta V
fig2, ax2 = plt.subplots(figsize=(10, 8))  # Linear S(V)

# Draw the zero/baseline lines once
ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)

all_results_dfs = []

print(f"Starting bidirectional active feedback loop for dT = {DT_STEP}...", flush=True)

for label, props in FILES_CONFIG.items():
    file_path = props['file']

    if not os.path.exists(file_path):
        print(f"  [SKIP] File not found: {file_path}")
        continue

    print(f"  [LOAD] Processing: {label} from {file_path}")
    df = pd.read_csv(file_path)
    V_array = df['Vl (V)'].values

    # Extract and average gradients
    grad_data = {}
    for col in df.columns:
        if 'Grad_' in col and '_I' in col:
            n = int(col.split('_')[1])
            dt_val = round(n * 0.02, 2)
            if dt_val not in grad_data:
                grad_data[dt_val] = []
            grad_data[dt_val].append(df[col].values)

    for dt in grad_data:
        grad_data[dt] = np.mean(grad_data[dt], axis=0)

    if 0.0 not in grad_data:
        print(f"  [ERROR] Baseline gradient (dT=0.0) missing in {file_path}. Skipping.")
        continue
    if DT_STEP not in grad_data:
        print(f"  [ERROR] Target gradient (dT={DT_STEP}) missing in {file_path}. Skipping.")
        continue

    I_col_0 = grad_data[0.0]
    I_col_target = grad_data[DT_STEP]

    results = []

    for v_idx, v_val in enumerate(V_array):
        # Stop processing if we exceed V_MAX to save calculation time
        if v_val > V_MAX:
            continue

        I_0 = I_col_0[v_idx]
        delta_V = walk_bidirectional_for_voltage(V_array, I_col_target, I_0, v_idx)

        # Standard Linear Values
        S_V = delta_V / DT_STEP if not np.isnan(delta_V) else np.nan

        results.append({
            'Gap': label,
            'Vl': v_val,
            'I_0': I_0,
            'delta_V': delta_V,
            'S(V)': S_V
        })

    res_df = pd.DataFrame(results)
    plot_df = res_df.dropna(subset=['S(V)']).copy()

    # Filter the final plot dataframe to strictly enforce the X-axis cutoff limit
    plot_df = plot_df[plot_df['Vl'] <= V_MAX]
    all_results_dfs.append(plot_df)

    # Add to GRAPH 1 (Linear Delta V)
    ax1.plot(plot_df['Vl'], plot_df['delta_V'], marker=props['marker'], markersize=6, color='black',
             markerfacecolor=props['color'], linestyle='-', linewidth=1.5, alpha=0.9, label=label)

    # Add to GRAPH 2 (Linear S(V))
    ax2.plot(plot_df['Vl'], plot_df['S(V)'], marker=props['marker'], markersize=6, color='black',
             markerfacecolor=props['color'], linestyle='-', linewidth=1.5, alpha=0.9, label=label)

# Concatenate all valid results and save them out
if all_results_dfs:
    final_data_export = pd.concat(all_results_dfs, ignore_index=True)
    final_data_export.to_csv(f'thermopower_results_across_gaps_dT_{DT_STEP}.csv', index=False)
    print("\nAll data exported successfully. Finalizing plots...", flush=True)

# ----------------------------------------------------
# Finalize GRAPH 1: Linear Delta V
# ----------------------------------------------------
ax1.set_xlim(right=V_MAX)  # Enforce visual cutoff limit
ax1.set_xlabel(r'Left Electrode $V_{left}$ ' + Volt_str, labelpad=15)
ax1.set_ylabel(r'Required Bias Shift $\Delta V$ ' + Volt_str, labelpad=15)
ax1.set_title(rf'Counteracting Bias to Maintain $I_0$ ($\Delta T/E_c = {round(DT_STEP/E_c,1)}$)', pad=20)
legend1 = ax1.legend(fontsize=18, loc='best')
if legend1: legend1.get_frame().set_linewidth(0.5)
ax1.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

fig1.tight_layout()
fig1.savefig('Delta_V_vs_Voltage_Across_Gaps.pdf', format='pdf', bbox_inches='tight')
plt.close(fig1)

# ----------------------------------------------------
# Finalize GRAPH 2: Linear Thermopower S(V)
# ----------------------------------------------------
ax2.set_xlim(right=V_MAX)  # Enforce visual cutoff limit
ax2.set_xlabel(r'Left Electrode $V_{left}$ ' + Volt_str, labelpad=15)
ax2.set_ylabel(r'Thermopower $S(V)$ ' + S_str, labelpad=15)
ax2.set_title(rf'Thermopower vs. Voltage ($\Delta T/E_c = {round(DT_STEP/E_c,1)}$)', pad=20)
legend2 = ax2.legend(fontsize=18, loc='best')
if legend2: legend2.get_frame().set_linewidth(0.5)
ax2.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

fig2.tight_layout()
fig2.savefig('Thermopower_S_vs_Voltage_Across_Gaps.pdf', format='pdf', bbox_inches='tight')
plt.close(fig2)

print("Batch processing complete: Active feedback loop for all gap states exported to vector PDFs.")