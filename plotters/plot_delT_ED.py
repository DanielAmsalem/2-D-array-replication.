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
FILE_NAME = 'NEW_IV_data_reintegrated_metal.csv'
T_0 = 0.001  # Replace with your actual T_0 value (in Kelvin)


# ---------------------

# ==========================================
# 2. Math Functions: Dynamic Bounding Box
# ==========================================
def dynamic_poly_dt_th(i_row, dt_arr, threshold=1e-7):
    """
    Finds the exact delta T where the current crosses zero.
    Walks from the most positive gradient (negative current)
    towards the negative gradient (positive current).
    """
    # 1. Start always at the most positive gradient. Current MUST be negative.
    if i_row[0] > -threshold:
        return np.nan

    # 2. Walk along the row until current is solidly positive
    pos_mask = i_row > threshold
    if not pos_mask.any():
        return np.nan
    idx_pos = np.argmax(pos_mask)  # First index where I > 1e-7

    # 3. Find the last point BEFORE idx_pos where current is solidly negative
    neg_mask = i_row[:idx_pos] < -threshold
    if not neg_mask.any():
        return np.nan
    idx_neg = np.where(neg_mask)[0][-1]

    # 4. Use indices from 1 before the negative anchor to 1 after the positive anchor
    start_idx = max(0, idx_neg - 1)
    end_idx = min(len(i_row) - 1, idx_pos + 1)

    indices = list(range(start_idx, end_idx + 1))

    # Need at least 2 points to fit a polynomial
    if len(indices) < 2:
        return np.nan

    # Dynamically scale degree (Max 3 to avoid Runge's phenomenon oscillations)
    degree = min(3, len(indices) - 1)

    i_sub = i_row[indices]
    dt_sub = dt_arr[indices]

    # Exact fit mapping I to dT
    coeffs = np.polyfit(i_sub, dt_sub, degree)
    poly = np.poly1d(coeffs)

    # Evaluate at I = 0
    return poly(0)


# ==========================================
# 3. Data Processing
# ==========================================
print("Loading and grouping raw data...", flush=True)
df = pd.read_csv(FILE_NAME)
V_array = df['Vl (V)'].values

# Extract all gradient arrays
grad_data = {}
for col in df.columns:
    if 'Grad_' in col and '_I' in col:
        n = int(col.split('_')[1])
        dt_val = n * (T_0 / 2.0)
        if dt_val not in grad_data:
            grad_data[dt_val] = []
        grad_data[dt_val].append(df[col].values)

# Average duplicates to ensure clean arrays
for dt in grad_data:
    grad_data[dt] = np.mean(grad_data[dt], axis=0)

# Sort strictly from MOST POSITIVE dT to MOST NEGATIVE dT
sorted_dts = sorted(grad_data.keys(), reverse=True)
dTs = np.array(sorted_dts)

# Build the 2D Current matrix: Rows = Voltage, Columns = Current
I_matrix = np.array([grad_data[dt] for dt in sorted_dts]).T

results = []
print("Calculating row-wise inverse polynomials...", flush=True)

# Iterate over every row (fixed Voltage)
for v_idx, v_val in enumerate(V_array):
    i_row = I_matrix[v_idx, :]
    dt_th = dynamic_poly_dt_th(i_row, dTs)

    results.append({
        'Vl': v_val,
        'dT_th_Final': dt_th
    })

res_df = pd.DataFrame(results)

# Filter for plotting to avoid plotting NaNs
plot_df = res_df.dropna(subset=['dT_th_Final']).copy()

# Compute final Thermopower S(V) using stable Central Difference
# S = -dV / d(dT). Using np.gradient directly on valid data arrays
dV_grad = np.gradient(plot_df['Vl'])
dT_grad = np.gradient(plot_df['dT_th_Final'])

# Safely invert the derivative
plot_df['S(V)'] = np.where(dT_grad != 0, -dV_grad / dT_grad, np.nan)

# Drop any NaN results from the derivative
plot_df = plot_df.dropna(subset=['S(V)'])

# Save the mathematical output cleanly
plot_df.to_csv('thermopower_results_V_domain_dynamic.csv', index=False)
print("Data exported. Generating Plots...", flush=True)

# ==========================================
# 4. KC Standard Plotting
# ==========================================

# ----------------------------------------------------
# GRAPH 1: Delta T_th vs Voltage
# ----------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 8))

ax1.plot(plot_df['Vl'], plot_df['dT_th_Final'], marker='o', markersize=6, color='black',
         markerfacecolor='dodgerblue', linestyle='-', linewidth=1.5, alpha=0.9,
         label=r'Extracted $\Delta T_{th}$')

ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

ax1.set_xlabel(r'Left Voltage $V_l$ ' + Volt_str, labelpad=15)
ax1.set_ylabel(r'Threshold Temperature Gradient $\Delta T_{th}$ ' + T_str, labelpad=15)
ax1.set_title(r'Threshold Gradient $\Delta T_{th}$ as a function of $V_l$', pad=20)

legend1 = ax1.legend(fontsize=18, loc='best')
legend1.get_frame().set_linewidth(0.5)
ax1.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

plt.tight_layout()
plt.savefig('dT_th_vs_Voltage_Dynamic.pdf', format='pdf', bbox_inches='tight')
plt.close(fig1)

# ----------------------------------------------------
# GRAPH 2: Thermopower S(V) vs Voltage
# ----------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 8))

ax2.plot(plot_df['Vl'], plot_df['S(V)'], marker='s', markersize=6, color='black',
         markerfacecolor='mediumseagreen', linestyle='-', linewidth=1.5, alpha=0.9,
         label=r'$S(V_l)$')

ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)

ax2.set_xlabel(r'Left Voltage $V_l$ ' + Volt_str, labelpad=15)
ax2.set_ylabel(r'Thermopower $S(V_l) = -dV_l / d(\Delta T_{th})$ ' + S_str, labelpad=15)
ax2.set_title(r'Thermopower $S(V_l)$ as a function of $V_l$', pad=20)

legend2 = ax2.legend(fontsize=18, loc='best')
legend2.get_frame().set_linewidth(0.5)
ax2.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

plt.tight_layout()
plt.savefig('Thermopower_S_vs_Voltage_Dynamic.pdf', format='pdf', bbox_inches='tight')
plt.close(fig2)

print("Batch processing complete: Row-wise Inverse Fits calculated and exported to vector PDFs.")