import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib import rcParams
from pathlib import Path

# ==========================================
# 1. KC's Strict Formatting Guidelines
# ==========================================
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Myriad Pro', 'Arial']
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5
rcParams['axes.titlesize'] = 25
rcParams['axes.labelsize'] = 22
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18

# Unified Unit Strings
Volt_str = r"$\left[ \frac{e}{\langle C \rangle} \right]$"
T_str = r"$\left[ \frac{e^2}{k_B \langle C \rangle} \right]$"
S_str = r"$\left[ \frac{k_B}{e} \right]$"


# ==========================================
# 2. Math Functions
# ==========================================
def exact_poly_dt_th(i_row, dTs, degree, threshold=1e-6):
    """
    Finds the exact delta T where the current breaks the threshold for a fixed V.
    Uses inverse polynomial interpolation: dT(I) evaluated at I=0.
    """
    mask = i_row > threshold
    if not mask.any():
        return np.nan

    idx = np.argmax(mask)

    # Define exact nodes based on degree
    if degree == 1:
        indices = [idx - 1, idx]
    elif degree == 3:
        indices = [idx - 1, idx, idx + 1, idx + 2]
    else:
        return np.nan

    # Ensure indices stay within bounds
    indices = [i for i in indices if 0 <= i < len(i_row)]

    # Must have exactly the required number of points for the degree
    if len(indices) != degree + 1:
        return np.nan

    dt_sub = dTs[indices]
    i_sub = i_row[indices]

    # Exact fit mapping I to dT
    coeffs = np.polyfit(i_sub, dt_sub, degree)
    poly = np.poly1d(coeffs)

    # Evaluate at I = 0
    return poly(0)


# ==========================================
# 3. Data Processing
# ==========================================
df = pd.read_csv('NEW_IV_data_reintegrated_metal.csv')
V_array = df.iloc[:, 0].values  # First column is Voltage

# Parse header to extract and sort delta T gradients
dt_map = {}
for col in df.columns[1:]:
    match = re.search(r'Grad_(-?\d+)_I', col)
    if match:
        n = int(match.group(1))
        dt_val = n * 0.02
        dt_map[col] = dt_val

# Sort columns so delta T strictly increases (vital for root finding)
sorted_cols = sorted(dt_map.keys(), key=lambda x: dt_map[x])
dTs = np.array([dt_map[c] for c in sorted_cols])

# Build the 2D Current matrix: Rows = V, Cols = I(dT)
I_matrix = df[sorted_cols].values

results = []
# Iterate over every row (fixed Voltage)
for v_idx, v_val in enumerate(V_array):
    i_row = I_matrix[v_idx, :]

    dt_th_linear = exact_poly_dt_th(i_row, dTs, degree=1)
    dt_th_cubic = exact_poly_dt_th(i_row, dTs, degree=3)

    # HYBRID SPLIT LOGIC: Adjust the voltage split threshold (e.g., 0.5) as needed based on your physics
    dt_th_chosen = dt_th_linear if v_val > 0.5 else dt_th_cubic

    results.append({
        'Vl': v_val,
        'dT_th_Linear': dt_th_linear,
        'dT_th_Cubic': dt_th_cubic,
        'dT_th_Final': dt_th_chosen
    })

res_df = pd.DataFrame(results)

# Calculate Thermopower S(V)
# Because S = - dV/dT, and we have dT_th as a function of V, we invert the derivative
res_df['dV'] = res_df['Vl'].diff()
res_df['d_dT_th'] = res_df['dT_th_Final'].diff()

# S(V) = - dV / d(dT_th). Use np.where to avoid division by zero
res_df['S(V)'] = np.where(res_df['d_dT_th'] != 0, -res_df['dV'] / res_df['d_dT_th'], np.nan)

# Save the mathematical output cleanly
res_df.to_csv('thermopower_results_V_domain_hybrid.csv', index=False)

# Filter for plotting to avoid plotting NaNs
plot_df = res_df.dropna(subset=['dT_th_Final']).copy()
plot_S_df = res_df.dropna(subset=['S(V)']).copy()

# ==========================================
# 4. KC Standard Plotting
# ==========================================

# GRAPH 1: Delta T_th vs Voltage
fig1, ax1 = plt.subplots(figsize=(10, 8))

ax1.plot(plot_df['Vl'], plot_df['dT_th_Linear'], color='dodgerblue', linestyle='-', alpha=0.4,
         label='Exact Linear (2-pt)')
ax1.plot(plot_df['Vl'], plot_df['dT_th_Cubic'], color='crimson', linestyle='--', alpha=0.4, label='Exact Cubic (4-pt)')

# Bold the chosen hybrid path
ax1.plot(plot_df['Vl'], plot_df['dT_th_Final'], marker='o', markersize=6, color='black', markerfacecolor='black',
         linestyle='', label='Chosen Hybrid')

ax1.axvline(0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.6, label='Hybrid Split (V=0.5)')

ax1.set_xlabel(r'Left Voltage $V_l$ ' + Volt_str, labelpad=15)
ax1.set_ylabel(r'Threshold Temperature Gradient $\Delta T_{th}$ ' + T_str, labelpad=15)
ax1.set_title(r'Threshold Gradient $\Delta T_{th}$ as a function of $V_l$', pad=20)

legend1 = ax1.legend(fontsize=18, loc='best')
legend1.get_frame().set_linewidth(0.5)
ax1.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

plt.tight_layout()
plt.savefig('dT_th_vs_Voltage_Hybrid.pdf', format='pdf', bbox_inches='tight')
plt.close(fig1)

# GRAPH 2: Thermopower S(V) vs Voltage
fig2, ax2 = plt.subplots(figsize=(10, 8))

ax2.plot(plot_S_df['Vl'], plot_S_df['S(V)'], marker='s', markersize=6, color='black', markerfacecolor='mediumseagreen',
         linestyle='-', linewidth=1.5, alpha=0.9, label='S(V)')

ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)

ax2.set_xlabel(r'Left Voltage $V_l$ ' + Volt_str, labelpad=15)
ax2.set_ylabel(r'Thermopower $S(V_l) = -dV / d(\Delta T_{th})$ ' + S_str, labelpad=15)
ax2.set_title(r'Thermopower $S(V_l)$ as a function of $V_l$', pad=20)

legend2 = ax2.legend(fontsize=18, loc='best')
legend2.get_frame().set_linewidth(0.5)
ax2.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

plt.tight_layout()
plt.savefig('Thermopower_S_vs_Voltage.pdf', format='pdf', bbox_inches='tight')
plt.close(fig2)

print("Batch processing complete: Row-wise Inverse Fits calculated and exported to vector PDFs.")