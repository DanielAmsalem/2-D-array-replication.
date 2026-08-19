import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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

# Font Sizes
rcParams['axes.titlesize'] = 25
rcParams['axes.labelsize'] = 22
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18

# Unified Unit Strings
T_str = r"$\left[ \frac{e^2}{k_B \langle C \rangle} \right]$"
Current_str = r"$\left[ \frac{e}{\langle R \rangle \langle C \rangle} \right]$"

# ==========================================
# 2. Configuration & File Mapping
# ==========================================
C_g = 20
E_c = 0.025  # derived from C_g

# Map the theoretical states to their respective filenames and plot styles
FILES_CONFIG = {
    r'$\Delta = 0$ (Metallic)': {
        'file': 'NEW_METAL_OLDCONFIG_IV_data_reintegrated.csv',
        'color': 'tomato',
        'marker': 's'
    },
    r'$\Delta = 0.2E_c$': {
        'file': 'IV_data_reintegrated_D0_2_oldconfig (2).csv',
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
# 3. Data Extraction & Plotting
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

print("Extracting V=0 thermocurrent data...")

for label, props in FILES_CONFIG.items():
    file_path = props['file']

    # Safely skip missing files
    if not os.path.exists(file_path):
        print(f"  [SKIP] File not found: {file_path}")
        continue

    print(f"  [LOAD] Processing: {file_path}")
    df = pd.read_csv(file_path)

    # Extract the first row as requested
    row_data = df.iloc[0]

    # Just a sanity check to print the actual voltage of the first row
    v_val = row_data['Vl (V)']
    print(f"         -> Confirmed baseline voltage for {label}: V = {v_val}")

    dT_vals = []
    I_vals = []

    # Parse the columns to match gradients
    for col in df.columns:
        if 'Grad_' in col and '_I' in col:
            n = int(col.split('_')[1])
            dt_val = round(n * 0.02, 3)
            if abs(dt_val) < 4*0.025:
                dT_vals.append(dt_val)
                I_vals.append(row_data[col])

    # Sort the arrays by Delta T to ensure the line plots smoothly from left to right
    sort_indices = np.argsort(dT_vals)
    dT_vals = np.array(dT_vals)[sort_indices]
    I_vals = np.array(I_vals)[sort_indices]

    # Plot this specific gap configuration
    ax.plot(dT_vals/E_c, I_vals, marker=props['marker'], markersize=8, color='black',
            markerfacecolor=props['color'], linestyle='-', linewidth=1.5, alpha=0.9,
            label=label)

# ==========================================
# 4. Finalizing KC Formatting
# ==========================================
ax.set_xlabel(r'$\Delta T/E_c$ ', labelpad=15)
ax.set_ylabel(r'Thermocurrent $I(V=0)$ ' + Current_str, labelpad=15)

# Clean, informative title defining the system electrostatics
ax.set_title(rf'Thermoelectric Current at Zero Bias', pad=20)

legend = ax.legend(fontsize=18, loc='best')
legend.get_frame().set_linewidth(0.5)
ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

plt.tight_layout()
out_name = 'Thermocurrent_vs_DeltaT_V0.pdf'
plt.savefig(out_name, format='pdf', bbox_inches='tight')
plt.close(fig)

print(f"\nSuccess: Plot saved as {out_name}")