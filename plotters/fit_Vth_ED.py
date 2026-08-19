import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ==========================================
# 1. KC's Strict Formatting Guidelines
# ==========================================
# Use Myriad Pro (Ensure it is installed on your OS)
rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Myriad Pro', 'Arial']  # Uncomment if fonts are locally linked
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
# 2. Data Loading & Processing
# ==========================================
# Load the datasets
df_sc_D_large = pd.read_csv("D2_0_island_vth_in_paper.csv")
df_sc = pd.read_csv('thermopower_results_exact_hybrid.csv')
df_metal = pd.read_csv('thermopower_METAL_20T0_results_exact_hybrid.csv')

# Extract actual physical Delta T to allow unscaled formula math
real_dT_D2 = df_sc_D_large['n'] * 0.02
real_dT_sc = df_sc['n'] * 0.02
real_dT_metal = df_metal['n'] * 0.02

# Extract Scaled Delta T (x-axis) --> divide by Ec = 0.025
delta_T_D_2 = real_dT_D2 / 0.025
Vth_D2 = df_sc_D_large['Vth']

delta_T_sc = real_dT_sc / 0.025
Vth_sc = df_sc['Vth_Final']

delta_T_metal = real_dT_metal / 0.025
Vth_metal = df_metal['Vth_Final']

# ==========================================
# 3. Analytic Model Definition
# ==========================================
def model_func(dT, a, b, c, q):
    return np.where(
        dT > 0,
        a * (np.exp(-np.abs(dT) / b) + c * dT),
        a * (np.exp(-np.abs(dT) / b) + q * dT)
    )

# Target parameters from previous script
a_man, b_man, c_man, q_man = 0.44, 0.05,5, 20

# Generate a dense array spanning the physical dT limits for a smooth curve
min_dT = min(real_dT_D2.min(), real_dT_sc.min(), real_dT_metal.min())
max_dT = max(real_dT_D2.max(), real_dT_sc.max(), real_dT_metal.max())
dT_dense = np.linspace(min_dT, max_dT, 400)

# Calculate theoretical Vth and map x-axis back to the scaled space (dT / 0.025)
Vth_model = model_func(dT_dense, a_man, b_man, c_man, q_man)
delta_T_model_scaled = dT_dense / 0.025

# ==========================================
# 4. Plotting the 2D Graph
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))

# Plot D=2 Data (green triangles)
ax.plot(delta_T_D_2, Vth_D2, marker='^', markersize=8, color='black',
        markerfacecolor='green', linewidth=1.5, alpha=0.9,
        label=r'Superconducting ($\Delta = 2 E_c$)')

# Plot Superconducting Data (Blue Circles)
ax.plot(delta_T_sc, Vth_sc, marker='o', markersize=8, color='black',
        markerfacecolor='dodgerblue', linewidth=1.5, alpha=0.9,
        label=r'Superconducting ($\Delta = 0.2 E_c$)')

# Plot Metallic Data (Red Squares)
ax.plot(delta_T_metal, Vth_metal, marker='s', markersize=8, color='black',
        markerfacecolor='tomato', linewidth=1.5, alpha=0.9,
        label=r'Metallic ($\Delta = 0$)')

# Plot Theoretical Analytic Line
label_model = rf'Analytic Model ($a={a_man}$, $b={b_man}$, $c={c_man}$, $q={q_man}$)'
ax.plot(delta_T_model_scaled, Vth_model, color='black', linestyle='-.', linewidth=2, zorder=3, label=label_model)

# Set labels and title using the stacked LaTeX fractions
ax.set_title('Threshold Voltage vs. Temperature Gradient', pad=20)
ax.set_xlabel(r'$\Delta T / E_c $', labelpad=15)
ax.set_ylabel(r'Threshold Voltage $V_{th}$ $\left[ \frac{e}{\langle C \rangle} \right]$', labelpad=15)

# Add Legend (with KC's 0.5 pt border rule applied to the legend box)
legend = ax.legend(fontsize=18, loc='best', frameon=True, edgecolor='black')
legend.get_frame().set_linewidth(0.5)

# Add a subtle grid to help the eye map points to the axes
ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

# Force Matplotlib to calculate margins cleanly before cropping
plt.tight_layout()

# Export strictly to PDF with tight bounding box
plt.savefig('2D_Vth_vs_DeltaT_Comparison.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

print("Plot successfully generated and saved as 2D_Vth_vs_DeltaT_Comparison.pdf")