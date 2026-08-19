import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

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
# 2. Data Loading & Processing
# ==========================================
# Load the datasets
df_sc_D_large = pd.read_csv("D2_0_island_vth_in_paper.csv")
df_sc = pd.read_csv('thermopower_results_exact_hybrid.csv')
df_metal = pd.read_csv('thermopower_METAL_20T0_results_exact_hybrid.csv')

all_metal_from = 3
dT_at_all_metal = all_metal_from * 0.02/0.025

# Extract Delta T (x-axis) -> n * 0.02 --> devide by Ec = 0.025
delta_T_D_2 = df_sc_D_large['n'] * 0.02 / 0.025
Vth_D2 = df_sc_D_large['Vth'].copy()

delta_T_sc = df_sc['n'] * 0.02 / 0.025
Vth_sc = df_sc['Vth_Final'].copy()

delta_T_metal = df_metal['n'] * 0.02 / 0.025
Vth_metal = df_metal['Vth_Final'].copy()

# ------------------------------------------
# Artificial Swap: Set Vth_D2 and Vth_sc to Vth_metal when deltaT > dT_at_all_metal
# ------------------------------------------
mask_D2 = delta_T_D_2 > dT_at_all_metal
Vth_D2.loc[mask_D2] = Vth_metal.loc[mask_D2]

mask_sc = delta_T_sc > dT_at_all_metal
Vth_sc.loc[mask_sc] = Vth_metal.loc[mask_sc]
# ------------------------------------------

# ==========================================
# 3. Plotting the 2D Graph
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