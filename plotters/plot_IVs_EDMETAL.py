import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

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

# Font Sizes (Reduced slightly to prevent messiness, keeping KC's ratio)
rcParams['axes.titlesize'] = 25
rcParams['axes.labelsize'] = 22
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18

# ==========================================
# 2. Data Loading & Processing
# ==========================================
# Load the dataset
df = pd.read_csv('NEW_IV_data_reintegrated.csv')

# 0. MOST IMPORTANT: Enforce Voltage Cutoff V <= 1.4
df = df[df.iloc[:, 0] <= 1.4]

# Extract Voltage (x-axis)
V = df.iloc[:, 0].values

# Extract Delta T values (y-axis) and build a 2D Current matrix (z-axis)
dTs = []
I_list = []

for col in df.columns[1:]:
    # FIX: Added '-?' to the regex to successfully capture negative numbers
    match = re.search(r'Grad_(-?\d+)_I', col)
    if match:
        n = int(match.group(1))
        dTs.append(n * 0.02)
        I_list.append(df[col].values)

# FIX: Sort the extracted arrays by Delta T to prevent 3D mesh folding/crashing
sorted_data = sorted(zip(dTs, I_list), key=lambda x: x[0])
dTs = np.array([item[0] for item in sorted_data])
I_matrix = np.array([item[1] for item in sorted_data]) * 1e3 # Scale current by 10^3

# Create 2D Meshgrids for X (Voltage) and Y (Delta T)
X, Y = np.meshgrid(V, dTs)
Z = I_matrix

# Pre-calculate delta T range for the Colormap (Blue to Red)
min_dT, max_dT = min(dTs), max(dTs)
cmap = plt.cm.coolwarm  # Maps minimum to blue, maximum to red
norm = plt.Normalize(vmin=min_dT, vmax=max_dT)


# Helper function to clean up 3D pane aesthetics and handle Z-axis specifics
def format_3d_axes(ax):
    # Enforce the 0.5 pt rule on the 3D panes
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_linewidth(0.5)
        axis.pane.set_edgecolor('black')
        axis.pane.fill = False  # Transparent panes look much cleaner
        axis._axinfo["grid"].update({"linewidth": 0.5, "color": "lightgray"})

    # Manually set z-tick width and label size to match x and y
    ax.tick_params(axis='z', which='major', width=0.5, labelsize=18)

    # Reduce the number of ticks to prevent ugly overlapping numbers
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(5))


# ==========================================
# 3. Plot 1: Surface Mesh 3D Plot
# ==========================================
fig1 = plt.figure(figsize=(12, 10))

# FIX: Squeezed the plot further to the right (left=0.25) to make room for the newly rotated Z-axis!
fig1.subplots_adjust(left=0.1, right=0.65, top=0.90, bottom=0.10)

ax1 = fig1.add_subplot(111, projection='3d')

# FIX: Dynamically calculate a mesh stride so the web has ~30 segments total.
# This prevents the black lines from merging into a solid gray blob.
r_stride = max(1, len(dTs) // 30)
c_stride = max(1, len(V) // 30)

# Plot the Surface Mesh
surf = ax1.plot_surface(X, Y, Z, facecolors=cmap(norm(Y)),
                        edgecolor='white', linewidth=0.3,
                        rstride=r_stride, cstride=c_stride,
                        antialiased=True, alpha=0.9, shade=False)

# Set labels and title
ax1.set_title('I-V Characteristics Under Thermal Gradient', y=1)

ax1.set_xlabel(r'Voltage $\left[ \frac{e}{\langle C \rangle} \right]$', labelpad=25)
ax1.set_ylabel(r'$\Delta T \left[ \frac{e^2}{k_B \langle C \rangle} \right]$', labelpad=25)
ax1.set_zlabel(r'Current $\left[ 10^{-3} \frac{e}{\langle R \rangle \langle C \rangle} \right]$', labelpad=15)

format_3d_axes(ax1)

# Reverse the axes to maintain the specific visual perspective requested
ax1.invert_xaxis()
ax1.invert_yaxis()

# Rotate view to favor V-axis
ax1.view_init(elev=25, azim=-45)

# Force Matplotlib to calculate margins before cropping
plt.tight_layout()

# Export to PDF with a heavy protective pad to guarantee no text is clipped
plt.savefig('3D_IV_Surface.pdf', format='pdf', bbox_inches='tight', pad_inches=0.5)
plt.close(fig1)

print("Plot successfully generated and saved as 3D_IV_Surface.pdf")