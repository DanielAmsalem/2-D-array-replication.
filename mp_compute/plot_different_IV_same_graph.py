import csv
import numpy as np
import matplotlib
import gc
from pathlib import Path

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Define the current directory where the script and CSV files run
directory = Path(__file__).parent

# Upgraded dictionary: Now holds the filename AND its specific vertical shift
plot_config = {
    "Cg2": {"filename": "Cg2_rep0.csv", "shift": 0.0},
    "Cg5": {"filename": "Cg5_rep0.csv", "shift": 0.5},
    "Cg10": {"filename": "Cg10_rep0.csv", "shift": 1.0}
}

fig = plt.figure()

for label, config in plot_config.items():
    csv_path = directory / config["filename"]

    if not csv_path.exists():
        print(f"Warning: {config['filename']} not found in {directory}. Skipping...")
        continue

    with csv_path.open() as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]

    V_column = [float(row[0]) for row in rows]
    I_column = [float(row[1]) for row in rows]

    idx_max = V_column.index(max(V_column))

    # Build vectors from data
    I_vec_forward = np.array(I_column[0:idx_max + 1])
    I_vec_backward = np.flip(I_column[idx_max:])

    # Build V vector
    Vleft = np.array(V_column[0:idx_max + 1])

    # Extract the specific shift for this file from the dictionary
    y_shift = config["shift"]

    # Assign labels only for the first file in the dictionary to prevent duplicates
    label_inc = "increasing" if label == "Cg2" else None
    label_dec = "decreasing" if label == "Cg2" else None

    # Both lines are now kept smooth using linestyle="-"
    plt.plot(
        Vleft,
        I_vec_forward + y_shift,
        label=label_inc,
        color="red",
        linestyle="-"
    )
    plt.plot(
        Vleft,
        I_vec_backward + y_shift,
        label=label_dec,
        color="blue",
        linestyle="-"
    )

# Restrict the x-axis to 0.75 < V < 4
plt.xlim(0.75, 4)

# Hard limit the y-axis to deliberately crop the higher values
plt.ylim(-0.3, 5)

plt.xlabel(r"$V$ $\left[\frac{e}{\langle C \rangle}\right]$")
plt.ylabel(r"$I(V)$ $\left[\frac{e}{\langle R \rangle \langle C \rangle}\right]$")

plt.title(r"$V_{th}$ and Hysteresis loop area as function of $C_g$")
plt.legend()

# Updated the output filename
pic_name = "Cg2_Cg5_Cg10_IV_Graph.png"
plt.savefig(fname=directory / pic_name, dpi=2100, bbox_inches="tight")

plt.close(fig)
gc.collect()

print(f"Graph successfully generated and saved as {pic_name}")