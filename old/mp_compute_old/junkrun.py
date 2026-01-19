import Functions as F
import numpy as np
import matplotlib
from curve_plotter import plot_capacitance_map
from gamma_functions import execute_transition
from preparation import compute_distributed_C_matrices

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import csv
from pathlib import Path

MP_COMPUTE_PATH = Path(__file__).parent.parent / "mp_compute"
csv_table_path = MP_COMPUTE_PATH / f"table.csv"

with open(csv_table_path) as f:
    rows = list(csv.reader(f))
    neg = [row[1] for row in rows]
    pos = [row[2] for row in rows]

print(neg)
print(pos)
