import csv
from pathlib import Path
import matplotlib
from define_objects import ExperimentInitialState
import orjson
import numpy as np

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import re

run_name = "20251129_01h21m01s"
directory = Path(__file__).parent.parent / f"results_{run_name}"

# get init from json
infile = Path(directory / f"{run_name}.json")
if infile.exists():
    json_txt = infile.read_text()
    raw_fields = orjson.loads(json_txt)
    # recreate old init state
    init = ExperimentInitialState(**raw_fields)
else:
    raise ImportError(f"json does not exist in {directory}")

# get each csv file in folder
for csv_path in directory.glob("*.csv"):
    name = csv_path.name
    print(f"processing {name}")
    repetition = re.search(r"rep(\d+)", name).group(1)  # filename should be "...rep%J" %J is int
    dT = int(repetition) / 20
    total_deltaT = dT*(init.row_num-1)

    with csv_path.open() as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]

    V_column = [row[0] for row in rows]
    I_column = [row[1] for row in rows]
    idx_max = V_column.index(max(V_column))

    # build I vectors from data
    I_vec_forward = np.array([float(I) for I in I_column[0:idx_max+1]])
    I_vec_backward = np.flip([float(I) for I in I_column[idx_max:]])

    # build V vector
    Vleft = np.array([float(V) for V in V_column[0:idx_max+1]])

    plt.figure()
    plt.plot(
        Vleft,
        I_vec_forward,
        label="increasing",
        color="red",
    )
    plt.plot(
        Vleft,
        I_vec_backward,
        label="decreasing",
        color="blue",
    )
    plt.xlabel("Voltage")
    plt.ylabel("Current L->R")
    plt.legend()
    pic_name = name + ".png"

    if init.flip:
        plt.title(r"Even |$\Delta$T| = " + f"{round(total_deltaT, 1)}*T0 gradient, Tleft > Tright" + "\n"
                  + f"T diff between sites : dT = {dT}*T0\n"
                  + f"T0 = {init.T0}\n")
    else:
        plt.title(r"Even |$\Delta$T| = " + f"{round(total_deltaT, 1)}*T0 gradient, Tleft < Tright" + "\n"
                  + f"T diff between sites : dT = {dT}*T0\n"
                  + f"T0 = {init.T0}\n")

    plt.savefig(fname=directory / pic_name, dpi=2100, bbox_inches="tight")

    #plt.show()
