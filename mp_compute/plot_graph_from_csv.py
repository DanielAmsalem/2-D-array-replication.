import csv
import re
import orjson
import numpy as np
import matplotlib
import gc  # Imported for explicit garbage collection
from pathlib import Path
from define_objects import ExperimentInitialState

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

run_names = [
    "20260607_14h34m28s"
]

for run_name in run_names:
    print(f"=== Processing run: {run_name} ===")

    directory = Path(__file__).parent.parent / f"results_{run_name}"

    # get init from json
    infile = Path(directory / f"{run_name}.json")
    if infile.exists():
        json_txt = infile.read_text()
        raw_fields = orjson.loads(json_txt)
        # recreate old init state
        init = ExperimentInitialState(**raw_fields)
    else:
        print(f"Warning: json does not exist in {directory}. Skipping this run...")
        continue

    # get each csv file in folder
    for csv_path in directory.glob("*.csv"):
        name = csv_path.name
        print(f"  processing {name}")

        match = re.search(r"rep(\d+)", name)
        if not match:
            print(f"    Warning: 'rep' not found in {name}. Skipping file...")
            continue

        repetition = match.group(1)  # filename should be "...rep%J" %J is int
        if (directory / f"book_{run_name}_rep{repetition}.csv.png").exists():
            print(f"book_{run_name}_rep{repetition}.csv.png exists, skipping file...")
            continue

        # find parameters
        txt_filename = f"parameters_{run_name}_rep{repetition}.txt"
        txt_path = directory / txt_filename

        if txt_path.exists():
            content = txt_path.read_text()

            # Search for the line that looks like "T : [num1, num2, ...]"
            # re.MULTILINE allows ^ to match the start of each line
            match = re.search(r"^T\s*:\s*\[(.*?)\]", content, re.MULTILINE)

            if match:
                # temps as a single string: "0.001, 0.0017..., 0.0052"
                t_string = match.group(1)

                # split by comma
                t_values = [float(val.strip()) for val in t_string.split(",")]

                if t_values:
                    t_first = t_values[0]
                    t_last = t_values[-1]

                    # Calculate the difference (absolute difference, or just last - first)
                    total_deltaT = (t_last - t_first) / init.T0
                    dT = total_deltaT / (init.row_num - 1)

                    print(f"  Extracted T values -> First: {t_first}, Last: {t_last}, Diff: {total_deltaT:.2f}*T0")

                else:
                    print(f"  Warning: T list is empty in {txt_filename}")
                    continue
            else:
                print(f"  Warning: Could not find 'T : [...]' line in {txt_filename}")
                continue
        else:
            print(f"  Warning: Text file {txt_filename} not found.")
            continue

        with csv_path.open() as f:
            reader = csv.reader(f)
            rows = [row for row in reader if row and any(cell.strip() for cell in row)]

        V_column = [row[0] for row in rows]
        I_column = [row[1] for row in rows]
        idx_max = V_column.index(max(V_column))

        # build I vectors from data
        I_vec_forward = np.array([float(I) for I in I_column[0:idx_max + 1]])
        I_vec_backward = np.flip([float(I) for I in I_column[idx_max:]])

        # build V vector
        Vleft = np.array([float(V) for V in V_column[0:idx_max + 1]])

        fig = plt.figure()
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
            plt.title(r"Even |$\Delta$T| = " + f"{round(total_deltaT, 3)}*T0 gradient, Tleft > Tright" + "\n"
                      + f"T diff between sites : dT = {round(dT, 3)}*T0\n"
                      + f"T0 = {init.T0}\n")
        else:
            plt.title(r"Even |$\Delta$T| = " + f"{round(total_deltaT, 2)}*T0 gradient, Tleft < Tright" + "\n"
                      + f"T diff between sites : dT = {round(dT, 3)}*T0\n"
                      + f"T0 = {init.T0}\n")

        # Restored high DPI
        plt.savefig(fname=directory / pic_name, dpi=2100, bbox_inches="tight")

        # Explicitly close the figure and force garbage collection
        plt.close(fig)
        gc.collect()
