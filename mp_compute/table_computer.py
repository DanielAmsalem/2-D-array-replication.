import os
import re

filename = os.path.basename(__file__)
iter_name = re.search(r"Tstd(\d+)", filename).group(1)
Cg = int(re.search(r"Cg(\d+)", filename).group(1))
ratio = int(re.search(r"ratio(\d+)", filename).group(1))
os.environ["OPENBLAS_NUM_THREADS"] = str(ratio)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
num_workers = max(int(total_cpus / ratio) - 2, 1)
print(f"worker number set to {num_workers} ; for {total_cpus} cpus", flush=True)

import csv
from pathlib import Path
import datetime
import time
import sys

# 1. Get the path to the parent directory (mp_compute)
# __file__ is the script, .parent is table_computers, .parent.parent is mp_compute
mp_compute_dir = Path(__file__).resolve().parent.parent

# 2. Add it to the system path
sys.path.append(str(mp_compute_dir))

# 3. NOW you can safely run your custom imports
from define_objects import IMPORT_EXPORT
from preparation import (
    prepare_initial_state,
    prepare_table_triplets,
    output_table_triplets,
)
import Functions as F

'''
THIS FILE SHOULD BE NAMED "...Tstd(\d+)..." where (\d+) is an int
A TABLE WILL BE PRODUCED FOR T LIST [T0,T0+Tstd*int,...T0+std*(row_num-1)]
SHOULD HAVE IN ITS DIRECTORY A "table.csv" WHICH LOOKS LIKE
# | NEG | POS
where for each position the appropriate bounds for dE calc are given
'''

EXPORT_PATH = Path(__file__).parent.parent.parent / "export"
CSV_PATH = Path(__file__).parent.parent / f"table_Cg{Cg}.csv"
t0 = time.time()


def main(export: IMPORT_EXPORT) -> None:
    loop_count = 100
    T0_unitless = 0.001
    flip = False
    row_num = 7

    ### get runname -> check if Tstd is in list
    iteration = int(re.search(r"Tstd(\d+)", filename).group(1))  # filename should be "compute_table_Tstd%J_20_ratio%K"
    T_std = iteration / 20

    ### get pos & neg for this Tstd by searching the first column
    neg = None
    pos = None
    with open(CSV_PATH) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:  # skip empty rows
                continue
            try:
                row_iter = int(row[0])
            except ValueError:
                continue  # skip headers or non-numeric first columns

            if row_iter == iteration:
                neg = float(row[1])
                pos = float(row[2])
                break  # Stop searching once the correct row is found

    if neg is None or pos is None:
        raise ValueError(f"Iteration {iteration} was not found in table.csv")

    ### prep tables
    init = prepare_initial_state(loop_count=loop_count,
                                 unitless_T0=T0_unitless,
                                 flip=flip,
                                 periodic_y=True,
                                 Cg_C_ratio=Cg,
                                 Rg_R_ratio=100,
                                 stdR_R_ratio=0.9,
                                 sigC_C_ratio=0.5)  # init only actually uses init.resolution and init.Ec
    init = F.swap_in_init("row_num", row_num, init)
    T_list_to_compute = [init.T0 + i * init.T0 * T_std for i in range(init.row_num)]
    table_triplets = prepare_table_triplets(init,
                                            T_list_to_compute,
                                            pos_energy_bound=pos,
                                            neg_energy_bound=neg,
                                            max_workers=num_workers)
    output_table_triplets(table_triplets, export.prepare_table_triplets_file_list[0])


if __name__ == "__main__":
    iter_name = re.search(r"Tstd(\d+)", filename).group(1)
    main(
        IMPORT_EXPORT(
            plot_results=True,
            prepare_table_triplets_file_list=[EXPORT_PATH / f"64bit_table_triplets_Tstd{iter_name}_20_Cg_{Cg}.npz"],
            # these are not relevant here
            csv_table_path=EXPORT_PATH / "tmp",
            export_path=EXPORT_PATH / "tmp",
            results_dir_path=EXPORT_PATH / "tmp",
        )
    )
