import os
import re

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
num_workers = max(total_cpus - 3, 1)
print(f"worker number set to {num_workers} ; for {total_cpus} cpus", flush=True)

import sys
import csv
import time
from pathlib import Path
import numpy as np

filename = os.path.basename(__file__)
mp_compute_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(mp_compute_dir))

from define_objects import IMPORT_EXPORT
from preparation import (
    prepare_initial_state,
    prepare_table_triplets_NIS,  # <-- STEP 2.2: UPDATED IMPORT
    output_table_triplets,
)
import Functions as F

'''
THIS FILE SHOULD BE NAMED "...Tstd(\d+)..." where (\d+) is an int
A TABLE WILL BE PRODUCED FOR T LIST [Tmid-#std,Tmid-(#-1)std,...Tmid..., Tmid + (#-1)std, Tmid +#std]

for Tmid = T0 + 3*0.55*T0 = 0.00265
maximum total gradient across grid allowed is therefore |ΔT| =3.3T0

for Tmid = T0 + 3*0.4*T0 = 0.0022

for Tmid = T0 + 3*0.2*T0 = 0.0016

for Tmid = T0 + 3*0.7*T0 = 0.0031
'''

# name should be Like table_computer_Tmid15_4_Tstd{n}_20_ratio1_Cg10_res100_GAP2_0.py
iter_name = re.search(r"Tstd(\d+)", filename).group(1)
Cg = int(re.search(r"Cg(\d+)", filename).group(1))
res = int(re.search(r"res(\d+)", filename).group(1))

# Extracting the raw strings to preserve exact formatting (e.g. "05" vs "5")
gap_match = re.search(r"GAP(\d+)_(\d+)", filename)
gap_int_str = gap_match.group(1)
gap_tenth_str = gap_match.group(2)

Tconst_match = re.search(r"Tmid(\d+)_(\d+)", filename)
Tconst_units_str = Tconst_match.group(1)
Tconst_tenth_str = Tconst_match.group(2)

# Converting raw string captures to physical floats
gap_ratio = float(f"{gap_int_str}.{gap_tenth_str}")
Tconst = float(f"{Tconst_units_str}.{Tconst_tenth_str}")

print(f"gap_ratio is : {gap_ratio}", flush=True)
print(f"Tconst is {Tconst}", flush=True)

# =========================================================================
# PATHS
# =========================================================================
EXPORT_PATH = Path(__file__).parent.parent.parent / "export"
# Rebuilding paths using the exact string captures to prevent file mismatches
CSV_PATH = Path(
    __file__).parent.parent / f"gapped_table_Tmid{Tconst_units_str}_{Tconst_tenth_str}_Cg{Cg}_D{gap_int_str}_{gap_tenth_str}.csv"
t0 = time.time()


def main(export: IMPORT_EXPORT) -> None:
    loop_count = 100
    T0_unitless = 0.001
    flip = False
    row_num = 7

    ### get runname -> check if Tstd is in list
    iteration = int(iter_name)

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
        raise ValueError(f"Iteration {iteration} was not found in {CSV_PATH.name}")

    ############## prep init ##########
    init = prepare_initial_state(loop_count=loop_count,
                                 unitless_T0=T0_unitless,
                                 flip=flip,
                                 periodic_y=True,
                                 Cg_C_ratio=Cg,
                                 Rg_R_ratio=100,
                                 stdR_R_ratio=0.9,
                                 sigC_C_ratio=0.5)  # only init.Ec and init.resolution are relevant
    init = F.swap_in_init("row_num", row_num, init)
    # for gapped compute we need to lower the resolution by an order of magnitude:
    norm_metal_resolution = init.resolution
    init = F.swap_in_init("resolution", norm_metal_resolution * res, init)
    print(f"set resolution times {res} to res={norm_metal_resolution * res}", flush=True)
    ##################################

    T_mid = Tconst * init.T0
    center_idx = (init.row_num - 1) / 2.0

    max_std = (T_mid - init.T0) / center_idx
    T_std = iteration * max_std / 20
    first_site_T = T_mid - (center_idx * T_std)
    T_list_to_compute = [first_site_T + i * T_std for i in range(init.row_num)]
    print(f"T_list_to_compute = {T_list_to_compute}")
    print(f"in units of T0 : {np.array(T_list_to_compute) / init.T0}", flush=True)

    # <-- STEP 2.3: ROUTED EXECUTION TO NIS
    table_triplets = prepare_table_triplets_NIS(
        init_state=init,
        expected_list=T_list_to_compute,
        pos_energy_bound=pos,
        neg_energy_bound=neg,
        max_workers=num_workers,
        gap_ratio=gap_ratio,
        midfix=True
    )
    output_table_triplets(table_triplets, export.prepare_table_triplets_file_list[0])


if __name__ == "__main__":
    # Appended Cg to the gapped export string to keep your data organized
    # using exact string captures to maintain file format integrity

    # <-- STEP 2.4: UPDATED OUTPUT FILENAME TO INCLUDE "NIS_table_triplets"
    export_filename = (f"64bit_GAP{gap_int_str}_{gap_tenth_str}_"
                       f"NIS_table_triplets_Tmid{Tconst_units_str}_{Tconst_tenth_str}_Tstd{iter_name}_20_Cg_{Cg}.npz")

    main(
        IMPORT_EXPORT(
            plot_results=True,
            prepare_table_triplets_file_list=[EXPORT_PATH / export_filename],
            # these are not relevant here
            csv_table_path=EXPORT_PATH / "tmp",
            export_path=EXPORT_PATH / "tmp",
            results_dir_path=EXPORT_PATH / "tmp",
        )
    )