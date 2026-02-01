import csv
from pathlib import Path
import datetime
import time
from define_objects import IMPORT_EXPORT
from preparation import (
    prepare_initial_state,
    prepare_table_triplets,
    output_table_triplets,
)

import re
import os
import Functions as F

'''
THIS FILE SHOULD BE NAMED "...Tstd(\d+)..." where (\d+) is an int
A TABLE WILL BE PRODUCED FOR T LIST [Tmid-#std,Tmid-(#-1)std,...Tmid..., Tmid + (#-1)std, Tmid +#std]

for Tmid = T0 + 3*0.55*T0 = 0.00265
maximum total gradient across grid allowed is therefore |ΔT| =3.3T0

for Tmid = T0 + 3*0.4*T0 = 0.0022

SHOULD HAVE IN ITS DIRECTORY A "table_mid.csv" WHICH LOOKS LIKE
#rep | POS | NEG
where for each position the appropriate bounds for dE calc are given
'''

EXPORT_PATH = Path(__file__).parent.parent / "export"
t0 = time.time()
filename = os.path.basename(__file__)


def main(export: IMPORT_EXPORT) -> None:
    loop_count = 100
    T0_unitless = 0.001
    constT = 2.2  # the average temperature is T0_untiless*constT
    T_std_list = [i for i in range(1, 20)]
    flip = False
    row_num = 7

    ### get runname -> check if Tstd is in list
    iteration = re.search(r"Tstd(\d+)", filename).group(1)  # filename should be "compute_table_Tstd%J_20" 1=<%J<=20
    if int(iteration) not in T_std_list:
        raise ValueError()

    iteration = int(iteration)
    ### get pos & neg for this Tstd
    # with open("table_mid.csv") as f:
    #     rows = list(csv.reader(f))
    #     neg = float(rows[iteration - 1][1])
    #     pos = float(rows[iteration - 1][2])
    pos = 0.05
    neg = -0.15

    ### prep tables
    init = prepare_initial_state(loop_count=loop_count,
                                 unitless_T0=T0_unitless,
                                 flip=flip,
                                 periodic_y=True)
    init = F.swap_in_init("row_num", row_num, init)

    T_mid = constT * init.T0
    max_std = 0.4 * init.T0
    T_std = iteration * max_std / 20
    first_site_T = T_mid - ((init.row_num - init.row_num % 2) / 2) * T_std
    T_list_to_compute = [first_site_T + i * T_std for i in range(init.row_num)]
    table_triplets = prepare_table_triplets(init,
                                            T_list_to_compute,
                                            pos_energy_bound=pos,
                                            neg_energy_bound=neg)
    output_table_triplets(table_triplets, export.prepare_table_triplets_file_list[0])


if __name__ == "__main__":
    iter_name = re.search(r"Tstd(\d+)", filename).group(1)  # filename should be "compute_table_Tmid_2_2_Tstd%J_20" %J<20
    main(
        IMPORT_EXPORT(
            plot_results=True,
            prepare_table_triplets_file_list=[EXPORT_PATH / f"table_triplets_Tmid_2_2_std{iter_name}_20.npz"],
            # these are not relevant here
            csv_table_path=EXPORT_PATH / "tmp",
            export_path=EXPORT_PATH / "tmp",
            results_dir_path=EXPORT_PATH / "tmp",
        )
    )
