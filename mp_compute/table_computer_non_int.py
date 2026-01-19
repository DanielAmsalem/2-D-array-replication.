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
A TABLE WILL BE PRODUCED FOR T LIST [T0,T0+Tstd*int,...T0+std*(row_num-1)]
SHOULD HAVE IN ITS DIRECTORY A "table.csv" WHICH LOOKS LIKE
# | POS | NEG
where for each position the appropriate bounds for dE calc are given
'''

EXPORT_PATH = Path(__file__).parent.parent / "export"
t0 = time.time()
filename = os.path.basename(__file__)


def main(export: IMPORT_EXPORT) -> None:
    loop_count = 100
    T0_unitless = 0.001
    flip = False
    row_num = 10

    ### get runname -> check if Tstd is in list
    iteration = float(re.search(r"Tstd(\d+(?:\.\d+)?)", filename).group(1))  # filename should be "compute_table_Tstd%J.H_20" 1=<%J<=20, 1<=H<=9
    T_std = iteration / 20
    # table frac 7 will have
    # 7.2
    # 7.4
    # 7.6
    # 7.8
    # all with the same bounds appropriate for 8
    neg = -0.17
    pos = 0.07

    ### prep tables
    init = prepare_initial_state(loop_count=loop_count,
                                 unitless_T0=T0_unitless,
                                 flip=flip,
                                 periodic_y=True)
    init = F.swap_in_init("row_num", row_num, init)
    T_list_to_compute = [init.T0 + i * init.T0 * T_std for i in range(init.row_num)]
    table_triplets = prepare_table_triplets(init,
                                            T_list_to_compute,
                                            pos_energy_bound=pos,
                                            neg_energy_bound=neg)
    output_table_triplets(table_triplets, export.prepare_table_triplets_file_list[0])


if __name__ == "__main__":
    iter_name = re.search(r"Tstd(\d+(?:\.\d+)?)", filename).group(1) # filename should be "compute_table_Tstd%J.H_20" 1=<%J<=20, 1<=H<=9
    main(
        IMPORT_EXPORT(
            plot_results=True,
            prepare_table_triplets_file_list=[EXPORT_PATH / f"table_triplets_Tstd{iter_name}_20.npz"],
            # these are not relevant here
            csv_table_path=EXPORT_PATH / "tmp",
            export_path=EXPORT_PATH / "tmp",
            results_dir_path=EXPORT_PATH / "tmp",
        )
    )
