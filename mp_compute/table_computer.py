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
    T_std_list = [i for i in range(3, 20)]

    ### get runname -> check if Tstd is in list
    iteration = re.search(r"Tstd(\d+)", filename).group(1)  # filename should be "compute_table_Tstd%J_20" 3=<%J<=19
    T_std = iteration / 20
    if iteration not in T_std_list:
        raise ValueError()

    ### get pos & neg for this Tstd
    with open("table.csv") as f:
        rows = list(csv.reader(f))
        neg = rows[iteration - 3][1]
        pos = rows[iteration - 3][2]

    ### prep tables
    init = prepare_initial_state(loop_count=loop_count,
                                 unitless_T0=T0_unitless)
    T_list_to_compute = [init.T0 + i * init.T0 * T_std for i in range(init.row_num)]
    table_triplets = prepare_table_triplets(init,
                                            T_list_to_compute,
                                            pos_energy_bound=pos,
                                            neg_energy_bound=neg)
    output_table_triplets(table_triplets, export.prepare_table_triplets_file_list[0])

    date_ = datetime.datetime.now()
    run_name = date_.strftime("%Y%m%d, %Hh%Mm%Ss")


if __name__ == "__main__":
    iter_name = re.search(r"Tstd(\d+)", filename).group(1)  # filename should be "compute_table_Tstd%J_20" 3=<%J<=19
    main(
        IMPORT_EXPORT(
            plot_results=True,
            prepare_table_triplets_file_list=[EXPORT_PATH / f"table_triplets_Tstd{iter_name}_20.npz"],
            csv_table_path=EXPORT_PATH / "tmp",
        )
    )
