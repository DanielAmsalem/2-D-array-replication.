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
A TABLE WILL BE PRODUCED FOR
T = T0 + 3*0.55*T0 = 0.00265
where for each position the appropriate bounds for dE calc are given
'''

EXPORT_PATH = Path(__file__).parent.parent / "export"
t0 = time.time()
filename = os.path.basename(__file__)


def main(export: IMPORT_EXPORT) -> None:
    const = 0.00265
    loop_count = 100
    T0_unitless = 0.001
    flip = False
    row_num = 10

    ### get pos & neg for this Tstd
    neg = -0.14
    pos = 0.04

    ### prep tables
    init = prepare_initial_state(loop_count=loop_count,
                                 unitless_T0=T0_unitless,
                                 flip=flip,
                                 periodic_y=True)
    init = F.swap_in_init("row_num", row_num, init)
    T_list_to_compute = [const]
    table_triplets = prepare_table_triplets(init,
                                            T_list_to_compute,
                                            pos_energy_bound=pos,
                                            neg_energy_bound=neg)
    output_table_triplets(table_triplets, export.prepare_table_triplets_file_list[0])


if __name__ == "__main__":
    iter_name = re.search(r"Tstd(\d+)", filename).group(1)  # filename should be "compute_table_Tstd%J_20" 3=<%J<=19
    main(
        IMPORT_EXPORT(
            plot_results=True,
            prepare_table_triplets_file_list=[EXPORT_PATH / f"table_triplets_T0_265e-5"],
            # these are not relevant here
            csv_table_path=EXPORT_PATH / "tmp",
            export_path=EXPORT_PATH / "tmp",
            results_dir_path=EXPORT_PATH / "tmp",
        )
    )
