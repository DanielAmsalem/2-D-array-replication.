from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from models import ExportFiles
from mp_compute.gamma_functions import Get_Steady_State
from mp_compute.preparation import (
    prepare_initial_state,
    validate_table_triplets_file,
    prepare_table_triplets,
    output_table_triplets,
)

EXPORT_PATH = Path(__file__).parent.parent / "export"


def main():
    # Prepare state for computation
    files = ExportFiles(
        prepare_table_triplets_file=EXPORT_PATH / "table_triplets.npz",
        results_file=EXPORT_PATH / "tmp",
        report_file=EXPORT_PATH / "tmp",
    )

    init = prepare_initial_state(loop_count=2)

    if not validate_table_triplets_file(files.prepare_table_triplets_file, init):
        table_triplets = prepare_table_triplets(init)
        output_table_triplets(table_triplets, files.prepare_table_triplets_file)
        table_val = table_triplets[:, 0]
        table_prob = table_triplets[:, 1]
    else:
        table_triplets = np.load(files.prepare_table_triplets_file.as_posix())
        table_val = table_triplets["val"]
        table_prob = table_triplets["prob"]

    V_diff = 4
    steps = 100
    Vleft = np.linspace(
        init.Vright * init.Volts, (init.Vright + V_diff) * init.Volts, num=steps
    )
    V_doubled = np.concatenate([Vleft, Vleft[-2::-1]])
    cycles = len(V_doubled)
    I_matrix = np.zeros((init.loop_count, cycles))

    with ProcessPoolExecutor() as executor:
        loaded_state_function = partial(
            Get_Steady_State,
            init=init,
            V_cycle=V_doubled,
            cycles=cycles,
            table_val=table_val,
            table_prob=table_prob,
        )

        results = list(executor.map(loaded_state_function, range(init.loop_count)))
        print(results)


if __name__ == "__main__":
    main()
