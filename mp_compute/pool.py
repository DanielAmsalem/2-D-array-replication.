from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from models import Export, SteadyStateResult
from mp_compute.gamma_functions import Get_Steady_State
from mp_compute.preparation import (
    prepare_initial_state,
    validate_table_triplets_file,
    prepare_table_triplets,
    output_table_triplets,
)

EXPORT_PATH = Path(__file__).parent.parent / "export"


def main(export: Export) -> None:
    loop_count = 10
    init = prepare_initial_state(loop_count=loop_count)

    if not validate_table_triplets_file(export.prepare_table_triplets_file, init):
        table_triplets = prepare_table_triplets(init)
        output_table_triplets(table_triplets, export.prepare_table_triplets_file)
        table_val = table_triplets[:, 0]
        table_prob = table_triplets[:, 1]
    else:
        table_triplets = np.load(export.prepare_table_triplets_file.as_posix())
        table_val = table_triplets["val"]
        table_prob = table_triplets["prob"]

    V_diff = 4
    steps = 100
    Vleft = np.linspace(
        init.Vright * init.Volts,
        (init.Vright + V_diff) * init.Volts,
        num=steps,
    )
    V_doubled = np.concatenate([Vleft, Vleft[-2::-1]])
    cycles = len(V_doubled)

    with ProcessPoolExecutor() as executor:
        loaded_state_function = partial(
            Get_Steady_State,
            init=init,
            V_cycle=V_doubled,
            cycles=cycles,
            table_val=table_val,
            table_prob=table_prob,
        )

        results: list[SteadyStateResult] = list(
            executor.map(loaded_state_function, range(init.loop_count))
        )

    I_mean_vector = np.zeros(cycles)
    print(I_mean_vector, I_mean_vector.shape)

    for run in results:
        print(I_mean_vector, I_mean_vector.shape, run.I_vec.shape)
        I_mean_vector += run.I_vec / len(results)

    print(I_mean_vector, I_mean_vector.shape)

    I_std_vector = np.std([r.I_vec for r in results])

    print(V_doubled[steps:].shape, I_mean_vector[steps:].shape)

    if export.plot_results:
        plt.plot(
            Vleft / init.Volts,
            I_mean_vector[:steps] / init.Amp,
            label="increasing",
            color="red",
        )
        plt.plot(
            V_doubled[steps:] / init.Volts,
            I_mean_vector[steps:] / init.Amp,
            label="decreasing",
            color="blue",
        )
        plt.xlabel("Voltage")
        plt.ylabel("Current")
        plt.show()


if __name__ == "__main__":
    main(
        Export(
            plot_results=True,
            prepare_table_triplets_file=EXPORT_PATH / "table_triplets.npz",
            results_file=EXPORT_PATH / "tmp",
            report_file=EXPORT_PATH / "tmp",
        )
    )
