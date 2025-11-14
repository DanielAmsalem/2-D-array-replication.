from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import datetime

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

from models import Export, SteadyStateResult
from gamma_functions import Get_Steady_State
from preparation import (
    prepare_initial_state,
    validate_table_triplets_file,
    prepare_table_triplets,
    output_table_triplets,
)
import curve_plotter

from dataclasses import asdict
import orjson

EXPORT_PATH = Path(__file__).parent.parent / "export"


def main(export: Export) -> None:
    loop_count = 100
    T0_unitless = 0.001
    T_std = 1
    init = prepare_initial_state(loop_count=loop_count, unitless_T0=T0_unitless)

    date_ = datetime.datetime.now()
    run_name = date_.strftime("%Y%m%d, %Hh%Mm%Ss")

    ### report parameters of run to report file
    outfile = Path(EXPORT_PATH / f"{run_name}.json")
    raw_fields = asdict(init)
    serialized_init_data = orjson.dumps(raw_fields, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")
    outfile.write_text(serialized_init_data)

    T_list_to_compute = [init.T0 + i * init.T0 * T_std for i in range(init.row_num)]
    if not validate_table_triplets_file(export.prepare_table_triplets_file, init,
                                        np.array(T_list_to_compute)):
        table_triplets = prepare_table_triplets(init, T_list_to_compute)
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
            flip=False,
            T=(T := [init.T0 + i * init.T0 * T_std for i in range(init.row_num)]),
            table_T=T,
            expected_error=0.01 * (init.row_num - 1)
        )

        results: list[SteadyStateResult] = list(
            executor.map(loaded_state_function, range(init.loop_count))
        )

    I_vec_avg, I_vec_std = curve_plotter.iv_curve_computer(init=init,
                                                           filename=run_name,
                                                           results=results,
                                                           Vleft=Vleft,
                                                           repetition=0)

    if export.plot_results:
        plt.plot(
            Vleft / init.Volts,
            I_vec_avg[:steps] / init.Amp,
            label="increasing",
            color="red",
        )
        plt.plot(
            V_doubled[steps:] / init.Volts,
            I_vec_avg[steps:] / init.Amp,
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
        )
    )
