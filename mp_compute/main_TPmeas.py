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
import Functions as F
import curve_plotter

from dataclasses import asdict
import orjson
import csv

EXPORT_PATH = Path(__file__).parent.parent / "export"


def main(export: Export) -> None:
    loop_count = 5
    T0_unitless = 0.001
    T_std = 1/20
    init = prepare_initial_state(loop_count=loop_count, unitless_T0=T0_unitless)

    date_ = datetime.datetime.now()
    run_name = date_.strftime("%Y%m%d, %Hh%Mm%Ss")

    ### report parameters of run to report file
    outfile = Path(EXPORT_PATH / f"{run_name}.json")
    raw_fields = asdict(init)
    serialized_init_data = orjson.dumps(raw_fields, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")
    outfile.write_text(serialized_init_data)

    full_expected_list = []
    for k in range(21):
        temps = [init.T0 + i * k * init.T0 * T_std for i in range(init.row_num)]
        full_expected_list.append(temps)

    full_expected_list = F.unique_significant_floats(full_expected_list, rtol=1e-5, atol=1e-8)

    if not validate_table_triplets_file(export.prepare_table_triplets_file, init, np.array(full_expected_list)):
        full_expected_list_over_T0 = [x/init.T0 for x in full_expected_list]
        table_triplets = prepare_table_triplets(init, full_expected_list_over_T0)
        output_table_triplets(table_triplets, export.prepare_table_triplets_file)
        table_val = table_triplets[:, 0]
        table_prob = table_triplets[:, 1]
        table_T = table_triplets[:len(full_expected_list), 2]
    else:
        table_triplets = np.load(export.prepare_table_triplets_file.as_posix())
        table_val = table_triplets["val"]
        table_prob = table_triplets["prob"]
        table_T = table_triplets["temp"]
        table_T = table_T[:len(full_expected_list)] #take only

    V_diff = 4
    steps = 100
    Vleft = np.linspace(
        init.Vright * init.Volts,
        (init.Vright + V_diff) * init.Volts,
        num=steps,
    )
    V_doubled = np.concatenate([Vleft, Vleft[-2::-1]])
    cycles = len(V_doubled)

    ## first run
    with ProcessPoolExecutor() as executor:
        loaded_state_function = partial(
            Get_Steady_State,
            init=init,
            V_cycle=V_doubled,
            cycles=cycles,
            table_val=table_val,
            table_prob=table_prob,
            table_T=table_T,
            T=[init.T0],
            expected_error=0.01 * (init.row_num - 1)
        )

        results: list[SteadyStateResult] = list(
            executor.map(loaded_state_function, range(init.loop_count))
        )

    ## find smallest I(V) > 2
    I_vec_avg, I_vec_std = curve_plotter.iv_curve_computer(init=init,
                                                           filename=run_name,
                                                           results=results,
                                                           Vleft=Vleft,
                                                           repetition=0)

    index_2, index_0 = np.searchsorted(I_vec_avg, 2 * init.Amp, side="right"), np.searchsorted(I_vec_avg, 0)
    if index_2 < len(I_vec_avg):
        V0 = Vleft[index_0]
        V2 = Vleft[index_2]  # find voltage for the current at 2 amper
    else:
        raise ValueError("no I(V) larger than 2")

    ## run thermopower until I(V)<0
    current_at_V0 = True
    repetition = 0
    V0_vec = [V0]
    dT_vec = [0]
    while current_at_V0:
        repetition += 1
        if repetition > 20:
            print("too many runs")
            current_at_V0 = False

        T_std = repetition * init.T0 / 20  # new temperature profile

        with ProcessPoolExecutor() as executor:
            loaded_state_function = partial(
                Get_Steady_State,
                init=init,
                V_cycle=V_doubled,
                cycles=cycles,
                table_val=table_val,
                table_prob=table_prob,
                table_T=table_T,
                T=(T := np.linspace(init.T0, init.T0 + init.row_num * T_std, init.row_num)),
                expected_error=0.01 * (init.row_num - 1) * np.sqrt(max(T) / init.T0)
            )

            results: list[SteadyStateResult] = list(
                executor.map(loaded_state_function, range(init.loop_count))
            )

        ## find I(V0)
        I_vec_avg, I_vec_std = curve_plotter.iv_curve_computer(init=init,
                                                               filename=run_name,
                                                               results=results,
                                                               Vleft=Vleft,
                                                               repetition=repetition)

        V0_vec += [float(np.searchsorted(I_vec_avg, 0))]
        dT_vec += [T_std]

        if I_vec_avg[index_2] < 0:
            current_at_V0 = False

    if export.plot_results:
        plt.plot(
            V0_vec / init.Volts,
            dT_vec / init.T0,
            label="Vth(dT) -- slope is S(T)",
            color="red",
        )
        plt.xlabel("Voltage")
        plt.ylabel("Current")
        plt.show()

    with open(f"book_{run_name}_TPgraph.csv", "w+") as f:
        file = csv.writer(f)
        for row in range(len(V0_vec)):
            to_write = [
                float(V0_vec[row] / init.Volts),
                float(dT_vec[row] / init.T0),
            ]
            file.writerow(to_write)

if __name__ == "__main__":
    main(
        Export(
            plot_results=True,
            prepare_table_triplets_file=EXPORT_PATH / "table_triplets.npz",
            results_file=EXPORT_PATH / "tmp",
        )
    )
