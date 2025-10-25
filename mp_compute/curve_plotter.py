import csv

import numpy as np
from matplotlib import pyplot as plt

from models import ExperimentInitialState, Export
from mp_compute.gamma_functions import Get_Steady_State


def iv_curve_plotter(
    init: ExperimentInitialState,
    files: Export,
    steps: int,
    V_diff: int,
) -> None:
    Vleft = np.linspace(
        init.Vright * init.Volts, (init.Vright + V_diff) * init.Volts, num=steps
    )
    V_doubled = np.concatenate([Vleft, Vleft[-2::-1]])

    # results matrix, ith column has the ith loop, jth row is the jth step of voltage
    cycles = len(V_doubled)
    I_matrix = np.zeros((init.loop_count, cycles))

    for loop in range(init.loop_count):
        I_matrix[loop] = Get_Steady_State(V_doubled, loop)

    I_vec_avg = np.zeros(cycles)  # results vector
    for run in I_matrix:
        I_vec_avg += run / len(I_matrix)

    I_vec_var = np.zeros(cycles)  # errors vector
    for run_num in range(len(I_matrix)):
        I_vec_var += np.abs(I_matrix[run_num] - I_vec_avg) ** 2 / len(I_matrix)

    I_vec_std = np.sqrt(I_vec_var)
    # w+ truncates file
    outfile_name = files.results_file.name

    with open(f"book_{outfile_name}.csv", "w+") as f:
        file = csv.writer(f)
        for row in range(len(V_doubled)):
            to_write = [
                float(V_doubled[row] / init.Volts),
                float(I_vec_avg[row] / init.Amp),
                float(I_vec_std[row] / init.Amp),
            ]
            file.writerow(to_write)

    plot = True
    if plot:
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
        if not (init.distribute_C or init.distribute_R):
            plt.title("IV through ordered lattice\n" + outfile_name)
        elif init.distribute_C and init.distribute_R:
            plt.title(
                outfile_name
                + "\n"
                + "<R> = "
                + str(round(init.R_avg, 2))
                + ", Rg = "
                + str(round(init.Rg / init.R_avg, 2))
                + "<R>, "
                + "<C> = "
                + str(round(init.C_avg, 2))
                + ", Cg = "
                + str(round(init.Cg / init.C_avg, 2))
                + "<C>"
            )
        elif init.distribute_C and not init.distribute_R:
            plt.title(
                outfile_name
                + "\n"
                + "<C> = "
                + str(round(init.C_avg, 2))
                + ", Cg = "
                + str(round(init.Cg / init.C_avg, 2))
                + "<C>"
            )
        elif init.distribute_R and not init.distribute_C:
            plt.title(
                outfile_name
                + "\n"
                + "<R> = "
                + str(round(init.R_avg, 2))
                + ", Rg = "
                + str(round(init.Rg / init.R_avg, 2))
                + "<R>"
            )
        plt.legend()
        plt.savefig(outfile_name + ".png", dpi=2100, bbox_inches="tight")
