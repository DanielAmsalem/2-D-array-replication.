import csv

import numpy as np

from models import ExperimentInitialState


def iv_curve_computer(
    init: ExperimentInitialState,
    filename: str,
    Vleft,
    results,
    repetition: int,
):
    V_doubled = np.concatenate([Vleft, Vleft[-2::-1]])

    # results matrix, ith column has the ith loop, jth row is the jth step of voltage
    cycles = len(V_doubled)
    I_vec_avg = np.zeros(cycles)

    for run in results:
        I_vec_avg += run.I_vec / len(results)

    I_vec_var = np.zeros(cycles)  # variance of current vector
    for run in results:
        I_vec_var += np.abs(run.I_vec - I_vec_avg) ** 2 / len(results)
    I_vec_std = np.sqrt(I_vec_var)

    # w+ truncates file
    with open(f"book_{filename}_rep{repetition}.csv", "w+") as f:
        file = csv.writer(f)
        for row in range(len(V_doubled)):
            to_write = [
                float(V_doubled[row] / init.Volts),
                float(I_vec_avg[row] / init.Amp),
                float(I_vec_std[row] / init.Amp),
            ]
            file.writerow(to_write)

    return I_vec_avg, I_vec_std
