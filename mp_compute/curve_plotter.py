import csv
from pathlib import Path
import numpy as np
import time
from define_objects import ExperimentInitialState


def iv_curve_compute_and_save_csv(
        init: ExperimentInitialState,
        filename: str,
        Vleft,
        results,
        repetition: int,
        results_path: Path
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
    with open(results_path / f"book_{filename}_rep{repetition}.csv", "w+") as f:
        file = csv.writer(f)
        for row in range(len(V_doubled)):
            to_write = [
                float(V_doubled[row] / init.Volts),
                float(I_vec_avg[row] / init.Amp),
                float(I_vec_std[row] / init.Amp),
            ]
            file.writerow(to_write)

    return I_vec_avg, I_vec_std


def report_param(init: ExperimentInitialState,
                 filename: str,
                 repetition: int,
                 T: list,
                 expected_error: float,
                 loop_count: int,
                 T_std: float,
                 t0,
                 results_path: Path):
    with open(results_path / f"parameters_{filename}_rep{repetition}.txt", "w") as f:
        f.write(f"repetition {repetition}\n")
        f.write("loop parameters" + "\n")
        f.write("---------------------------------------------" + "\n")
        f.write("row_num : " + str(init.row_num) + "\n")
        f.write("distribute_R : " + str(init.distribute_R) + "\n")
        if init.distribute_R:
            f.write("stdR (exponent) : " + str(init.stdR) + "\n")
        f.write("distribute_C : " + str(init.distribute_C) + "\n")
        if init.distribute_C:
            f.write("sig (normal) : " + str(init.sig) + "\n")
        f.write("e : " + str(init.e) + "\n")
        f.write("C : " + str(init.C_avg) + "\n")
        f.write("Cg : " + str(np.mean(init.Cg)) + "\n")
        f.write("R : " + str(init.R_avg) + "\n")
        f.write("Rg : " + str(init.CondRg) + "\n")
        f.write("default_dt : " + str(init.default_dt) + "\n")
        f.write("time step : " + str(init.timeStep) + "\n")
        f.write("---------------------------------------------" + "\n")
        f.write("\n")
        f.write("\n")
        f.write("loop variables : " + str(loop_count) + "\n")
        f.write("---------------------------------------------" + "\n")
        f.write("these are the raw variances and means\n")
        f.write("<R> : " + str(init.R_avg) + ", std(Rt_ij) : " + str(np.std(init.R_t_ij)) + "\n")
        f.write("<Rt_i> : " + str(np.mean(init.R_t_i)) + ", std(Rt_i) : " + str(np.std(np.array(init.R_t_i))) + "\n")
        f.write("<C> : " + str(init.mean_allCs) + ", std(C) : " + str(init.std_allCs) + "\n")
        f.write("<Cix> : " + str(init.mean_sideCs) + ", std(Cix) : " + str(init.std_sideCs)+ "\n")
        f.write("\n")
        f.write("T0 : " + str(init.T0) + "\n")
        f.write("T_std : " + str(T_std) + "\n")
        f.write("T : " + str(T) + "\n")
        f.write("---------------------------------------------" + "\n")
        f.write(f"flip : {init.flip}")
        f.write("steady_state_rep : " + str(init.Steady_state_rep) + "\n")
        f.write("expected_error : " + str(expected_error) + "\n")
        f.write("resolution : " + str(init.resolution) + "\n")
        end_time = time.time()
        mins = int((end_time - t0) / 60)
        f.write("runtime : " + str(mins) + "m\n")
    return True
