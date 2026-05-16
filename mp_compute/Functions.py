import csv

import numpy as np
import mpmath as mp
from mpmath import exp, sqrt
from scipy.integrate import quad
from scipy.special import erf

# parameters
from define_objects import ExperimentInitialState
from dataclasses import replace


def flattenToColumn(a):
    """
    Returns the given array, reshaped into a column array.
    :param a: (N,M) numpy array.
    :return: (N*M,1) numpy array.
    """
    return a.reshape((a.size, 1))


def neighbour_list(n, i, periodic_y):
    """
    :param n: integer.
    :param i: integer
    :param periodic_y: bool
    :return: positions of neighbours of ith position in nxn matrix
    position of denoted as [(0,...,n-1),(n...2n-1),...(n(n-1),...n^2-1)]
    """
    x = i % n
    y = (i - i % n) / n
    neighbours = []
    if x + 1 <= n - 1:
        neighbours += [i + 1]
    if x - 1 >= 0:
        neighbours += [i - 1]
    if y + 1 <= n - 1:
        neighbours += [i + n]
    elif y == n and periodic_y:
        neighbours += [i - n * (n - 1)]
    if y - 1 >= 0:
        neighbours += [i - n]
    elif y == 0:
        neighbours += [i + n * (n - 1)]

    return neighbours


def return_neighbours(n, I, J):
    """
    :param n: integer
    :param I: integer
    :param J: integer
    :return: positions of neighbours of (i,j) in nxn matrix
    """
    I, J = int(I), int(J)
    neighbours = []
    if I + 1 < n:  # right neighbour
        neighbours += [(I + 1, J)]
    if I > 0:  # left neighbour
        neighbours += [(I - 1, J)]
    if J + 1 < n:  # down neighbour
        neighbours += [(I, J + 1)]
    if J > 0:  # up neighbour
        neighbours += [(I, J - 1)]

    return neighbours


def getVoltage(n, Qg, C_inverse, VxCix, e):
    return np.dot(C_inverse, e * n + VxCix + Qg)


def isNonNegative(x):
    if x < 0:
        raise ValueError
    else:
        return x


def update_statistics(value, avg, n_var, total_time, time_step):
    # from https://github.com/kasirershaharbgu/random_2D_tunneling_arrays/blob/main/random_2d_array_simulation.py#L1957
    # "Updating the statistics of a measured value according to West's
    #         algorithm (as described in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2973983/)"
    new_time = total_time + time_step
    dist_from_avg = value - avg

    local_std = dist_from_avg * time_step / new_time

    new_n_var = n_var + dist_from_avg * total_time * local_std
    new_avg = avg + local_std
    return new_avg, new_n_var


def Get_current_from_gamma(gamma_list, reaction_index, near_right, near_left, row_num, periodic_y):
    # e == 1
    I_right = 0
    I_down = 0
    for i in range(len(gamma_list)):
        l, m = reaction_index[i]  # electron in isle l moved to isle m

        if gamma_list[i] < 0:
            raise ValueError
        # negative side current
        if ((l in near_left) and m == "to") or ((l in near_right) and m == "from"):
            I_right -= gamma_list[i]

        # positive side current
        elif ((l in near_left) and m == "from") or ((l in near_right) and m == "to"):
            I_right += gamma_list[i]

        # right isle to isle current
        elif l - m == -1:
            I_right += gamma_list[i]

        # left isle to isle current
        elif l - m == 1:
            I_right -= gamma_list[i]

        # up isle to isle current
        elif l - m == -row_num:
            I_down -= gamma_list[i]

        # up isle to isle current on the y boundary
        elif l - m == row_num * (row_num - 1) and periodic_y:
            I_down -= gamma_list[i]

        # down isle to isle current
        elif l - m == row_num:
            I_down += gamma_list[i]

        # down isle to isle current
        elif l - m == -row_num * (row_num - 1) and periodic_y:
            I_down += gamma_list[i]

    return I_right / (row_num + 1), I_down / (row_num + 1)


def Get_current_map(gamma_list, reaction_index, near_right, near_left, row_num, n_list, periodic_y):
    # e == 1
    n = row_num  # row_num

    # the x position of isle k is :  (k % n) + 1
    # the y position of isle k is :  k // n
    # the entry in J with position x and y is J[y][x]
    Jy = np.zeros((n, n + 1))
    Jx = np.zeros((n, n + 1))
    with open("map.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(n_list)
        for j in range(len(gamma_list)):
            entry = gamma_list[j]
            l, m = reaction_index[j]
            writer.writerow([l, m, entry])
    for i in range(len(gamma_list)):
        l, m = reaction_index[i]  # electron in isle l moved to isle m

        # negative side current
        if ((l in near_left) and m == "to") or ((l in near_right) and m == "from"):
            if m == "to":  # left side
                Jx[l // n][0] -= gamma_list[i]
            else:
                Jx[l // n][-1] -= gamma_list[i]

        # positive side current
        elif ((l in near_left) and m == "from") or ((l in near_right) and m == "to"):
            if m == "to":  # right side
                Jx[l // n][-1] += gamma_list[i]
            else:
                Jx[l // n][0] += gamma_list[i]

        # right isle to isle current
        elif l - m == -1:
            Jx[l // n][(l % n) + 1] += gamma_list[i]

        # left isle to isle current
        elif l - m == 1:
            Jx[l // n][(l % n) + 1] -= gamma_list[i]

        # up isle to isle current
        elif l - m == -row_num:
            Jy[l // n][(l % n) + 1] += gamma_list[i]

        elif l - m == row_num * (row_num - 1) and periodic_y:
            Jy[l // n][(l % n) + 1] += gamma_list[i]

        # down isle to isle current
        elif l - m == row_num:
            Jy[l // n][(l % n) + 1] -= gamma_list[i]

        elif l - m == -row_num * (row_num - 1) and periodic_y:
            Jy[l // n][(l % n) + 1] -= gamma_list[i]

    return Jx, Jy


def developQ(Q, dt, n, VxCix, init: ExperimentInitialState):
    # gate charge relaxation, for dQ/dt=inv_tau*Q + b, b=-inv_tau*Qn
    b = -init.C_inv.dot(n + VxCix) / init.CondRg

    # exponent for time step
    exponent = np.exp(init.InvTauEigenValues * dt)

    # basis change
    Q_in_eigenbasis, b = init.InvTauEigenVectorsInv.dot(Q), init.InvTauEigenVectorsInv.dot(b)

    # solution in time
    Q_new_in_eigenbasis = (exponent * Q_in_eigenbasis) + (b / init.InvTauEigenValues) * (exponent - 1)

    # revert to old basis
    return init.InvTauEigenVectors.dot(Q_new_in_eigenbasis)


def return_Qn_for_n(n, VxCix, init: ExperimentInitialState):
    """
    returns Qn for given n vector of array (NxN)
    :param init:
    :param n: (1,N) numpy array
    :param VxCix: (1,N) numpy array
    :return:
    """
    # sum = Tau.dot(n_prime / Cg) / Rg
    # #return sum - n_prime
    n_prime = init.e * n + VxCix
    return init.matrixQnPart.dot(n_prime)


def getWork(i, j, C_inv, curr_V, e):
    Work = e * (2 * curr_V[j] + e * C_inv[j][i] - e * C_inv[j][j] - (
            2 * curr_V[i] + e * C_inv[i][i] - e * C_inv[i][j])) / 2
    return Work


def integrand(T, dE, Ec):
    """
    P- function for high impedance.
    :param dE: Energy difference == dE.
    :param Ec: Electrostatic energy of environment == Ec.
    :param T: Temperature.
    :return: P(E)*E*f_BE(-E)
    """

    def conv(E):
        # mp.fabs instead of np.abs
        if mp.fabs(E) < 1e-8:
            # mp.exp, mp.sqrt, and mp.pi
            zero_limit_gauss = mp.exp(-((dE + Ec) ** 2) / (4 * Ec * T))
            return zero_limit_gauss * mp.sqrt(T / (4 * mp.pi * Ec))

        gauss = mp.exp(-((E + dE + Ec) ** 2) / (4 * Ec * T))
        gauss = gauss / mp.sqrt(mp.pi * 4 * Ec * T)

        bose_mean = E / (1 - mp.exp(-E / T))

        return bose_mean * gauss

    return conv


def contains_allclose(needles, haystack, rtol=1e-8, atol=1e-8):
    """Return True if every value in `needles` is close to some value in `haystack`."""
    for t in needles:
        if not np.any(np.isclose(t, haystack, rtol=rtol, atol=atol)):
            print(t, haystack)
            for h in haystack:
                print(h - t)
            return False
    return True


def get_VxCix(Vl, Vr, array_size, near_left, near_right, Cix):
    _VxCix = np.zeros(array_size)
    for u in near_left:
        _VxCix[u] = Cix[u] * Vl
    for u in near_right:
        _VxCix[u] = Cix[u] * Vr
    return np.array(_VxCix)


def orjson_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError


def unique_significant_floats(list_of_lists, rtol=1e-5, atol=1e-8):
    # Flatten to one array
    arr = np.array(list_of_lists).ravel()

    # Sort so close values are adjacent
    arr = np.sort(arr)

    # Keep only values that differ by more than the tolerance
    unique = [arr[0]]
    for x in arr[1:]:
        if not np.isclose(x, unique[-1], rtol=rtol, atol=atol):
            unique.append(x)
    return np.array(unique).tolist()


def has_neg(n):
    if len(n[n < 0]) > 0:
        return True
    return False


def fix_types(init: ExperimentInitialState, loop_count: int):
    return ExperimentInitialState(
        e=int(init.e),
        kB=float(init.kB),

        Tau_inv=np.array(init.Tau_inv),
        InvTauEigenValues=np.array(init.InvTauEigenValues),
        InvTauEigenVectorsInv=np.array(init.InvTauEigenVectorsInv),
        InvTauEigenVectors=np.array(init.InvTauEigenVectors),
        matrixQnPart=np.array(init.matrixQnPart),
        Cix=np.array(init.Cix),

        array_size=int(init.array_size),
        row_num=int(init.row_num),
        loop_count=loop_count,

        islands=[int(isle) for isle in init.islands],
        near_left=[int(isle) for isle in init.near_left],
        near_right=[int(isle) for isle in init.near_right],

        distribute_R=bool(init.distribute_R),
        distribute_C=bool(init.distribute_C),

        R_t_ij=np.array(init.R_t_ij),
        R_t_i=list(init.R_t_i),

        CondRg=float(init.CondRg),
        Rg=np.array(init.Rg),
        Cg=np.array(init.Cg),
        C_avg=float(init.C_avg),
        R_avg=float(init.R_avg),

        default_dt=float(init.default_dt),
        Tau=np.array(init.Tau),
        C_inv=np.array(init.C_inv),
        timeStep=float(init.timeStep),

        Volts=float(init.Volts),
        Amp=float(init.Amp),
        Vright=float(init.Vright),
        max_count=int(init.max_count),

        T0=float(init.T0),
        Ec=float(init.Ec),
        resolution=float(init.resolution),

        Steady_state_rep=int(init.Steady_state_rep),
        sig=float(init.sig),
        stdR=float(init.stdR),
        mean_allCs=float(init.mean_allCs),
        mean_sideCs=float(init.mean_sideCs),
        std_allCs=float(init.std_allCs),
        std_sideCs=float(init.std_sideCs),
        flip=bool(init.flip),
        periodic_y=bool(init.periodic_y),
    )


def swap_in_init(categry: str, suggestion, init: ExperimentInitialState):
    # check if attribute exists
    if not hasattr(init, categry):
        raise AttributeError(f"{categry} is not a valid category")

    # check type
    old_val = getattr(init, categry)
    if not isinstance(suggestion, type(old_val)):
        print(type(suggestion), type(old_val))
        raise TypeError(f"{categry} is not a valid suggestion and is of type {type(suggestion)}, "
                        f"should be {type(old_val)}")

    return replace(init, **{categry: suggestion})


def change_top_and_bottom_rows_to_insulate(R_t_ij, insulate_R):
    R_new = R_t_ij.copy()
    array_size = R_t_ij.shape[0]
    row_num = sqrt(array_size)
    for i in range(array_size):
        for j in range(array_size):
            y_i = i - i % row_num
            y_j = j - j % row_num
            ## if in row 0 or n-1 and same row neighbours
            if (y_i == 0 or y_i == array_size - row_num) and y_i == y_j:
                if abs(i - j) == 1:
                    R_new[i, j] = insulate_R

    return R_new


def Gamma_cp(dE, T, Ec, Ej):
    # P(-dE)
    gauss = mp.exp(-((dE + Ec) ** 2) / (4 * Ec * T))
    gauss = gauss / mp.sqrt(mp.pi * 4 * Ec * T)
    return gauss * Ej * Ej * mp.pi


def calc_expected_dist_std(T_array, T0):
    T = np.asarray(T_array) / T0

    # T_array should be an N*N long list with entries [T0/T0...T0/T0,(T0+dT)/T0...(T0+dt/T0),...(T0+(N-1)dT)/T0]
    # std of each site
    sigma = 0.01 * np.sqrt(T)
    sqrt2_sigma = np.sqrt(2) * sigma

    # CDF of the maximum absolute deviation
    def F_Z(z):
        # np.prod over the array of error functions (independence assumption)
        return np.prod(erf(z / sqrt2_sigma))

    # integrands for the 1st and 2nd moments
    def integrand_1st_moment(z):
        return 1.0 - F_Z(z)

    def integrand_2nd_moment(z):
        return 2.0 * z * (1.0 - F_Z(z))

    # 10 sigma of the hottest site
    upper_limit = np.max(sigma) * 10.0

    # epsrel is tightened slightly for high-precision stability
    moment_1, err_1 = quad(integrand_1st_moment, 0, upper_limit, epsabs=1e-10, epsrel=1e-10)
    moment_2, err_2 = quad(integrand_2nd_moment, 0, upper_limit, epsabs=1e-10, epsrel=1e-10)

    variance_Z = moment_2 - (moment_1 ** 2)
    std_Z = np.sqrt(variance_Z)

    return std_Z  # expected err
