import csv

import numpy as np
import mpmath as mp
from mpmath import exp, sqrt
from scipy.integrate import quad
from scipy.special import erf
from scipy.optimize import fsolve
import mpmath

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


def Get_current_from_gamma(gamma_list, reaction_index_list, near_right, near_left, row_num, periodic_y):
    # e == 1
    I_right = 0
    I_down = 0

    for i in range(len(gamma_list)):
        try:
            l, m = reaction_index_list[i]
            charge_multiplier = 1
        except ValueError:
            l, m, particles_moved = reaction_index_list[i]
            charge_multiplier = particles_moved

        if gamma_list[i] < 0:
            raise ValueError

        # negative side current
        if ((l in near_left) and m == "to") or ((l in near_right) and m == "from"):
            I_right -= gamma_list[i] * charge_multiplier

        # positive side current
        elif ((l in near_left) and m == "from") or ((l in near_right) and m == "to"):
            I_right += gamma_list[i] * charge_multiplier

        # right isle to isle current
        elif l - m == -1:
            I_right += gamma_list[i] * charge_multiplier

        # left isle to isle current
        elif l - m == 1:
            I_right -= gamma_list[i] * charge_multiplier

        # up isle to isle current
        elif l - m == -row_num:
            I_down -= gamma_list[i] * charge_multiplier

        # up isle to isle current on the y boundary
        elif l - m == row_num * (row_num - 1) and periodic_y:
            I_down -= gamma_list[i] * charge_multiplier

        # down isle to isle current
        elif l - m == row_num:
            I_down += gamma_list[i] * charge_multiplier

        # down isle to isle current
        elif l - m == -row_num * (row_num - 1) and periodic_y:
            I_down += gamma_list[i] * charge_multiplier

    return I_right / (row_num + 1), I_down / (row_num + 1)


def Get_current_map(gamma_list, reaction_index_list, near_right, near_left, row_num, n_list, periodic_y):
    # e == 1
    n = row_num  # row_num

    # the x position of isle k is :  (k % n) + 1
    # the y position of isle k is :  k // n
    # the entry in J with position x and y is J[y][x]
    Jy = np.zeros((n, n + 1))
    Jx = np.zeros((n, n + 1))

    for i in range(len(gamma_list)):
        try:
            l, m = reaction_index_list[i]
            charge_multiplier = 1
        except ValueError:
            l, m, particles_moved = reaction_index_list[i]
            charge_multiplier = particles_moved

        # negative side current
        if ((l in near_left) and m == "to") or ((l in near_right) and m == "from"):
            if m == "to":  # left side
                Jx[l // n][0] -= gamma_list[i] * charge_multiplier
            else:
                Jx[l // n][-1] -= gamma_list[i] * charge_multiplier

        # positive side current
        elif ((l in near_left) and m == "from") or ((l in near_right) and m == "to"):
            if m == "to":  # right side
                Jx[l // n][-1] += gamma_list[i] * charge_multiplier
            else:
                Jx[l // n][0] += gamma_list[i] * charge_multiplier

        # right isle to isle current
        elif l - m == -1:
            Jx[l // n][(l % n) + 1] += gamma_list[i] * charge_multiplier

        # left isle to isle current
        elif l - m == 1:
            Jx[l // n][(l % n) + 1] -= gamma_list[i] * charge_multiplier

        # up isle to isle current
        elif l - m == -row_num:
            Jy[l // n][(l % n) + 1] += gamma_list[i] * charge_multiplier

        elif l - m == row_num * (row_num - 1) and periodic_y:
            Jy[l // n][(l % n) + 1] += gamma_list[i] * charge_multiplier

        # down isle to isle current
        elif l - m == row_num:
            Jy[l // n][(l % n) + 1] -= gamma_list[i] * charge_multiplier

        elif l - m == -row_num * (row_num - 1) and periodic_y:
            Jy[l // n][(l % n) + 1] -= gamma_list[i] * charge_multiplier

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


def Gamma_cp(dE, T, Ec, gap, Rt):
    tanh = mp.tanh(gap / (2 * T))
    # Ej = (hbar/2eRt)(pi*gap/2e)*tanh
    # set h=1 -> hbar = 1/2pi
    # Ej = tanh * gap/8Rt
    Ej = tanh * gap / (8 * Rt)

    # P2(-dE)
    kappa_2 = mp.mpf('4')
    mu = kappa_2 * Ec
    gauss = mp.exp(-((dE + mu) ** 2) / (4 * mu * T))
    gauss = gauss / mp.sqrt(mp.pi * 4 * mu * T)

    # Gamma_cp(dE) = (pi/2hbar)Ej^2 P(-dE)
    # set h=1 -> 2hbar = 1/pi
    # Gamma_cp(dE) = (pi*Ej)^2 P(-dE)
    return float(gauss * (Ej * mp.pi) ** 2)


def qp_integrand(T, dE, Ec, D):
    """Quasi-particle Convolution Integrand."""

    def conv(E, Etag):
        n_E = dos(E, D)
        if n_E == mp.mpf('0'):
            return mp.mpf('0')

        n_Etag = dos(Etag - dE, D)
        if n_Etag == mp.mpf('0'):
            return mp.mpf('0')

        gauss = exp(-((E - Etag - Ec) ** 2) / (4 * Ec * T))
        gauss = gauss / sqrt(mp.pi * 4 * Ec * T)

        return n_E * n_Etag * f(E, T) * (mp.mpf('1') - f(Etag - dE, T)) * gauss

    return conv


def f(x, t):
    """Fermi-Dirac distribution strictly using mpmath for precision preservation."""
    if x / t > mp.mpf('100'):
        return exp(-x / t)
    if x / t < mp.mpf('-100'):
        return mp.mpf('1')
    expon = exp(x / t)
    return mp.mpf('1') / (mp.mpf('1') + expon)


def dos(E, D):
    """Superconducting Density of States."""
    if mpmath.fabs(E) <= D:
        return mp.mpf('0')
    val = E * E - D * D
    if val <= 0:
        return mp.mpf('0')
    return mpmath.fabs(E) / sqrt(val)


def calc_expected_dist_std(T_grad, T0, gap_array, Rt_ij, Ec, periodic_y=True):
    # T should be unitless
    T = np.asarray(T_grad) / T0
    # array_size is n_side^2
    n_side = T.shape[0]
    array_size = n_side * n_side

    # Each site temperature
    T_array = np.repeat(T, n_side)
    # ----------------------------------

    # qp component, std of each site
    sigma = 0.01 * np.sqrt(T_array)

    if np.any(gap_array > 1e-3):
        # cp variance component
        adj_mask = np.zeros((array_size, array_size))
        for i in range(array_size):
            neighbors = neighbour_list(n_side, i, periodic_y)
            for j in neighbors:
                adj_mask[i, j] = 1.0

        # evaluating physical neighbors
        with np.errstate(divide='ignore', invalid='ignore'):
            G_ij = np.where((adj_mask > 0) & (Rt_ij > 0) & (Rt_ij < np.inf), 1.0 / Rt_ij, 0.0)

        # We now use gap_array instead of the scalar 'gap'
        # G_ij is (N^2, N^2), T_col is (N^2, 1), gap_col is (N^2, 1)
        T_col = T_array[:, np.newaxis]
        gap_col = gap_array[:, np.newaxis]

        # Calculate Ej_ij = tanh(gap_i / 2T_i) * gap_i / 8Rt_ij
        Ej_ij = np.tanh(gap_col / (2.0 * T_col)) * gap_col / 8.0 * G_ij

        # <Q^2>_cp,i = sum_j (Ej_ij^2 / 8*Ec^2)
        var_cp = np.sum(Ej_ij ** 2, axis=1) / (8.0 * Ec ** 2)

        # combine variances (Independent error propagation)
        sigma = np.sqrt(sigma**2 + var_cp)

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

    return moment_1  # expected err


def get_mapping(a, b, threshold):
    """Affine mapping generator for tanh-sinh infinity bounds."""
    if a == -mp.inf:
        return lambda t: (b - t / (mp.mpf('1') - t), mp.mpf('1') / ((mp.mpf('1') - t) ** 2))
    elif b == mp.inf:
        return lambda t: (a + t / (mp.mpf('1') - t), mp.mpf('1') / ((mp.mpf('1') - t) ** 2))
    else:
        width = b - a
        if width < threshold:
            return None
        return lambda t: (a + t * width, width)


def exact_bcs_gap(T_array, Delta_0):
    """
    Numerically solves the BCS self-consistency equation for a given array of temperatures.
    Returns the exact Delta(T).
    """
    Tc = Delta_0 / 1.764

    # The BCS integral to minimize:
    # ln(Delta_0 / Delta(T)) = Integral from 0 to infinity of [f(E) / E] dE
    # where f(E) = 1 / (exp(E / kBT) + 1)
    def bcs_integral(Delta, T):
        if Delta <= 0:
            return 1e9
        integrand = lambda E: 2.0 / (np.exp(np.sqrt(E ** 2 + Delta ** 2) / T) + 1.0) / np.sqrt(E ** 2 + Delta ** 2)
        # Integrate up to a cutoff (e.g. 100*Delta_0 is effectively infinity for this converging function)
        val, _ = quad(integrand, 0, 100 * Delta_0)
        return val - np.log(Delta_0 / Delta)

    Delta_exact = []

    for T in T_array:
        if T >= Tc:
            Delta_exact.append(0.0)
        elif T < 0.01 * Tc:
            Delta_exact.append(Delta_0)
        else:
            # Use fsolve to find the Delta that makes the integral equation = 0
            # We use the previous Delta as the initial guess for faster convergence
            guess = Delta_exact[-1] if len(Delta_exact) > 0 and Delta_exact[-1] > 0 else Delta_0
            sol = fsolve(bcs_integral, guess, args=(T,))[0]
            Delta_exact.append(sol)

    return np.array(Delta_exact)
