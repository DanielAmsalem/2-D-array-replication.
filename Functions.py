import numpy as np
from mpmath import exp, sqrt

# parameters
from models import ExperimentInitialState


def flattenToColumn(a):
    """
    Returns the given array, reshaped into a column array.
    :param a: (N,M) numpy array.
    :return: (N*M,1) numpy array.
    """
    return a.reshape((a.size, 1))


def neighbour_list(n, i):
    """
    :param n: integer.
    :param i: integer
    :return: positions of neighbours of ith position in nxn matrix
    position of denoted as [(0,...,n-1),(n...2n-1),..(n(n-1),...n^2-1)]
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
    if y - 1 >= 0:
        neighbours += [i - n]

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
    return np.dot(C_inverse, e * n + e * VxCix - Qg)


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


def Get_current_from_gamma(gamma_list, reaction_index, near_right, near_left):
    # e == 1
    I_right = 0
    I_down = 0
    for i in range(len(gamma_list)):
        l, m = reaction_index[i]

        # positive side current
        if ((l in near_left) and m == "to") or ((l in near_right) and m == "from"):
            I_right += gamma_list[i]

        # negative side current
        elif ((l in near_left) and m == "from") or ((l in near_right) and m == "to"):
            I_right -= gamma_list[i]

        # right isle to isle current
        elif l - m == -1:
            I_right += gamma_list[i]

        # left isle to isle current
        elif l - m == 1:
            I_right -= gamma_list[i]

        # up isle to isle current
        elif l - m == 4:
            I_down -= gamma_list[i]

        # down isle to isle current
        elif l - m == -4:
            I_down += gamma_list[i]

    return I_right, I_down


def developQ(Q, dt, n, VxCix, init_state: ExperimentInitialState):
    # gate charge relaxation, for dQ/dt=inv_tau*Q + b
    b = -init_state.Tau_inv.dot(return_Qn_for_n(n, VxCix, init_state))

    # exponent for time step
    exponent = np.exp(init_state.InvTauEigenValues * dt)

    # basis change
    Q_in_eigenbasis, b = (
        init_state.InvTauEigenVectorsInv.dot(Q),
        init_state.InvTauEigenVectorsInv.dot(b),
    )

    # solution in time
    Q_new_in_eigenbasis = (exponent * Q_in_eigenbasis) + (
            b / init_state.InvTauEigenValues
    ) * (exponent - 1)

    # revert to old basis
    return init_state.InvTauEigenVectors.dot(Q_new_in_eigenbasis)


def return_Qn_for_n(n, VxCix, init_state: ExperimentInitialState):
    """
    returns Qn for given n vector of array (NxN)
    :param n: (1,N) numpy array
    :param VxCix: (1,N) numpy array
    :return:
    """
    # sum = Tau.dot(n_prime / Cg) / Rg
    # #return sum - n_prime
    n_prime = init_state.e * n + init_state.e * VxCix
    return init_state.matrixQnPart.dot(n_prime)


def getWork(i, j, C_inv, curr_V, e):
    Work = (
            e
            * (
                    2 * curr_V[j]
                    + e * C_inv[j][i]
                    - e * C_inv[j][j]
                    - (2 * curr_V[i] + e * C_inv[i][i] - e * C_inv[i][j])
            )
            / 2
    )
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
        if np.abs(E) < 1e-8:
            zero_limit_gauss = exp(-((dE + Ec) ** 2) / (4 * Ec * T))
            return zero_limit_gauss * sqrt(T / (4 * np.pi * Ec))

        gauss = exp(-((E + dE + Ec) ** 2) / (4 * Ec * T))
        gauss = gauss / sqrt(np.pi * 4 * Ec * T)

        bose_mean = E / (1 - exp(-E / T))

        return bose_mean * gauss

    return conv


def contains_allclose(needles, haystack, rtol=1e-8, atol=1e-10):
    """Return True if every value in `needles` is close to some value in `haystack`."""
    for t in needles:
        if not np.any(np.isclose(t, haystack, rtol=rtol, atol=atol)):
            return False
    return True


def VxCix(Vl, Vr, array_size, near_left, near_right, Cix):
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
