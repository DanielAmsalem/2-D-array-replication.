import numpy as np
from mpmath import quad, ninf, inf, mp, exp, sqrt

# parameters
kB = 1
e = 1


# for debugging Warnings may set np.seterr(all='raise')


def flattenToColumn(a):
    """
    Returns the given array, reshaped into a column array.
    :param a: (N,M) numpy array.
    :return: (N*M,1) numpy array.
    """
    return a.reshape((a.size, 1))


def neighbour_list(n, i):
    """
    :param n: integer     :param i: integer
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
    :param n: integer  :param I: integer  :param J: integer
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


def getVoltage(n, Qg, C_inverse, VxCix):
    return np.dot(C_inverse, e * n + e * VxCix - Qg)


def isNonNegative(x):
    if x < 0:
        raise ValueError
    else:
        return x


def taylor(x, T):
    # 4th degree expansion of x/(1-e^(x*T))
    return float(
        T -
        x / 2 +
        T * np.power(x, 2) / 12 -
        np.power(T, 3) * np.power(x, 4) / 720 +
        np.power(T, 5) * np.power(x, 6) / 30240)


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


def Paper_developQ(Q, dt, InvTauEigenVec, InvTauEigenVal, n,
                   InvTauEigenVecInv, InvTau,
                   C_inverse, VxCix, Rg, Tau, Cg):
    # Legacy version
    # gate charge relaxation, for dQ/dt=inv_tau*Q + b
    b = -C_inverse.dot(e * n + e * VxCix) / Rg

    # exponent for time step
    exponent = np.exp(InvTauEigenVal * dt)

    # basis change
    Q_in_eigenbasis, b = InvTauEigenVecInv.dot(Q), InvTauEigenVecInv.dot(b)

    # solution in time
    Q_new_in_eigenbasis = (exponent * Q_in_eigenbasis) + (b / InvTauEigenVal) * (exponent - 1)

    # revert to old basis
    return InvTauEigenVec.dot(Q_new_in_eigenbasis)


def return_Qn_for_n(n, VxCix, Cg, Rg, Tau):
    """
    returns Qn for given n vector of array (NxN)
    :param n: (1,N) numpy array
    :param VxCix: (1,N) numpy array
    :param Cg: (1,N) numpy array
    :param Rg: (1,N) numpy array
    :param Tau: (N,N) numpy array
    :return:
    """
    return Tau.dot((e * n + e * VxCix) / Cg) / Rg - e * n - e * VxCix
    # return matrixQn.dot((e * n + e * VxCix))


def developQ(Q, dt, InvTauEigenVec, InvTauEigenVal, n,
             InvTauEigenVecInv, InvTau,
             C_inverse, VxCix, Rg, Tau, Cg, matrixQn):
    # gate charge relaxation, for dQ/dt=inv_tau*(Q + b), b = -Qn
    b = -return_Qn_for_n(n, VxCix, Cg, Rg, Tau)

    # exponent for time step
    exponent = np.exp(InvTauEigenVal * dt)

    # basis change
    Q_in_eigenbasis, b_in_eigenbasis = InvTauEigenVecInv.dot(Q), InvTauEigenVecInv.dot(b)

    # solution in time
    Q_new_in_eigenbasis = exponent * (Q_in_eigenbasis + b_in_eigenbasis)

    # revert to old basis
    return InvTauEigenVec.dot(Q_new_in_eigenbasis) - b


def getWork(i, j, C_inv, curr_V):
    Work = e * (2 * curr_V[j] + e * C_inv[j][i] - e * C_inv[j][j] -
                (2 * curr_V[i] + e * C_inv[i][i] - e * C_inv[i][j])) / 2
    return Work


def high_impedance_p(x, mu, Temp):
    """
    P- function for high impedance.
    :param x: function input (energy) == E+dE.
    :param mu: Electrostatic energy of environment == Ec.
    :param Temp: Temperature.
    :return: P(x)
    """
    sigma_squared = 2 * mu * Temp
    mu = -mu
    return exp(-(x - mu) ** 2 / (2 * sigma_squared)) / sqrt(2 * np.pi * sigma_squared)


def integrand_gauss(x, temperature, value, Ec):
    if x == 0:
        return temperature
    result = high_impedance_p(x + value, Ec, temperature) * x / (1 - exp(-x / temperature))
    return result


def make_integrand(temp1, val1, Ec1):
    def f1(x):
        return integrand_gauss(x, temp1, val1, Ec=Ec1)

    return f1


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
            zero_limit_gauss = exp(-(dE + Ec) ** 2 / (4*Ec*T))
            return zero_limit_gauss * sqrt(T/(4*np.pi*Ec))

        gauss = exp(-(E + dE + Ec) ** 2 / (4*Ec*T))
        gauss /= sqrt(np.pi * 4*Ec*T)

        bose_mean = E / (1 - exp(-E / T))

        return bose_mean*gauss

    return conv
