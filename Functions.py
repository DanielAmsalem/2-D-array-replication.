import numpy as np
from mpmath import quad, ninf, inf, mp, exp, sqrt

# parameters
kB = 1
e = 1
taylor_limit = 0.0001
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


def gamma(dE, Temp, Rt):
    """
    dE is a strictly negative real number; dE<0
    """
    try:
        beta = 1 / (Temp * kB)
        a = dE * beta
    except OverflowError:  # T may be too small
        return ValueError

    exponent = np.exp(a)
    const = e * e * Rt

    # for an a smaller than -0.0001 we do not expect non-regular behaviour
    if a <= -taylor_limit:
        return isNonNegative(float(-dE / (const * (1 - exponent))))
    # for 0 > a > -0.0001, we expand by x/(1-e^x) = -1 + x/2 - x^2/12 + x^4/720 + O(x^6)
    elif -taylor_limit < a < 0:
        print("expand")
        return isNonNegative(taylor(dE, beta) / const)
    else:
        raise ValueError


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
        if gamma_list[i] > 100:
            print(gamma_list[i])
            print(reaction_index[i])
            exit()
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

        if I_right > 100:
            print(I_right)
            print(gamma_list[i])
            print(reaction_index[i])
            exit()

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
    if abs(x - mu) > 7 * sigma_squared:
        return 0
    return exp(-(x - mu) ** 2 / (2 * sigma_squared)) / sqrt(2 * np.pi * sigma_squared)


def integrand_gauss(x, temperature, dE, Ec):
    ## x=dE
    if x == 0:
        return temperature
    if x / temperature < -12:  #x*fermi(x) will give 10^-4
        return 0
    result = high_impedance_p(x + dE, Ec, temperature) * x / (1 - exp(-x / temperature))
    return result


def make_integrand(temp1, val1, Ec):
    def f1(x):
        return integrand_gauss(x, temp1, val1, Ec)

    return f1


def energy_negative_enough(mean, s, T):
    #return true if h(x) will be big enough
    fictive_x = mean + 3 * s
    if fictive_x == 0:  # at x=0 h(0) = T
        if T < 1e-2:
            return False
        else:
            return True
    h = fictive_x / (1 - np.exp(-fictive_x / T))
    if h > 1e-2:
        return True
    else:
        #print("c")
        return False

def approximate_gamma_integral(dE, Temperature):
    global table_val
    global table_prob
    global table_T

    temp_idx = np.where(T == Temperature)[0][0]

    sorted_vals = table_val[temp_idx::len(T)]
    probs = table_prob[temp_idx::len(T)]

    idx = bisect.bisect_left(sorted_vals, dE)

    # Compare the two closest values: sorted_list[idx - 1] and sorted_list[idx]
    if idx == 0:
        if abs(dE - sorted_vals[0]) <= abs(dE - sorted_vals[1]):
            return probs[0]
        else:
            return probs[1]
    elif idx == len(sorted_vals):
        if abs(dE - sorted_vals[-1]) <= abs(dE - sorted_vals[-2]):
            return probs[-1]
        else:
            return probs[-2]
    else:
        if abs(sorted_vals[idx - 1] - dE) <= abs(sorted_vals[idx] - dE):
            return probs[idx - 1]
        else:
            return probs[idx]
