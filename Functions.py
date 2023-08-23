import numpy as np
import math

# parameters
kB = 1
e = 1


def neighbour_list(n, i):  # return positions of neighbours of ith position in nxn matrix
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


def return_neighbours(n, I, J):  # Return positions of neighbours of (i,j) in nxn matrix
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


def V_t(n, Qg, vl, vr, C_inverse, VxCix):
    return np.dot(C_inverse, e * n + VxCix(vl, vr) + Qg)


def isNonNegative(x):
    if x < 0:
        raise ValueError
    else:
        return x


def taylor(x, T):
    # 4th degree expansion of dE/(1-e^(dE*beta))
    return float(
        T -
        x / 2 +
        T * math.pow(x, 2) / 12 -
        math.pow(T, 3) * math.pow(x, 4) / 720 +
        math.pow(T, 5) * math.pow(x, 6) / 30240)


def gamma(dE, T, Rt):
    # dE is strictly negative; dE<0
    try:
        beta = 1 / (T * kB)
        a = dE * beta
    except OverflowError:  # T may be too small
        return NameError

    exponent = math.exp(a)
    const = e * e * Rt

    taylor_limit = 0.001

    # for an a smaller than -0.0001 we do not expect non-regular behaviour
    if a <= -taylor_limit:
        return isNonNegative(float(-dE / (const * (1 - exponent))))
    # for 0 > a > -0.0001, we expand by x/(1-e^x) = -1 + x/2 - x^2/12 + x^4/720 + O(x^6)
    elif -taylor_limit < a < 0:
        return isNonNegative(taylor(dE, beta) / const)
    else:
        raise ValueError


def update_statistics(value, avg, n_var, total_time, time_step):
    # from https://github.com/kasirershaharbgu/random_2D_tunneling_arrays/blob/main/random_2d_array_simulation.py#L1957
    new_time = total_time + time_step
    dist_from_avg = value - avg
    local_std = dist_from_avg * time_step / new_time
    new_n_var = n_var + dist_from_avg * total_time * local_std
    new_avg = avg + local_std
    return new_avg, new_n_var


def return_Qn_for_n(n, VxCix, Rg, Cg, islands, Tau):
    Qn = []
    for i in islands:
        summ = 0
        for j in islands:
            summ += Tau[i][j] * (e * n[j] + VxCix[j]) / Cg[j]
        Qn += [(summ / Rg[i]) - e * n[i] - VxCix[i]]
    return Qn

def Get_current_from_gamma(gamma, reaction_index, near_right, near_left):
    I_right = 0
    I_down = 0
    for i in range(len(gamma)):
        l, m = reaction_index[i]

        # positive side current
        if ((l in near_left) and m =="to") or ((l in near_right) and m =="from"):
            I_right += gamma[i]

        # negative side current
        elif ((l in near_left) and m =="from") or ((l in near_right) and m =="to"):
            I_right -= gamma[i]

        # right isle to isle current
        elif l-m == -1:
            I_right += gamma[i]

        # left isle to isle current
        elif l-m == 1:
            I_right -= gamma[i]

        # up isle to isle current
        elif l-m == 4:
            I_down -= gamma[i]

        # down isle to isle current
        elif l-m == -4:
            I_down += gamma[i]

    return I_right, I_down