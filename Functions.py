import numpy as np
import math

# parameters
n0ne = 0

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
    if J + 1 < n:  # right neighbour
        neighbours += [(I, J + 1)]
    if J > 0:  # left neighbour
        neighbours += [(I, J - 1)]
    if I + 1 < n:  # down neighbour
        neighbours += [(I + 1, J)]
    if I > 0:  # up neighbour
        neighbours += [(I - 1, J)]

    return neighbours


def V_t(n, Qg, vl, vr, C_inverse, VxCix):
    return np.dot(C_inverse, (n * e + e * VxCix(vl, vr)) + Qg)


def gamma(dE, T, Rt):
    global n0ne
    try:
        a = dE / (T * kB)
        exponent = math.exp(a)
        b = -dE / (e * e * Rt * (1 - exponent))
    except OverflowError:
        b = 0
        n0ne += 1
    return b
