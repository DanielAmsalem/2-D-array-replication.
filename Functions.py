import numpy as np
import math
import random

# parameters
import Conditions

row_num = Conditions.row_num
array_size = row_num**2

kB = 1
e = 1  # in coulon


def neighbour_list(n, i):  # return positions of neighbours of ith position in nxn matrix
    x = i % row_num
    y = (i - i % row_num) / n
    neighbours = []
    if x + 1 < n - 1:
        neighbours += [i + 1]
    if x - 1 > 0:
        neighbours += [i - 1]
    if y + 1 < n - 1:
        neighbours += [i + n]
    if y - 1 > 0:
        neighbours += [i - n]

    return neighbours


def diff(dbl):
    return dbl[0] is dbl[1]


def random_sum_to(n, num_terms=None):  # return fixed sum randomly distributed
    num_terms = (num_terms or random.randint(2, n)) - 1
    a = random.sample(range(1, n), num_terms) + [0, n]
    list.sort(a)
    return [a[j + 1] - a[j] for j in range(len(a) - 1)]


def V_t(n, Q, vl, vr):
    return np.dot(Conditions.C_inverse, (n*e + Conditions.VxCix(vl, vr)) + Q)


def gamma(dE, T, Rt):
    try:
        a = dE / (T * kB)
        exponant = math.exp(a)
        b = -dE / (e * e * Rt * (1 - exponant))
    except OverflowError:
        b = 0
    return b
