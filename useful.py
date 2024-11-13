

import random
def random_sum_to(n, num_terms=None):  # return fixed sum randomly distributed
    num_terms = (num_terms or random.randint(2, n)) - 1
    a = random.sample(range(1, n), num_terms) + [0, n]
    list.sort(a)
    return [a[j + 1] - a[j] for j in range(len(a) - 1)]

Qg = F.Paper_developQ(Qg, dt, Cond.InvTauEigenVectors, Cond.InvTauEigenValues,
                                  n, Cond.InvTauEigenVectorsInv, Cond.Tau_inv, C_inv, VxCix,
                                  Rg, Tau, Cg)

Qg = F.developQ(Qg, dt, Cond.InvTauEigenVectors, Cond.InvTauEigenValues,
                            n, Cond.InvTauEigenVectorsInv,
                            Cond.Tau_inv, C_inv, VxCix,
                            Rg, Tau, Cg, Cond.matrixQnPart)


def f(E):
    return high_impedance_p(E + dE, Ec, Temp) * E / (1 - exp(-E / Temp))

#-----------------

def Gamma_Gaussian(dE, Temp, Rt):
    global Ec
    s = np.sqrt(2 * Ec * T)
    lower_cutoff = -1
    high_cuttoff = 0.5

    if dE < lower_cutoff:  # small enough energy, integral is pretty much x time gaussian which analytical
        part1 = s * np.exp(-((Ec + dE) ** 2) / (2 * (s ** 2))) / np.sqrt(2 * np.pi)
        part2 = 0.5 * (Ec + dE) * (math.erf((dE + Ec) / (np.sqrt(2) * s)) - 1)
        return part1 + part2

    if dE > high_cuttoff:
        return 0

    def integrand_gauss(x):
        val = (1 / (e * e * Rt)) * high_impedance_p(x + dE, Ec, Temp) * x / (1 - exp(-x / Temp))
        return val

    mp.dps = 30
    probability = quad(integrand_gauss, [-dE - 0.1, -dE + 0.1])
    mp.dps = 15
    return float(probability)
