import numpy as np
from mpmath import quad, mp, exp, sqrt, ninf, inf
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.linalg import eig

e = 1
Vr = 0
Cl = 2
Cr = 1
Rl = 1
Rr = 10
Rg = 1000 * (Cl + Cr)
Cg = 10 * (Cl + Cr)
Cs = Cg + Cl + Cr


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


def Gamma(w, T, Rt, mu=0.5 / Cg):
    # we set mu =0 since 2/cg is included in W.
    absw = abs(w)
    mp.dps = 40
    probability = quad(integrand(T, w, mu), [-6, -absw - 0.2, 0, absw + 0.1, 7])
    mp.dps = 15
    return probability / (Rt * e * e)


def U(n, Qg, Vl):
    return (Qg + n * e + Cl * Vl) / (Cl + Cr)


def Qn(Vl, n):
    return -Cg * (Cl * Vl + n * e) / Cs


def W(n, Qg, Vl, in_out, left_right):
    # in is 1 : n->n+1
    # out is -1 : n-> n-1
    if abs(in_out) != 1:
        raise ValueError

    if left_right == "left":
        return in_out * e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2 - in_out * e * Vl
    elif left_right == "right":
        return in_out * e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2
    else:
        raise ValueError("left_right must be either 'left' or 'right'")


def Gamma_n(n, Qg, Vl, in_out, T, Rt):
    # compute w
    dE_left = W(n, Qg, Vl, in_out, "left")
    dE_right = W(n, Qg, Vl, in_out, "right")
    return Gamma(dE_left, T, Rt) + Gamma(dE_right, T, Rt)


N = 120
T = 0.001
V = 4

G_L_plus = np.zeros(N + 1)
G_R_plus = np.zeros(N + 1)
G_L_minus = np.zeros(N + 1)
G_R_minus = np.zeros(N + 1)

for n in np.arange(N+1):
    # n -> n+1 (Entering the dot)
    print(n)
    G_L_plus[n] = Gamma(W(n, Qn(V,n), V, 1, "left"), T, Rl)
    G_R_plus[n] = Gamma(W(n, Qn(V,n), V, 1, "right"), T*10, Rr)

    # n -> n-1 (Leaving the dot)
    G_L_minus[n] = Gamma(W(n, Qn(V,n), V, -1, "left"), T*5, Rl)
    G_R_minus[n] = Gamma(W(n, Qn(V,n), V, -1, "right"), T*5, Rr)

# Total rates for the master equation
G_plus = G_L_plus + G_R_plus
G_minus = G_L_minus + G_R_minus

# Enforce boundary conditions
G_plus[-1] = 0.0
G_minus[0] = 0.0

# 3. Construct the matrix
diag_main = -(G_plus + G_minus)
diag_sub = G_plus[:-1]
diag_super = G_minus[1:]

M = np.diag(diag_main) + np.diag(diag_sub, k=-1) + np.diag(diag_super, k=1)

# 4. Diagonalize
eigenvalues, eigenvectors = eig(M)

# 5. Extract steady state and calculate current
# The steady state is the eigenvector associated with the eigenvalue closest to 0
zero_idx = np.argmin(np.abs(eigenvalues))
p_stat = np.real(eigenvectors[:, zero_idx])

# Normalize the probability distribution so it sums to 1
p_stat = p_stat / np.sum(p_stat)

# Calculate Current across the Left junction
current = e * np.sum(p_stat * (G_L_plus - G_L_minus))

print(f"Steady-state current: {current}")

