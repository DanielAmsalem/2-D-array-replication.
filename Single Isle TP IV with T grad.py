import numpy as np
from mpmath import quad, mp, exp, sqrt
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.linalg import eig
import os
import sys
import re

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
    if abs(in_out) != 1:
        raise ValueError

    if left_right == "left":
        return in_out * e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2 - in_out * e * Vl
    elif left_right == "right":
        return in_out * e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2
    else:
        raise ValueError("left_right must be either 'left' or 'right'")


# --- Updated function signature to accept T_l and T_r ---
def calculate_current(Vl, N, T_l, T_r, Tdot):
    G_L_plus = np.zeros(N + 1)
    G_R_plus = np.zeros(N + 1)
    G_L_minus = np.zeros(N + 1)
    G_R_minus = np.zeros(N + 1)

    for n in range(N + 1):
        # Pass T_l to the left junction rates
        G_L_plus[n] = Gamma(W(n, Qn(Vl, n), Vl, 1, "left"), T_l, Rl)
        G_L_minus[n] = Gamma(W(n, Qn(Vl, n), Vl, -1, "left"), Tdot, Rl)

        # Pass T_r to the right junction rates
        G_R_plus[n] = Gamma(W(n, Qn(Vl, n), Vl, 1, "right"), T_r, Rr)
        G_R_minus[n] = Gamma(W(n, Qn(Vl, n), Vl, -1, "right"), Tdot, Rr)

        if not n%10:
            print(n)

    # Total rates for the master equation
    G_plus = G_L_plus + G_R_plus
    G_minus = G_L_minus + G_R_minus

    # Enforce boundary conditions
    G_plus[-1] = 0.0
    G_minus[0] = 0.0

    # Construct the matrix
    diag_main = -(G_plus + G_minus)
    diag_sub = G_plus[:-1]
    diag_super = G_minus[1:]

    M = np.diag(diag_main) + np.diag(diag_sub, k=-1) + np.diag(diag_super, k=1)

    # Diagonalize
    eigenvalues, eigenvectors = eig(M)

    # Extract steady state and calculate current
    zero_idx = np.argmin(np.abs(eigenvalues))
    p_stat = np.real(eigenvectors[:, zero_idx])
    p_stat = p_stat / np.sum(p_stat)

    current = e * np.sum(p_stat * (G_L_plus - G_L_minus))
    return current


script_name = os.path.basename(sys.argv[0])
# Searches for the first contiguous block of digits in the filename
match = re.search(r'\d+', script_name)
multiplier = int(match.group()) if match else 0

#temperature gradient
T0 = 0.001
T_left = T0
T_dot = 0.5
T_right = 1

# Define voltage sweep range
N_states = 120
num_points = 21
V_vals = np.linspace(0, 4, num_points)
currents = []

print(f"Starting voltage sweep with T_l={T_left/T0}*T0 ; Tdot={T_dot/T0}*T; T_r={T_right/T0}*T")
for V in V_vals:
    print(f"Calculating for Vl = {V:.2f} V...")
    I = calculate_current(V, N=N_states, T_l=T_left, T_r=T_right, Tdot=T_dot)
    currents.append(I)

# --- Plot the Results ---
plt.figure(figsize=(8, 6))
plt.plot(V_vals, currents, linestyle='-', color='r')
plt.title(f"I-V Characteristic with $\Delta T$ ($T_l={T_left/T0}*T$, $T_r={T_right/T0}*T$)", fontsize=14)
plt.xlabel("Left Voltage $V_l$ (V)", fontsize=12)
plt.ylabel("Steady-State Current I L->R", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()