import numpy as np
from mpmath import quad, mp, exp, sqrt, ninf, inf
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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
    print(w)
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
        return e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2 - e * Vl
    elif left_right == "right":
        return e * (U(n + in_out, Qg, Vl) + U(n, Qg, Vl)) / 2
    else:
        raise ValueError("left_right must be either 'left' or 'right'")
    # if left_right == "left":
    #     same = -Cg * e / (2 * Cs)
    #     flip = -Cg * (Cl * Vl + n * e) / Cs - e * Vl
    #     return same + in_out * flip
    # elif left_right == "right":
    #     same = -Cg * e / (2 * Cs)
    #     flip = -Cg * (Cl * Vl + n * e) / Cs - e * Vr
    #     return same + in_out * flip
    return


# find n so W is Large
N = 0
V = 4
delE = -0.07
val = 1
delE = W(N, Qn(V, N), V, 1, left_right="left")
print(delE, val)
exit()
w_values = np.concatenate((np.linspace(-1, -0.2, 50, endpoint=False),
                           np.linspace(-0.2, 0.1, 100, endpoint=False),
                           np.linspace(0.1, 1, 50, endpoint=False)))

# 2. Define the list of temperatures you want to plot
T_values = [0.001, 0.01, 0.1]

plt.figure(figsize=(8, 5))

# 3. Loop through each T, calculate Gamma, and plot it
for T in T_values:
    # Calculate gamma for the current T
    gamma_values = [Gamma(w, T, 1) for w in w_values]

    # Plot the curve. The label parameter is what the legend will display!
    plt.plot(w_values, gamma_values, linewidth=2, label=f'T = {T}')

# 4. Format the plot
plt.title("Gamma vs w, Rt=1")  # Fixed a small stray parenthesis here
plt.xlabel("w")
plt.ylabel("Gamma")
plt.grid(True, linestyle='--', alpha=0.7)

# 5. Add the legend to display the labels we set in the loop
plt.legend()

plt.tight_layout()
plt.show()

# # find n so W is Large
# N = 0
# V = 4
# overflow = False
# while not overflow:
#     N += 1
#     new_dE = W(N, Qn(Vl=V,n=N), V, 1, "left")
#     rate = Gamma(new_dE, 0.001, 1)
#     if Gamma(new_dE, 0.001, 1) < 1e-6:
#         overflow = True
#         print(N)