import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # for cluster Agg, pc TkAgg
import matplotlib.pyplot as plt
import warnings
from Functions import exact_bcs_gap, calc_expected_dist_std


warnings.filterwarnings('ignore', category=RuntimeWarning)

if __name__ == "__main__":
    # Parameters
    E_c=0.025
    Delta_0 = E_c*2
    Tc = Delta_0 / 1.764
    print(f"Calculated Critical Temperature (Tc): {Tc:.5f}")

    # Generate temperature array from 0.001 to 0.06 (crossing Tc)
    T_array = np.linspace(0.001, 0.06, 100)

    print("Solving exact BCS self-consistency equation. This may take a few seconds...")
    Delta_T = exact_bcs_gap(T_array, Delta_0)/E_c
    print(Delta_T)

    T_array = T_array/E_c

    plt.figure(figsize=(8, 5))
    plt.plot(T_array, Delta_T, '-', color='b', linewidth=2.5, label=r'$\Delta(T)/E_c$')

    # critical temperature
    plt.axvline(x=Tc/E_c, color='r', linestyle='--', label=f'$T_c \\approx$ {Tc/E_c:.4f}$E_c$')
    plt.axhline(y=0, color='k', linewidth=0.8)

    # Formatting
    plt.title(r'BCS Gap $\Delta(T=0)=2E_c$ Change with $T$', fontsize=25)
    plt.xlabel('$T/E_c$', fontsize=22)
    plt.ylabel(r'Gap $\frac{\Delta(T)}{E_c}$', fontsize=22)
    plt.xlim(0, 0.063/E_c)
    plt.ylim(-0.005, 0.11/E_c)
    plt.legend(fontsize=20)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
