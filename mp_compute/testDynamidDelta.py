import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # for cluster Agg, pc TkAgg
import matplotlib.pyplot as plt
import warnings
from Functions import exact_bcs_gap, calc_expected_dist_std


# Suppress harmless exp overflow warnings during low-T integration
warnings.filterwarnings('ignore', category=RuntimeWarning)


if __name__ == "__main__":
    # Parameters
    Delta_0 = 0.05*0.2
    Tc = Delta_0 / 1.764
    print(f"Calculated Critical Temperature (Tc): {Tc:.5f}")

    # Generate temperature array from 0.001 to 0.06 (crossing Tc)
    T_array = np.linspace(0.001, 0.001, 7)

    print("Solving exact BCS self-consistency equation. This may take a few seconds...")
    Delta_T = exact_bcs_gap(T_array, Delta_0)
    print(Delta_T)
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(T_array, Delta_T, '-', color='b', linewidth=2.5, label=r'Exact $\Delta(T)$')

    # Mark the theoretical critical temperature
    plt.axvline(x=Tc, color='r', linestyle='--', label=f'$T_c \\approx$ {Tc:.4f}')
    plt.axhline(y=0, color='k', linewidth=0.8)

    # Formatting
    plt.title(r'Exact BCS Superconducting Gap $\Delta(T)$ vs Temperature', fontsize=14)
    plt.xlabel('Temperature $T$', fontsize=12)
    plt.ylabel(r'Gap $\Delta(T)$', fontsize=12)
    plt.xlim(0, 0.031)
    plt.ylim(-0.005, 0.105)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    # Display the plot
    plt.show()