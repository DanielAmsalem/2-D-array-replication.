import numpy as np
from scipy.integrate import quad
from scipy.special import erf

'''
COMPARE CORRELATED AND UNCORRALATED ERROR CALC
'''


def calc_expected_dist_std(T_array):
    # T_array should be an N*N long list with entries [T0/T0...T0/T0,(T0+dT)/T0...(T0+dt/T0),...(T0+(N-1)dT)/T0]
    # Standard deviation of each site
    sigma = 0.01 * np.sqrt(np.asarray(T_array))
    sqrt2_sigma = np.sqrt(2) * sigma

    # CDF of the maximum absolute deviation
    def F_Z(z):
        # np.prod over the array of error functions
        return np.prod(erf(z / sqrt2_sigma))

    # Integrands for the 1st and 2nd moments
    def integrand_1st_moment(z):
        return 1.0 - F_Z(z)

    def integrand_2nd_moment(z):
        return 2.0 * z * (1.0 - F_Z(z))

    # Determine a safe upper bound for integration
    # The max absolute value is exceedingly unlikely to exceed 10 sigma of the hottest site
    upper_limit = np.max(sigma) * 10.0

    # Calculate moments using quadrature
    # epsrel is tightened slightly for high-precision stability
    moment_1, err_1 = quad(integrand_1st_moment, 0, upper_limit, epsabs=1e-10, epsrel=1e-10)
    moment_2, err_2 = quad(integrand_2nd_moment, 0, upper_limit, epsabs=1e-10, epsrel=1e-10)

    # Variance and Standard Deviation
    variance_Z = moment_2 - (moment_1 ** 2)
    std_Z = np.sqrt(variance_Z)

    return std_Z, moment_1  # Returns (Expected Standard Deviation, Expected Mean)


# Example usage for N=51 points with some temperature profile
N = 7
Tmax = 0.001 * (N - 1) * 19 / 20 + 0.001
T_profile = np.repeat(np.linspace(0.001 / 0.001, Tmax / 0.001, N), N)
T_CHAIN_profile = np.linspace(0.001 / 0.001, 0.0067 / 0.001, N)
expected_std, expected_mean = calc_expected_dist_std(T_profile)
print(f"T GRADIENT 19/20 ; Tmax {Tmax:.10f}")
print(f"Expected Mean of dist: {expected_mean:.10f}")
print(f"with Covariance : {np.sum(0.01 * np.sqrt(np.array(T_CHAIN_profile)))}")
print("")

Tmax = 0.001 * (N - 1) * 80 / 20 + 0.001
T_profile = np.repeat(np.linspace(0.001 / 0.001, Tmax / 0.001, N), N)
T_CHAIN_profile = np.linspace(0.001 / 0.001, Tmax / 0.001, N)
expected_std, expected_mean = calc_expected_dist_std(T_profile)
print(f"T GRADIENT 80/20 ; Tmax {Tmax:.10f}")
print(f"Expected Mean of dist: {expected_mean:.10f}")
print(f"with Covariance : {np.sum(0.01 * np.sqrt(np.array(T_CHAIN_profile)))}")
print("")

Tmax = 0.001 * (N - 1) * 500 / 20 + 0.001
T_profile = np.repeat(np.linspace(0.001 / 0.001, Tmax / 0.001, N), N)
T_CHAIN_profile = np.linspace(0.001 / 0.001, Tmax / 0.001, N)
expected_std, expected_mean = calc_expected_dist_std(T_profile)
print(f"T GRADIENT 500/20 ; Tmax {Tmax:.10f}")
print(f"Expected Mean of dist: {expected_mean:.10f}")
print(f"with Covariance : {np.sum(0.01 * np.sqrt(np.array(T_CHAIN_profile)))}")
