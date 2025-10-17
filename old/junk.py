def high_impedance_p(x, mu, Temp):
    """
    P- function for high impedance.
    :param x: function input (energy) == E+dE.
    :param mu: Electrostatic energy of environment == Ec.
    :param Temp: Temperature.
    :return: P(x)
    """
    sigma_squared = 2 * mu * Temp
    mu = -mu
    return exp(-((x - mu) ** 2) / (2 * sigma_squared)) / sqrt(2 * np.pi * sigma_squared)


def integrand_gauss(x, temperature, value, Ec):
    if x == 0:
        return temperature
    result = (
        high_impedance_p(x + value, Ec, temperature) * x / (1 - exp(-x / temperature))
    )
    return result


def make_integrand(temp1, val1, Ec1):
    def f1(x):
        return integrand_gauss(x, temp1, val1, Ec=Ec1)

    return f1
