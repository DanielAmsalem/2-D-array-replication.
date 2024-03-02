import numpy as np
import math
import matplotlib.pyplot as plt


def sum_np(n, T, p):
    beta = 1 / (2 * T)
    summ = float(0)
    for i in range(n):
        summ += (i ** p) * np.exp(-beta * i ** 2)
    return summ


m = 100
Temp = []
mult = 1
steps = 200
for i in range(steps):
    Temp.append(0.07 - ((0.07-0.001)/steps) * i)

variance = []
for t in Temp:
    variance.append(sum_np(m, t, 2)/sum_np(m, t, 0) - (sum_np(m, t, 1)/sum_np(m, t, 0)) ** 2)

plt.plot(Temp, variance)
plt.xlabel("T [e^2/kB*C]")
plt.ylabel("variance")
plt.title("variance as a function of T")
plt.show()
a = sum_np(m, 0.1, 2)
b = sum_np(m, 0.1, 1)**2
print(a,b)
print(Temp)
print(variance)
