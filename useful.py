import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import random


def random_sum_to(n, num_terms=None):  # return fixed sum randomly distributed
    num_terms = (num_terms or random.randint(2, n)) - 1
    a = random.sample(range(1, n), num_terms) + [0, n]
    list.sort(a)
    return [a[j + 1] - a[j] for j in range(len(a) - 1)]


Vleft = np.array([1, 2, 3, 4, 5, 6])
Volts = 1
I_vec_avg = np.array([1, 2, 3, 4, 5, 6]) * 6
Amp = 5
increase = True
I_V = plt.plot(Vleft / Volts, I_vec_avg / Amp)
plt.xlabel("Voltage")
plt.ylabel("Current")
if increase:
    plt.title("increasing voltage")
else:
    plt.title("decreasing voltage")
plt.show()
