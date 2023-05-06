import Conditions
import numpy as np

n = np.array([0,1,2,3,4,5,6])

n_tag = np.array(list(n))
n_tag[3] += 1
print(n, n_tag)

