import Conditions
import numpy as np

matrix = np.zeros((2, 3))
for i in range(2):
    matrix[i] = np.array([1,3,i+4])

sum = np.zeros(3)
for j in matrix:
    print(sum)
    sum += j

print(sum)