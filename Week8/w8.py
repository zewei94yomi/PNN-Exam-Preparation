import numpy as np
from scipy.linalg import solve

a = np.array([[26, 24, -15, 1], [24, 26, -15, 1], [15, 15, -9, 1], [1, 1, -1, 0]])
b = np.array([1, 1, -1, 0])
x = solve(a, b)
print(x)
