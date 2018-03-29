from __future__ import print_function
from __future__ import division
from six.moves import range

import numpy as np

# Write Cython codes in simple_iter.pyx.
# Use Cython to compile the library, see setup_simple_iter.py.
# Import the library just like it is a Python library.
import simple_iter

n = 100
A = np.eye(n+1)
for i in range(1, n):
    A[i, i-1] = -1
    A[i, i+1] = -1
    A[i, i] = 2 + 4.0 * (1/n)**2
b = np.zeros((n+1,))
b[0] = -1
b[-1] = 2

x_0 = np.copy(b)
x_direct = np.linalg.solve(A, b)
x_jacobi = simple_iter.jacobi(A, b, x_0, tol=1e-12, maxSteps=300)
x_gs = simple_iter.gaussSeidel(A, b, x_0, tol=1e-12, maxSteps=300)
x_sor = simple_iter.sor(A, b, x_0, w=1.9375, tol=1e-12, maxSteps=300)