# author: Xin Wang
# email: xwang320@gatech.edu
# A library for simple iterative solvers with dense A.
from __future__ import print_function
from __future__ import division

import numpy as np

def jacobi(A, b, x_0, tol=1e-3, maxSteps=100):
    """
    Jacobi iteration solver.

    @args
    A: coefficient matrix, numpy ndarray of two dimensions
    b: right hand side vector, numpy ndarray of one dimension
    x_0: initial value vector, numpy ndarray of one dimension
    tol: tolerance, iteration stops when the residual is smaller than tol, float
    maxSteps: maximum number of steps, int

    @return
    solution at the end of the iteration, numpy ndarray of one dimension
    """
    m, n = A.shape
    step = 0
    residual_magnitude = np.linalg.norm(b - np.dot(A, x_0))

    x = np.copy(x_0)
    x_temp = np.copy(x)

    while residual_magnitude > tol and step < maxSteps:
        x_temp = b - np.dot((A - np.diag(np.diag(A))), x)
        x_temp /= np.diag(A)
        x = x_temp
        residual_magnitude = np.linalg.norm(b - np.dot(A, x))
        step += 1

    return x

def gaussSeidel(A, b, x_0, tol=1e-3, maxSteps=100):
    """
    Gauss-Seidel iteration solver.

    @args
    A: coefficient matrix, numpy ndarray of two dimensions
    b: right hand side vector, numpy ndarray of one dimension
    x_0: initial value vector, numpy ndarray of one dimension
    tol: tolerance, iteration stops when the residual is smaller than tol, float
    maxSteps: maximum number of steps, int

    @return
    solution at the end of the iteration, numpy ndarray of one dimension
    """
    m, n = A.shape
    step = 0
    residual_magnitude = np.linalg.norm(b - np.dot(A, x_0))

    x = np.copy(x_0)

    while residual_magnitude > tol and step < maxSteps:
        for i in range(n):
            temp = b[i] - np.dot(A[i, :], x) + A[i, i] * x[i]
            x[i] = temp/A[i, i]
        residual_magnitude = np.linalg.norm(b - np.dot(A, x))
        step += 1
    
    return x


def sor(A, b, x_0, w=1, tol=1e-3, maxSteps=100):
    """
    Succesive Over Relaxation iteration solver.

    @args
    A: coefficient matrix, numpy ndarray of two dimensions
    b: right hand side vector, numpy ndarray of one dimension
    x_0: initial value vector, numpy ndarray of one dimension
    w: over relaxation coefficient, 1 <= w < 2, float
    tol: tolerance, iteration stops when the residual is smaller than tol, float
    maxSteps: maximum number of steps, int

    @return
    solution at the end of the iteration, numpy ndarray of one dimension
    """
    m, n = A.shape
    step = 0
    residual_magnitude = np.linalg.norm(b - np.dot(A, x_0))

    x = np.copy(x_0)

    while residual_magnitude > tol and step < maxSteps:
        for i in range(n):
            temp = b[i] - np.dot(A[i, :], x) + A[i, i] * x[i]
            x[i] *= (1 - w)
            x[i] += w * (temp/A[i, i])
        residual_magnitude = np.linalg.norm(b - np.dot(A, x))
        step += 1
    
    return x