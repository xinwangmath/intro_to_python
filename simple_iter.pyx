# author: Xin Wang
# email: xwang320@gatech.edu
# A library for simple iterative solvers with dense A.
from __future__ import print_function
from __future__ import division

import numpy as np
cimport numpy as c_np
cimport cython

def jacobi(c_np.ndarray[double, ndim=2] A, 
           c_np.ndarray[double, ndim=1] b, 
           c_np.ndarray[double, ndim=1] x_0, 
           double tol=1e-3, 
           int maxSteps=100):
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
    cdef int m, n, step = 0
    m = A.shape[0]
    n = A.shape[1]

    cdef double residual_magnitude
    residual_magnitude = np.linalg.norm(b - np.dot(A, x_0))

    cdef c_np.ndarray[double, ndim=1] x
    cdef c_np.ndarray[double, ndim=1] x_temp

    x = np.copy(x_0)
    x_temp = np.copy(x)

    while residual_magnitude > tol and step < maxSteps:
        x_temp = b - np.dot((A - np.diag(np.diag(A))), x)
        x_temp /= np.diag(A)
        x = x_temp
        residual_magnitude = np.linalg.norm(b - np.dot(A, x))
        step += 1

    return x

def gaussSeidel(c_np.ndarray[double, ndim=2] A, 
                c_np.ndarray[double, ndim=1] b, 
                c_np.ndarray[double, ndim=1] x_0, 
                double tol=1e-3, 
                int maxSteps=100):
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
    cdef int m, n, step = 0
    m = A.shape[0]
    n = A.shape[1]

    cdef double residual_magnitude
    residual_magnitude = np.linalg.norm(b - np.dot(A, x_0))

    cdef c_np.ndarray[double, ndim=1] x
    x = np.copy(x_0)

    cdef double temp
    while residual_magnitude > tol and step < maxSteps:
        for i in range(n):
            temp = b[i] - np.dot(A[i, :], x) + A[i, i] * x[i]
            x[i] = temp/A[i, i]
        residual_magnitude = np.linalg.norm(b - np.dot(A, x))
        step += 1
    
    return x


def sor(c_np.ndarray[double, ndim=2] A, 
        c_np.ndarray[double, ndim=1] b, 
        c_np.ndarray[double, ndim=1] x_0, 
        double w=1, 
        double tol=1e-3, 
        double maxSteps=100):
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
    cdef int m, n, step = 0
    m = A.shape[0]
    n = A.shape[1]

    cdef double residual_magnitude
    residual_magnitude = np.linalg.norm(b - np.dot(A, x_0))

    cdef c_np.ndarray[double, ndim=1] x
    x = np.copy(x_0)

    cdef double temp

    while residual_magnitude > tol and step < maxSteps:
        for i in range(n):
            temp = b[i] - np.dot(A[i, :], x) + A[i, i] * x[i]
            x[i] *= (1 - w)
            x[i] += w * (temp/A[i, i])
        residual_magnitude = np.linalg.norm(b - np.dot(A, x))
        step += 1
    
    return x