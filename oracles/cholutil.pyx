# -*- coding: utf-8 -*-
from __future__ import print_function

from pprint import pprint
import numpy as np
cimport numpy as np
cimport cython
from cpython cimport array
import array

"""
>>> sin(0)
0.0
"""

cdef extern from "math.h":
    cpdef double sqrt(double x)

DTYPE = np.float
ctypedef np.float_t DTYPE_t

class cholutil:
    def __init__(self, int n):
        self.R = np.zeros((n, n))

    @cython.boundscheck(False) # turn off bounds-checking
    @cython.wraparound(False)  # turn off negative index wrapping
    def factor(self, getA):
        '''
         If $A$ is positive definite, then $p$ is zero.
         If it is not, then $p$ is a positive integer,
         such that $v = R^{-1} e_p$ is a certificate vector
         to make $v'*A[:p,:p]*v < 0$
        '''
        # cdef np.ndarray[dtype=DTYPE_t, ndim=2] R = np.zeros((n, n))
        # self.R = np.zeros((n, n))
        cdef int n = len(self.R)
        cdef DTYPE_t[:, ::1] R = self.R
        cdef int p = 0
        cdef int i, j, k
        cdef DTYPE_t d
        
        for i in range(n):
            for j in range(i+1):
                d = getA(i, j)
                for k in range(j):
                    d -= R[k, i] * R[k, j]
                if i != j:
                    R[j, i] = 1. / R[j, j] * d
            if d <= 0.:  # strictly positive???
                p = i + 1
                R[i, i] = sqrt(-d)
                break
            R[i, i] = sqrt(d)

        # self.R = R
        self.p = p

    def is_spd(self):
        return self.p == 0

    @cython.boundscheck(False) # turn off bounds-checking
    @cython.wraparound(False)  # turn off negative index wrapping
    def witness(self):
        assert self.p > 0

        cdef int i, k
        cdef DTYPE_t s
        cdef int p = self.p
        cdef res = np.zeros(p)
        # cdef np.ndarray[dtype=DTYPE_t, ndim=1] v = np.zeros(p)
        # cdef np.ndarray[dtype=DTYPE_t, ndim=2] R = self.R
        cdef DTYPE_t[::1] v = res
        cdef DTYPE_t[:, ::1] R = self.R

        v[p-1] = 1. / R[p-1, p-1]
        for i in range(p - 2, -1, -1):
            s = 0.
            for k in range(i+1, p):
                s += R[i,k] * v[k]
            # s = np.dot(R[i, i+1:p], v[i+1:p])
            v[i] = -s / R[i, i]
        return res


# def print_case(l1):
#     m1 = np.array(l1)
#     R, p = chol_ext(m1)
#     pprint(R)
#     if p > 0:
#         v = witness(R, p)
#         print(np.dot(v, m1[:p, :p].dot(v)))


# if __name__ == "__main__":
#     l1 = [[25., 15., -5.],
#           [15., 18., 0.],
#           [-5., 0., 11.]]
#     print_case(l1)

#     l2 = [[18., 22., 54., 42.],
#           [22., -70., 86., 62.],
#           [54., 86., -174., 134.],
#           [42., 62., 134., -106.]]
#     print_case(l2)
