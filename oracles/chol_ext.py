# -*- coding: utf-8 -*-
from __future__ import print_function

from pprint import pprint
import numpy as np
import math


class chol_ext:
    def __init__(self, n):
        self.R = np.zeros((n, n))
        self.p = 0

    def factorize(self, A):
        '''
         If $A$ is positive definite, then $p$ is zero.
         If it is not, then $p$ is a positive integer,
         such that $v = R^{-1} e_p$ is a certificate vector
         to make $v'*A[:p,:p]*v < 0$
        '''
        # n = len(A)
        # self.p = 0
        self.p = 0
        R = self.R
        n = len(R)
        for i in range(n):
            for j in range(i+1):
                d = A[i, j] - np.dot(R[:j, i], R[:j, j])
                if i != j:
                    R[j, i] = 1. / R[j, j] * d
            if d <= 0.:  # strictly positive???
                self.p = i + 1
                R[i, i] = math.sqrt(-d)
                break
            R[i, i] = math.sqrt(d)

    def factor(self, getA):
        '''
         (lazy evalution of A)
         If $A$ is positive definite, then $p$ is zero.
         If it is not, then $p$ is a positive integer,
         such that $v = R^{-1} e_p$ is a certificate vector
         to make $v'*A[:p,:p]*v < 0$
        '''
        # n = len(A)
        # self.p = 0
        self.p = 0
        R = self.R
        n = len(R)
        for i in range(n):
            for j in range(i+1):
                d = getA(i, j) - np.dot(R[:j, i], R[:j, j])
                if i != j:
                    R[j, i] = 1. / R[j, j] * d
            if d <= 0.:  # strictly positive???
                self.p = i + 1
                R[i, i] = math.sqrt(-d)
                break
            R[i, i] = math.sqrt(d)

    def is_spd(self):
        return self.p == 0

    def witness(self):
        assert not self.is_spd()
        p = self.p
        v = np.zeros(p)
        v[p-1] = 1. / self.R[p-1, p-1]
        for i in range(p - 2, -1, -1):
            s = np.dot(self.R[i, i+1:p], v[i+1:p])
            v[i] = -s / self.R[i, i]
        return v

    def sym_quad(self, v, F):
        # v = self.witness()
        p = self.p
        return v.dot(F[:p, :p].dot(v))


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
