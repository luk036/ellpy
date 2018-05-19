# -*- coding: utf-8 -*-
import numpy as np
from .chol_ext import chol_ext
# import cholutil


class qmi_oracle:
    """Oracle for Quadratic Matrix Inequality
        F(x).T * F(x) <= I*t
     where
        F(x) = F0 - (F1 * x1 + F2 * x2 + ...)
    """

    def __init__(self, F, F0):
        self.F = F
        self.F0 = F0
        #self.B = None
        self.Fx = np.zeros(F0.shape)
        # A = self.B.copy()
        self.A = np.zeros(F0.shape)
        self.t = None
        self.count = -1
        self.Q = chol_ext(len(F0))

    def update(self, t):
        self.t = t

    def __call__(self, x):
        self.count = -1
        nx = len(x)
        # Fx = self.F0.copy()
        # A = np.zeros(self.F0.shape)

        def getA(i, j):
            assert i >= j
            if self.count < i:
                self.count = i
                self.Fx[i] = self.F0[i]
                self.Fx[i] -= sum(self.F[k][i] * x[k]
                                  for k in range(nx))
            self.A[i, j] = -self.Fx[i].dot(self.Fx[j])
            if i == j:
                self.A[i, j] += self.t
            return self.A[i, j]

        self.Q.factor(getA)

        if self.Q.is_spd():
            return None, 1
        v = self.Q.witness()
        p = len(v)
        Av = v.dot(self.Fx[:p])
        g = -2.*np.array([v.dot(self.F[k][:p]).dot(Av)
                          for k in range(nx)])
        return (g, 1.), False
