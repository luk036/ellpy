# -*- coding: utf-8 -*-
import numpy as np
from .chol_ext import chol_ext


class qmi_oracle:
    """
     Oracle for Quadratic Matrix Inequality
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

        Q = chol_ext(getA, len(self.A))
        if Q.is_spd():
            return (None, None), 1
        v = Q.witness()
        p = len(v)
        Av = v.dot(self.Fx[:p])
        g = -2.*np.array([v.dot(self.F[k][:p]).dot(Av)
                          for k in range(nx)])
        return (g, 1.), 0
