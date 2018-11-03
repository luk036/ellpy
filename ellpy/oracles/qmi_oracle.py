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
        # self.A = np.zeros(F0.shape)
        self.t = None
        self.count = 0
        self.Q = chol_ext(len(F0))

    def update(self, t):
        self.t = t

    def __call__(self, x):
        self.count = 0
        nx = len(x)
        # Fx = self.F0.copy()
        # A = np.zeros(self.F0.shape)

        def getA(i, j):
            if i < j:
                raise AssertionError()
            if self.count < i + 1:
                self.count = i + 1 
                self.Fx[i] = self.F0[i]
                self.Fx[i] -= sum(self.F[k][i] * x[k]
                                  for k in range(nx))
            a = -self.Fx[i].dot(self.Fx[j])
            if i == j:
                a += self.t
            return a

        self.Q.factor(getA)

        if self.Q.is_spd():
            return None, True
        v, f = self.Q.witness()
        p = len(v)
        Av = v.dot(self.Fx[:p])
        g = np.array([-2*v.dot(self.F[k][:p]).dot(Av)
                          for k in range(nx)])
        return (g, f), False
