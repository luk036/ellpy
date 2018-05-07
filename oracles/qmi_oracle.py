# -*- coding: utf-8 -*-
import numpy as np
from .chol_ext import chol_ext


class qmi_oracle:
    """
     Oracle for Quadratic Matrix Inequality
        F(x).T * F(x) <= B
     where
        F(x) = F0 - (F1 * x1 + F2 * x2 + ...)
    """

    def __init__(self, F, F0, B):
        self.F = F
        self.F0 = F0
        self.B = B

    def __call__(self, x):
        nx = len(x)
        Fx = self.F0.copy()
        for k in range(nx):
            Fx -= self.F[k] * x[k]
        A = self.B.copy()

        def getA(i, j):
            A[i, j] -= Fx[i, :].dot(Fx[j, :])
            A[j, i] = A[i, j]
            return A[i, j]

        Q = chol_ext(getA, A.shape[0])
        if Q.is_spd():
            return None, None, 1
        v = Q.witness()
        p = len(v)
        # fj = -np.dot(v, A[:p, :p].dot(v))
        fj = 1.
        g = np.zeros(nx)
        Av = Fx[:, :p].dot(v)
        for k in range(nx):
            g[k] = 2. * self.F[k][:, :p].dot(v).dot(Av)
        return g, fj, 0
