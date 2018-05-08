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
        self.count = -1

    def __call__(self, x):
        self.count = -1
        nx = len(x)
        A = self.B.copy()
        Fx = self.F0.copy()

        def getA(i, j):
            assert i >= j
            if self.count < i:
                self.count = i
                for k in range(nx):
                    Fx[i] -= self.F[k][i] * x[k]
            A[i, j] -= Fx[i].dot(Fx[j])
            # A[j, i] = A[i, j]
            return A[i, j]

        Q = chol_ext(getA, A.shape[0])
        if Q.is_spd():
            return None, None, 1
        v = Q.witness()
        p = len(v)
        # fj = -np.dot(v, A[:p, :p].dot(v))
        fj = 1.
        g = np.zeros(nx)
        # Av = (Fx[:p].T).dot(v)
        Av = v.dot(Fx[:p])
        for k in range(nx):
            g[k] = -2. * v.dot(self.F[k][:p]).dot(Av)
        return g, fj, 0
