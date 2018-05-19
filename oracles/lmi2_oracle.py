# -*- coding: utf-8 -*-
import numpy as np
from .chol_ext import chol_ext


class lmi2_oracle:
    """
        Oracle for Linear Matrix Inequality constraint
            0 <= F * x <= U
    """

    def __init__(self, F, U):
        self.F = F
        self.U = U
        self.A = np.zeros(U.shape)
        self.S = np.zeros(U.shape)
        self.Q = chol_ext(len(U))

    def __call__(self, x):
        #A = self.U.copy()
        #S = np.zeros(A.shape)
        n = len(x)

        def getS(i, j):
            self.S[i, j] = sum(self.F[k][i, j] * x[k] for k in range(n))
            return self.S[i, j]

        def getA(i, j):
            # for k in range(n):
            #     S[i, j] = self.F[k][i, j] * x[k]
            self.A[i, j] = self.U[i, j]
            self.A[i, j] -= getS(i, j)
            return self.A[i, j]

        self.Q.factor(getA)
        if not self.Q.is_spd():
            v = self.Q.witness()
            p = len(v)
            #fj = np.dot(v, S[:p, :p].dot(v))
            g = np.array([v.dot(self.F[i][:p, :p].dot(v)) for i in range(n)])
            return (g, (1., 0.)), 0

        self.Q.factor(getS)
        if not self.Q.is_spd():
            v = self.Q.witness()
            p = len(v)
            fj = -np.dot(v, self.U[:p, :p].dot(v))
            g = np.array([-v.dot(self.F[i][:p, :p].dot(v)) for i in range(n)])
            return (g, (1., fj)), 0

        return None, 1
