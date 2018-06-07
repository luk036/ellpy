# -*- coding: utf-8 -*-
import numpy as np
from .chol_ext import chol_ext


class lmi_oracle:
    """Oracle for Linear Matrix Inequality constraint
            F * x <= B
        Or
            (B - F * x) must be a semidefinte matrix
    """

    def __init__(self, F, B):
        self.F = F
        self.F0 = B
        self.A = np.zeros(B.shape)
        self.Q = chol_ext(len(B))

    def __call__(self, x):
        # A = self.F0.copy()
        n = len(x)

        def getA(i, j):
            self.A[i, j] = self.F0[i, j]
            self.A[i, j] -= sum(self.F[k][i, j] * x[k]
                                for k in range(n))
            return self.A[i, j]

        self.Q.factor(getA)
        if self.Q.is_spd():
            return None, 1
        v = self.Q.witness()
        # p = len(v)
        # g = np.array([v.dot(self.F[i][:p, :p].dot(v))
        #               for i in range(n)])
        g = np.array([self.Q.sym_quad(v, self.F[i])
                      for i in range(n)])
        return (g, 1.), False