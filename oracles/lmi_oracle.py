# -*- coding: utf-8 -*-
import numpy as np
from .chol_ext import chol_ext


class lmi_oracle:

    """
        Oracle for Linear Matrix Inequality constraint
            F * x <= B
        Or
            (B - F * x) must be a semidefinte matrix
    """

    def __init__(self, F, B):
        self.F = F
        self.F0 = B

    def __call__(self, x):
        A = self.F0.copy()
        n = len(x)

        def getA(i, j):
            # for k in range(n):
            #     A[i, j] -= self.F[k][i, j] * x[k]
            A[i, j] -= sum(self.F[k][i, j] * x[k] for k in range(n))
            return A[i, j]

        Q = chol_ext(getA, len(A))
        if Q.is_spd():
            return (None, None), 1
        v = Q.witness()
        p = len(v)
        # fj = -np.dot(v, A[:p, :p].dot(v))
        # g = np.zeros(n)
        # for i in range(n):
        #     g[i] = v.dot(self.F[i][:p, :p].dot(v))
        g = np.array([v.dot(self.F[i][:p, :p].dot(v)) for i in range(n)])
        return (g, 1.), 0
