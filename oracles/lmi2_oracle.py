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

    def __call__(self, x):
        A = self.U.copy()
        S = np.zeros(A.shape)
        n = len(x)

        def getA(i, j):
            # for k in range(n):
            #     S[i, j] = self.F[k][i, j] * x[k]
            S[i, j] = sum(self.F[k][i, j] * x[k] for k in range(n))
            A[i, j] -= S[i, j]
            return A[i, j]

        Q = chol_ext(getA, A.shape[0])
        if Q.is_spd():
            return (None, None), 1
        v = Q.witness()
        p = len(v)
        # fj = -np.dot(v, A[:p, :p].dot(v))
        # fj = 1.
        # g = np.zeros(n)
        # for i in range(n):
        #     g[i] = v.dot(self.F[i][:p, :p].dot(v))
        g = np.array([v.dot(self.F[i][:p, :p].dot(v)) for i in range(n)])
        return (g, 1.), 0
