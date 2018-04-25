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

    def chk_mtx(self, A, x):
        n = len(x)

        def getA(i, j):
            for k in range(n):
                A[i,j] -= self.F[k][i,j] * x[k]
            return A[i,j]

        Q = chol_ext(getA, A.shape[0])
        if Q.is_spd():
            return None, None, 1
        v = Q.witness()
        p = len(v)
        # fj = -np.dot(v, A[:p, :p].dot(v))
        fj = 1.
        g = np.zeros(n)
        for i in range(n):
            g[i] = np.dot(v, self.F[i][:p, :p].dot(v))
        return g, fj, 0

    # def chk_spd_t(self, x, t):
    #     A = np.array(self.F0)
    #     # ???
    #     # m = len(A)
    #     # A(range(m), range(m)) += t
    #     A += t
    #     return self.chk_mtx(A, x)

    def __call__(self, x):
        A = self.F0.copy()
        return self.chk_mtx(A, x)


    # def __call__(self, x, t):
    #     g, fj, _, _ = self.chk_spd_t(x, t)
    #     return g, fj
