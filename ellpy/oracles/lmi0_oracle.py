# -*- coding: utf-8 -*-
import numpy as np
from .chol_ext import chol_ext


class lmi0_oracle:
    """
        Oracle for Linear Matrix Inequality constraint
            F * x >= 0
    """

    def __init__(self, F):
        self.F = F
        self.A = np.zeros(F[0].shape)
        self.Q = chol_ext(len(self.A))

    def __call__(self, x):
        n = len(x)

        def getA(i, j):
            self.A[i, j] = sum(self.F[k][i, j] * x[k]
                               for k in range(n))
            return self.A[i, j]

        self.Q.factor(getA)
        if self.Q.is_spd():
            return None, True
        v = self.Q.witness()
#        p = len(v)
#        g = np.array([-v.dot(self.F[i][:p, :p].dot(v))
        g = np.array([-self.Q.sym_quad(v, self.F[i])
                      for i in range(n)])
        return (g, 1.), False
