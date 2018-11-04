# -*- coding: utf-8 -*-
import numpy as np
from .chol_ext import chol_ext
# from .cholutil import cholutil


class lmi_old_oracle:
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
        n = len(x)
        self.A = self.F0.copy()
        self.A -= sum(self.F[k] * x[k] for k in range(n))
        self.Q.factorize(self.A)
        if self.Q.is_spd():
            return None, True
        v, ep = self.Q.witness()
        g = np.array([self.Q.sym_quad(v, self.F[i])
                      for i in range(n)])
        return (g, ep), False
