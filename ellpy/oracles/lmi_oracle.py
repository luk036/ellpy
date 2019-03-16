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
        """[summary]

        Arguments:
            F {[type]} -- [description]
            B {[type]} -- [description]
        """
        self.F = F
        self.F0 = B
        self.Q = chol_ext(len(self.F0))

    def __call__(self, x):
        """[summary]

        Arguments:
            x {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        n = len(x)

        def getA(i, j):
            return self.F0[i, j] - sum(
                self.F[k][i, j] * x[k] for k in range(n))

        self.Q.factor(getA)
        if self.Q.is_spd():
            return None, True
        v, ep = self.Q.witness()
        g = np.array([self.Q.sym_quad(v, self.F[i])
                      for i in range(n)])
        return (g, ep), False
