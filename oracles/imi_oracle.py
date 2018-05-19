# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from .chol_ext import chol_ext
from lmi0_oracle import lmi0_oracle


class imi_oracle:
    """Oracle for Linear Matrix Inequality constraint
            Sig * x[:n] >= 0
            S * x[n:] >= 0
            [Sig, I
              I,  S] >= 0
    """

    def __init__(self, F):
        self.F = F
        n = len(F[0])*2
        self.A = np.zeros((n, n))
        self.Q = chol_ext(n)

    def __call__(self, x):
        n = len(x) // 2
        m = len(self.A) // 2

        def getA(i, j):
            if i < m:
                self.A[i, j] = sum(self.F[k][i, j] * x[k]
                                   for k in range(n))
            elif j < m:
                self.A[i, j] = 1. if i-m == j else 0.
            else:
                i2 = i-m
                j2 = j-m
                self.A[i, j] = sum(self.F[k][i2, j2] * x[k+n]
                                   for k in range(n))
            self.A[j, i] = self.A[i, j]  # for later use
            return self.A[i, j]

        self.Q.factor(getA)
        if self.Q.is_spd():
            return None, 1
        v = self.Q.witness()
        p = len(v)
        if p < m:
            g1 = np.array([-v.dot(self.F[i][:p, :p].dot(v))
                           for i in range(n)])
            g = np.concatenate((g1, np.zeros(n)))
        else:
            v1 = v[:m]
            v2 = v[m:]
            g1 = np.array([-v1.dot(self.F[i].dot(v1))
                           for i in range(n)])
            p2 = p - m
            g2 = np.array([-v2.dot(self.F[i][:p2, :p2].dot(v2))
                           for i in range(n)])
            g = np.concatenate((g1, g2))
            g += -2.*v1[:p2].dot(v2)

        return (g, 1.), 0
