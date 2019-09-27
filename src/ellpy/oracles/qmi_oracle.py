# -*- coding: utf-8 -*-
import numpy as np

from .chol_ext import chol_ext

# import cholutil


class qmi_oracle:
    """Oracle for Quadratic Matrix Inequality
        F(x).T * F(x) <= I*t
     where
        F(x) = F0 - (F1 * x1 + F2 * x2 + ...)
    """
    t = None
    count = 0

    def __init__(self, F, F0):
        """[summary]

        Arguments:
            F {[type]} -- [description]
            F0 {[type]} -- [description]
        """
        self.F = F
        self.F0 = F0
        self.Fx = np.zeros(F0.shape)
        self.Q = chol_ext(len(F0))

    def update(self, t):
        """[summary]

        Arguments:
            t {[type]} -- [description]
        """
        self.t = t

    def __call__(self, x):
        """[summary]

        Arguments:
            x {[type]} -- [description]

        Raises:
            AssertionError -- [description]

        Returns:
            [type] -- [description]
        """
        self.count = 0
        nx = len(x)

        def getA(i, j):
            """[summary]

            Arguments:
                i {[type]} -- [description]
                j {[type]} -- [description]

            Raises:
                AssertionError -- [description]

            Returns:
                [type] -- [description]
            """
            if i < j:
                raise AssertionError()
            if self.count < i + 1:
                self.count = i + 1
                self.Fx[i] = self.F0[i]
                self.Fx[i] -= sum(self.F[k][i] * x[k] for k in range(nx))
            a = -self.Fx[i].dot(self.Fx[j])
            if i == j:
                a += self.t
            return a

        self.Q.factor(getA)

        if self.Q.is_spd():
            return None
        ep = self.Q.witness()
        s, n = self.Q.p
        # n = p[-1] + 1
        v = self.Q.v[s:n]
        Av = v.dot(self.Fx[s:n])
        g = np.array([-2 * v.dot(self.F[k][s:n]).dot(Av) for k in range(nx)])
        return g, ep
