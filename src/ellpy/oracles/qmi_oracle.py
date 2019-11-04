# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple, Union

import numpy as np

from .gmi_oracle import gmi_oracle

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]

# import cholutil


class qmi_oracle:
    class QMI:
        """Oracle for Quadratic Matrix Inequality

              find  x
              s.t.​  t*I - F(x)' F(x) ⪰ 0

            where

              F(x) = F0 - (F1 * x1 + F2 * x2 + ...)
        """
        t = None
        count = 0

        def __init__(self, F: List[Arr], F0: Arr):
            """[summary]

            Arguments:
                F {List[Arr]} -- [description]
                F0 {Arr} -- [description]
            """
            self.F = F
            self.F0 = F0
            n, m = F0.shape
            self.Fx = np.zeros([m, n])

        def update(self, t: float):
            """[summary]

            Arguments:
                t {float} -- the best-so-far optimal value
            """
            self.t = t

        def eval(self, i, j, x: Arr) -> float:
            if i < j:
                raise AssertionError()
            if self.count < i + 1:
                nx = len(x)
                self.count = i + 1
                self.Fx[i] = self.F0[:, i]
                self.Fx[i] -= sum(self.F[k][:, i] * x[k] for k in range(nx))
            a = -self.Fx[i].dot(self.Fx[j])
            if i == j:
                a += self.t
            return a

        def neg_grad_sym_quad(self, Q, x: Arr):
            s, n = Q.p
            v = Q.v[s:n]
            Av = v.dot(self.Fx[s:n])
            g = np.array([-2 * v.dot(Fk[s:n]).dot(Av) for Fk in self.F])
            return g

    def __init__(self, F, F0):
        """[summary]

        Arguments:
            F {List[Arr]} -- [description]
            F0 {Arr} -- [description]
        """
        n, m = F0.shape
        self.qmi = self.QMI(F, F0)
        self.gmi = gmi_oracle(self.qmi, m)
        self.Q = self.gmi.Q

    def update(self, t: float):
        """[summary]

        Arguments:
            t {float} -- the best-so-far optimal value
        """
        self.qmi.update(t)

    def __call__(self, x: Arr) -> Optional[Cut]:
        """[summary]

        Arguments:
            x {Arr} -- [description]

        Returns:
            Optional[Cut] -- [description]
        """
        self.qmi.count = 0
        return self.gmi(x)
