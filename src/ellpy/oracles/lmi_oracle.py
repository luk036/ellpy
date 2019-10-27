# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import numpy as np

from .gmi_oracle import gmi_oracle

Cut = Tuple[np.ndarray, float]


class lmi_oracle:
    """Oracle for Linear Matrix Inequality constraint
            F * x <= B
       or
            (B - F * x) must be a semidefinte matrix
    """
    class __lmi:
        def __init__(self, F, B):
            self.F = F
            self.F0 = B

        def eval(self, i, j, x):
            n = len(x)
            return self.F0[i, j] - sum(self.F[k][i, j] * x[k]
                                       for k in range(n))

        def neg_grad_sym_quad(self, Q, x):
            return np.array([Q.sym_quad(Fk) for Fk in self.F])

    def __init__(self, F, B):
        """[summary]

        Arguments:
            F {[type]} -- [description]
            B {[type]} -- [description]
        """
        self.gmi = gmi_oracle(self.__lmi(F, B), len(B))
        self.Q = self.gmi.Q

    def __call__(self, x: np.ndarray) -> Optional[Cut]:
        """[summary]

        Arguments:
            x {np.ndarray} -- [description]

        Returns:
            Optional[Cut] -- [description]
        """
        return self.gmi(x)
