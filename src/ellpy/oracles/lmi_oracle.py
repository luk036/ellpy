# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Union

import numpy as np

from .chol_ext import chol_ext

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


class lmi_oracle:
    """Oracle for Linear Matrix Inequality constraint.

    This oracle solves the following feasibility problem:

        find  x
        s.t.  (B − F * x) ⪰ 0

    """

    def __init__(self, F, B):
        """Construct a new lmi oracle object

        Arguments:
            F (List[Arr]): [description]
            B (Arr): [description]
        """
        self.F = F
        self.F0 = B
        self.Q = chol_ext(len(B))

    def __call__(self, x: Arr) -> Optional[Cut]:
        """[summary]

        Arguments:
            x (Arr): [description]

        Returns:
            Optional[Cut]: [description]
        """

        def getA(i, j):
            n = len(x)
            return self.F0[i, j] - sum(self.F[k][i, j] * x[k] for k in range(n))

        if self.Q.factor(getA):
            return None

        ep = self.Q.witness()
        g = np.array([self.Q.sym_quad(Fk) for Fk in self.F])
        return g, ep
