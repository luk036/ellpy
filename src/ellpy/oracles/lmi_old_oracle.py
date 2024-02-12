# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Union

import numpy as np

from .chol_ext import chol_ext

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


class lmi_old_oracle:
    """Oracle for Linear Matrix Inequality constraint.

    This oracle solves the following feasibility problem:

        find  x
        s.t.  (B − F * x) ⪰ 0

    """

    def __init__(self, F, B):
        """[summary]

        Arguments:
            F (List[Arr]): [description]
            B (Arr): [description]
        """
        self.F = F
        self.F0 = B
        # self.A = np.zeros(B.shape)
        self.Q = chol_ext(len(B))

    def __call__(self, x: Arr) -> Optional[Cut]:
        """[summary]

        Arguments:
            x (np.ndarray): [description]

        Returns:
            Optional[Cut]: [description]
        """
        n = len(x)
        A = self.F0.copy()
        A -= sum(self.F[k] * x[k] for k in range(n))
        if not self.Q.factorize(A):
            ep = self.Q.witness()
            g = np.array([self.Q.sym_quad(self.F[i]) for i in range(n)])
            return g, ep
        return None
