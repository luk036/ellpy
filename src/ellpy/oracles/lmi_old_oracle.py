# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Union

import numpy as np

from .chol_ext import chol_ext

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


class lmi_old_oracle:
    """Oracle for Linear Matrix Inequality constraint
            F * x <= B
        Or
            (B - F * x) must be a semidefinte matrix
    """
    def __init__(self, F, B):
        """[summary]

        Arguments:
            F (List[Arr]): [description]
            B (Arr): [description]
        """
        self.F = F
        self.F0 = B
        self.A = np.zeros(B.shape)
        self.Q = chol_ext(len(B))

    def __call__(self, x: Arr) -> Optional[Cut]:
        """[summary]

        Arguments:
            x (np.ndarray): [description]

        Returns:
            Optional[Cut]: [description]
        """
        n = len(x)
        self.A = self.F0.copy()
        self.A -= sum(self.F[k] * x[k] for k in range(n))
        self.Q.factorize(self.A)
        if not self.Q.is_spd():
            ep = self.Q.witness()
            g = np.array([self.Q.sym_quad(self.F[i]) for i in range(n)])
            return g, ep
        return None