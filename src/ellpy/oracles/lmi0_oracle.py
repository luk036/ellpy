# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Union

import numpy as np

from .chol_ext import chol_ext

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


class lmi0_oracle:
    """Oracle for Linear Matrix Inequality constraint

    find  x
    s.t.​  F * x ⪰ 0

    """

    def __init__(self, F):
        """[summary]

        Arguments:
            F (List[Arr]): [description]
        """
        self.F = F
        self.Q = chol_ext(len(F[0]))

    def __call__(self, x: Arr) -> Optional[Cut]:
        """[summary]

        Arguments:
            x (Arr): [description]

        Returns:
            Optional[Cut]: [description]
        """

        def getA(i, j):
            n = len(x)
            return sum(self.F[k][i, j] * x[k] for k in range(n))

        if not self.Q.factor(getA):
            ep = self.Q.witness()
            g = np.array([-self.Q.sym_quad(Fk) for Fk in self.F])
            return g, ep
        return None
