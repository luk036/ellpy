# -*- coding: utf-8 -*-
from typing import List, Tuple, Union

import numpy as np

from .lmi0_oracle import lmi0_oracle
from .qmi_oracle import qmi_oracle

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


class lsq_oracle:
    """[summary]

        min   ‖ F0 − F(x) ‖
        s.t.  F(x) ⪰ 0

    Transform the problem into:

        min   t
        s.t.  x[n+1] ≤ t
              x[n+1]*I − F(x)' F(x) ⪰ 0

    where
        F(x) = F[1] x[1] + ··· + F[n] x[n]

        {Fk}i,j = Ψk(‖sj − si‖)

    Returns:
        [type]: [description]
    """
    def __init__(self, F: List[Arr], F0: Arr):
        """[summary]

        Arguments:
            F (List[Arr]): [description]
            F0 (Arr): [description]
        """
        self.qmi = qmi_oracle(F, F0)
        self.lmi0 = lmi0_oracle(F)

    def __call__(self, x: Arr, t: float) -> Tuple[Cut, float]:
        """[summary]

        Arguments:
            x (Arr): [description]
            t (float): the best-so-far optimal value

        Returns:
            Tuple[Cut, float]: [description]
        """
        n = len(x)
        g = np.zeros(n)

        cut = self.lmi0(x[:-1])
        if cut:
            g1, fj = cut
            g[:-1] = g1
            g[-1] = 0.
            return (g, fj), t

        self.qmi.update(x[-1])
        cut = self.qmi(x[:-1])
        if cut:
            g1, fj = cut
            g[:-1] = g1
            self.qmi.Q.witness()
            # n = self.qmi.Q.p[-1] + 1
            s, n = self.qmi.Q.p
            v = self.qmi.Q.v[s:n]
            g[-1] = -v.dot(v)
            return (g, fj), t

        g[-1] = 1
        tc = x[-1]
        fj = tc - t
        if fj > 0:
            return (g, fj), t
        return (g, 0.), tc
