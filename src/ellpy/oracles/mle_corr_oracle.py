# -*- coding: utf-8 -*-
# import cvxpy as cvx
from typing import Tuple, Union

import numpy as np

from .lmi0_oracle import lmi0_oracle
from .lmi_oracle import lmi_oracle

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


class mle_oracle:
    def __init__(self, Sig: Arr, Y: Arr):
        """Maximum likelyhood estimation:

            min  ​log det Ω(p) + Tr( Ω(p)^{−1} Y )
            ​s.t.​ 2Y ⪰ Ω(p) ⪰ 0,​

        Arguments:
            Sig (Arr): Covariance matrix
            Y (Arr): Biased covariance matrix
        """
        self.Y = Y
        self.Sig = Sig
        self.lmi0 = lmi0_oracle(Sig)
        self.lmi = lmi_oracle(Sig, 2 * Y)
        # self.lmi2 = lmi2_oracle(Sig, 2*Y)

    def __call__(self, x: Arr, t: float) -> Tuple[Cut, float]:
        """[summary]

        Arguments:
            x (Arr): coefficients of basis functions
            t (float): the best-so-far optimal value

        Returns:
            Tuple[Cut, float]: [description]
        """
        cut = self.lmi(x)
        if cut:
            return cut, t

        cut = self.lmi0(x)
        if cut:
            return cut, t

        R = self.lmi0.Q.sqrt()
        invR = np.linalg.inv(R)
        S = invR @ invR.T
        SY = S @ self.Y
        diag = np.diag(R)
        f1 = 2 * np.sum(np.log(diag)) + np.trace(SY)

        f = f1 - t
        if f < 0.:
            t = f1
            f = 0.

        n = len(x)
        m = len(self.Y)
        g = np.zeros(n)
        for i in range(n):
            SFsi = S @ self.Sig[i]
            # g[i] = sum(S[k] @ self.Sig[k] for k in range(m))
            g[i] = np.trace(SFsi)
            g[i] -= sum(SFsi[k, :] @ SY[:, k] for k in range(m))

        return (g, f), t
