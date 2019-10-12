# -*- coding: utf-8 -*-
import numpy as np

from .lmi0_oracle import lmi0_oracle
from .qmi_oracle import qmi_oracle


class lsq_oracle:
    """[summary]

    Returns:
        [type] -- [description]
    """
    def __init__(self, F, F0):
        """[summary]

        Arguments:
            F {[type]} -- [description]
            F0 {[type]} -- [description]
        """
        self.qmi = qmi_oracle(F, F0)
        self.lmi0 = lmi0_oracle(F)

    def __call__(self, x, t):
        """[summary]

        Arguments:
            x {[type]} -- [description]
            t {[type]} -- [description]

        Returns:
            [type] -- [description]
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
