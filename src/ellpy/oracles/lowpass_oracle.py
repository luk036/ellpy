# -*- coding: utf-8 -*-
from __future__ import print_function

from itertools import chain
from typing import Tuple, Union

import numpy as np

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


class lowpass_oracle:
    """[summary]

    Returns:
        [type]: [description]
    """

    # for round robin counters
    i_Anr = 0
    i_As = 0
    i_Ap = 0
    count = 0

    def __init__(self, Ap: Arr, As: Arr, Anr: Arr, Lpsq, Upsq):
        """[summary]

        Arguments:
            Ap ([type]): [description]
            As ([type]): [description]
            Anr ([type]): [description]
            Lpsq ([type]): [description]
            Upsq ([type]): [description]
        """
        self.Ap = Ap
        self.As = As
        self.Anr = Anr
        self.Lpsq = Lpsq
        self.Upsq = Upsq

    def __call__(self, x: Arr, Spsq: float):
        """[summary]

        Arguments:
            x (Arr): coefficients of autocorrelation
            Spsq (float): the best-so-far Sp^2

        Returns:
            [type]: [description]
        """
        # 1. nonnegative-real constraint
        n = len(x)

        # case 1,
        if x[0] < 0:
            g = np.zeros(n)
            g[0] = -1.
            f = -x[0]
            return (g, f), Spsq

        # case 2,
        # 2. passband constraints
        N, n = self.Ap.shape
        i_Ap = self.i_Ap
        for k in chain(range(i_Ap, N), range(i_Ap)):
            v = self.Ap[k, :].dot(x)
            if v > self.Upsq:
                g = self.Ap[k, :]
                f = (v - self.Upsq, v - self.Lpsq)
                self.i_Ap = k + 1
                return (g, f), Spsq

            if v < self.Lpsq:
                g = -self.Ap[k, :]
                f = (-v + self.Lpsq, -v + self.Upsq)
                self.i_Ap = k + 1
                return (g, f), Spsq

        # case 3,
        # 3. stopband constraint
        N = self.As.shape[0]
        fmax = float("-inf")
        imax = 0
        i_As = self.i_As
        for k in chain(range(i_As, N), range(i_As)):
            v = self.As[k, :].dot(x)
            if v > Spsq:
                g = self.As[k, :]
                f = (v - Spsq, v)
                self.i_As = k + 1
                return (g, f), Spsq

            if v < 0:
                g = -self.As[k, :]
                f = (-v, -v + Spsq)
                self.i_As = k + 1
                return (g, f), Spsq

            if v > fmax:
                fmax = v
                imax = k

        # case 4,
        # 1. nonnegative-real constraint
        N = self.Anr.shape[0]
        i_Anr = self.i_Anr
        for k in chain(range(i_Anr, N), range(i_Anr)):
            v = self.Anr[k, :].dot(x)
            if v < 0:
                f = -v
                g = -self.Anr[k, :]
                self.i_Anr = k + 1
                return (g, f), Spsq

        # Begin objective function
        Spsq = fmax
        f = (0., fmax)
        # f = 0.
        g = self.As[imax, :]
        return (g, f), Spsq
