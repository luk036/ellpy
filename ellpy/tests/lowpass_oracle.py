# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from itertools import chain


class lowpass_oracle:

    def __init__(self, Ap, As, Anr, Lpsq, Upsq):
        self.Ap = Ap
        self.As = As
        self.Anr = Anr
        self.Lpsq = Lpsq
        self.Upsq = Upsq

        # for round robin counters
        self.i_Anr = 0
        self.i_As = 0
        self.i_Ap = 0
        self.count = 0

    def __call__(self, x, Spsq):
        # 1. nonnegative-real constraint
        n = len(x)

        # case 1,
        if x[0] < 0.:
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
        imax = -1
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
                self.i_Anr = k
                return (g, f), Spsq

        # Begin objective function
        Spsq = fmax
        f = (0., fmax)
        g = self.As[imax, :]
        return (g, f), Spsq
