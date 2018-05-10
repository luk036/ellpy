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
        m = len(x)

        # case 1,
        if x[0] < 0.:
            g = np.zeros(m)
            g[0] = -1.
            f = -x[0]
            return (g, f), Spsq

#     	u = x(m:-1:1)'
#     	u(m) = 0.5*x(1)-0.00001
#     	d = roots(u)
#         md = abs(d)
#         [mdmin, idx] = min(md)
#     	if mdmin <= 1,
#         g = -real([0.5, d(idx).^(1:m-1)]')
#         f = 0.00001+g'*x
#         return
#         end

        # case 2,
        # 2. passband constraints
        n, m = self.Ap.shape
        i_Ap = self.i_Ap
        for k in chain(range(i_Ap, n), range(i_Ap)):
            # k += 1
            # if k == n:
            #     k = 0    # round robin
            v = self.Ap[k, :].dot(x)
            if v > self.Upsq:
                #f = v - Upsq
                g = self.Ap[k, :]
                f = (v - self.Upsq, v - self.Lpsq)
                self.i_Ap = k + 1
                return (g, f), Spsq

            if v < self.Lpsq:
                #f = Lpsq - v
                g = -self.Ap[k, :]
                f = (-v + self.Lpsq, -v + self.Upsq)
                self.i_Ap = k + 1
                return (g, f), Spsq

        # case 3,
        # 3. stopband constraint
        n = self.As.shape[0]
        w = np.zeros(n)
        i_As = self.i_As
        for k in chain(range(i_As, n), range(i_As)):
            # k += 1
            # if k == n:
            #     k = 0    # round robin
            w[k] = self.As[k, :].dot(x)
            if w[k] > Spsq:
                #f = v - Spsq
                g = self.As[k, :]
                #f = (w[k] - Spsq, w[k])
                f = w[k] - Spsq
                self.i_As = k + 1
                return (g, f), Spsq

            if w[k] < 0:
                #f = v - Spsq
                g = -self.As[k, :]
                f = (-w[k], -w[k] + Spsq)
                self.i_As = k + 1
                return (g, f), Spsq

        # case 4,
        # 1. nonnegative-real constraint
        n = self.Anr.shape[0]
        for k in range(n):
            v = self.Anr[k, :].dot(x)
            if v < 0:
                f = -v
                g = -self.Anr[k, :]
                #self.i_Anr = k
                return (g, f), Spsq

        # Begin objective function
        Spsq, imax = w.max(), w.argmax()  # update best so far Spsq
        f = (0., w[imax])
        #f = 0
        g = self.As[imax, :]
        return (g, f), Spsq
