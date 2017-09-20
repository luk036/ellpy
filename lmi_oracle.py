# -*- coding: utf-8 -*-
import numpy as np
from chol_ext import *

class lmi_oracle:
    def __init__(F):
        self.F = F

	def chk_mtx(self, A, x):
        n = len(x)
        g = np.zeros(n)
        fj = -1.0
        v = 0.0
        for i in range(n):
            A += self.F[i] * x[i]
        R, p = chol_ext(A)
        if p == 0: return g, fj, R, v
        v = witness(R, p)
        fj = -np.dot(v, A[:p, :p].dot(v))
        for i in range(n):
            g[i] = -np.dot(v, self.F[i][:p, :p].dot(v))
        return g, fj, R, v
 
    def chk_spd_t(self, x, t):
        A = np.array(self.F[-1])
        m = len(A)
        A(range(m), range(m)) += t
        return chk_mtx(self, A, x)

    def chk_spd(self, x):
        A = np.array( self.F[-1] )
        return chk_mtx(self, A, x)

    def __call__(self, x, t):
        g, fj, _, _ = chk_spd_t(self, x, t)
        if fj<0.0: t -= 1.0
        return g, fj, t

