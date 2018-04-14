# -*- coding: utf-8 -*-
import numpy as np
from math import *
from .chol_ext import *


class rank1_oracle:

    def __init__(self, N, A):
        self.N = N
        self.i_diag = 0
        self.iD = np.zeros(N)
        k = 0
        m = N
        for i in range(N):
            self.iD[i] = k
            k = k + m
            m = m - 1
        B = 2. * np.ones((N, N))  # twice because of symmetric
        B[(range(N), range(N))] = 1.  # but not the diagonals
        self.inds = np.triu_indices_from(A)
        self.w = B[self.inds]
        self.c = self.w * A[self.inds]

    def assess(self, x0, t, restart):
        # Begin constraints checking
        G = self.toMat(x0, self.N)
        n = len(x0)
        x = np.array(x0)
        for loop in range(restart, 2):
            # 1. 1 <= trace(A*G) <= 2
            v = self.c.dot(x)
            if v > 2:
                g = self.c
                f = np.array([v - 2, v - 1])
                return g, f, t, x, loop
            if v < 1:
                g = -self.c
                f = np.array([-v + 1, -v + 2])
                return g, f, t, x, loop

            # 5. max(diag(G)) <= t
            v = np.diag(G)
            k = self.i_diag
            for i in range(self.N):
                k = k + 1
                if k == self.N:
                    k = 0
                if v[k] > t:
                    f = v[k] - t
                    g = np.zeros(n)
                    g[self.iD[k]] = 1.0
                    self.i_diag = k
                    return g, f, t, x, loop

            if loop == 1:
                break

            # 4. G >= 0 constraint
            [R, p] = chol_ext(G)
            if p != 0:
                v = witness(R, p)
                # f = -v'*G(1:p,1:p)*v
                f = -np.dot(v, G[:p, :p].dot(v))
                g = np.zeros(n)
                v2 = np.zeros(self.N)
                v2[:p] = v
                # for i in range(n):
                # g[i] = -v'*self.F{i}(1:p,1:p)*v
                # end
                g = -v2[self.inds[0]] * v2[self.inds[1]]
                g = self.w * g
                return g, f, t, x, loop
            v, d, _ = np.linalg.svd(G)
            h = v[1] * np.sqrt(d(1, 1))
            #[v,d] = eigs(G,1)
            # h = v * sqrt(d)
            G = np.outer(h, h)
            x = G[self.inds]
            s = 1. / (self.c.dot(x))
            x = s * x  # rescale to the closest point
            G = self.toMat(x0, self.N)

        # Begin selfective def
        v = np.diag(G)
        t, imax = np.max(v)  # update best so far t
        f = 0
        # g = np.zeros(len(x))
        g = np.zeros(n)
        g[self.iD[imax]] = 1
        return g, f, t, x, loop

    def toMat(self, x, N):
        G = np.zeros((N, N))
        G[self.inds] = x
        G[(self.inds[1], self.inds[0])] = x
        return G
