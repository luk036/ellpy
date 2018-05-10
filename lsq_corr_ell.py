# -*- coding: utf-8 -*-
from __future__ import print_function

from pprint import pprint
from scipy.interpolate import BSpline
import numpy as np
from cutting_plane import cutting_plane_feas, bsearch, bsearch_adaptor
from ell import ell
from oracles.qmi_oracle import qmi_oracle
# from oracles.lmi_oracle import lmi_oracle


class bsp_oracle:
    def __init__(self, F, F0):
        self.F0 = F0
        self.qmi = qmi_oracle(F, F0)

    def update(self, t):
        self.qmi.update(t)

    def __call__(self, x):
        n = len(x)
        g = np.zeros(n)

        for i in range(n - 1):
            fj = x[i + 1] - x[i]
            if fj > 0.:
                g[i] = -1.
                g[i + 1] = 1.
                return (g, fj), 0

        # if x[-1] < 0.:
        #     g[-1] = -1.
        #     return (g, -x[-1]), 0

        return self.qmi(x)


def lsq_corr_poly(Y, s, m):
    n = len(s)
    a = np.zeros(m)
    D1 = construct_distance_matrix(s)

    D = np.ones((n, n))
    Sig = [D]
    for _ in range(m - 1):
        D = np.multiply(D, D1)
        Sig += [D]
    Sig.reverse()

    # P = mtx_norm_oracle(Sig, Y, a)
    Q = qmi_oracle(Sig, Y)
    E = ell(10., a)
    P = bsearch_adaptor(Q, E)
    normY = np.linalg.norm(Y, 'fro')
    _, niter, flag = bsearch(P, [0., normY*normY])

    print(niter, flag)
    a = P.x_best
    return np.poly1d(a)
#  return prob.is_dcp()


def lsq_corr_bspline(Y, s, m):
    k = 2  # quadratic bspline
    h = s[-1] - s[0]
    d = np.sqrt(np.dot(h, h))
    t = np.linspace(0, d * 1.2, m + k + 1)
    spls = []
    for i in range(m):
        coeff = np.zeros(m)
        coeff[i] = 1
        spls += [BSpline(t, coeff, k)]

    D = construct_distance_matrix(s)

    Sig = []
    for i in range(m):
        Sig += [spls[i](D)]
    c = np.zeros(m)
    # P = bspline_oracle(Sig, Y, c)

    Q = bsp_oracle(Sig, Y)
    E = ell(10., c)
    P = bsearch_adaptor(Q, E)

    normY = np.linalg.norm(Y, 'fro')
    _, niter, flag = bsearch(P, [0., normY*normY])

    print(niter, flag)

    c = P.x_best

    return BSpline(t, c, k)


def construct_distance_matrix(s):
    n = len(s)
    # c = cvx.Variable(m)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            h = s[j] - s[i]
            d = np.sqrt(np.dot(h, h))
            D[i, j] = d
            D[j, i] = d
    return D
