# -*- coding: utf-8 -*-
#import cvxpy as cvx
from scipy.interpolate import BSpline
import numpy as np
from cutting_plane import cutting_plane_feas, bsearch
from ell import ell
# from oracles.qmi_oracle import qmi_oracle
from oracles.lmi_oracle import lmi_oracle



class bsp_oracle:
    """
     Oracle for Quadratic Matrix Inequality
        F(x).T * F(x) <= B
     where
        F(x) = F0 - (F1 * x1 + F2 * x2 + ...)
    """

    def __init__(self, F, F0, B):
        self.F0 = F0
        self.B = B
        self.qmi = qmi_oracle(F, F0, B)

    def __call__(self, x):
        n = len(x)
        g = np.zeros(n)

        for i in range(n - 1):
            fj = x[i + 1] - x[i]
            if fj > 0.:
                g[i] = -1.
                g[i + 1] = 1.
                return g, fj, 0

        # if x[-1] < 0.:
        #     g[-1] = -1.
        #     return g, -x[-1], 0

        return self.qmi(x)


class poly_oracle:
    def __init__(self, Sig, Y):
        self.Y = Y
        self.Sig = Sig
        self.lmi1 = lmi_oracle(Sig, 2.*Y)
        self.lmi2 = lmi_oracle(-Sig, np.zeros())

    def __call__(self, x, t):
        g, fj, flag = self.lmi1(x)
        if flag == 1:
            return g, fj, t

        g, fj, flag = self.lmi2(x)
        if flag == 1:
            return g, fj, t


        return False


def mle_corr_poly(Y, s, m):
    n = len(s)
    a = np.zeros(m)
    D1 = construct_distance_matrix(s)

    D = np.ones((n, n))
    Sig = [D]
    for _ in range(m - 1):
        D = np.multiply(D, D1)
        Sig += [D]
    Sig.reverse()

    P = poly_oracle(Sig, Y, a)
    normY = np.linalg.norm(Y, 'fro')
    _, _, _ = bsearch(P, [0., normY*normY])
    a = P.x_best
    return np.poly1d(a)
#  return prob.is_dcp()


def mle_corr_bspline(Y, s, m):
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
    P = bspline_oracle(Sig, Y, c)
    normY = np.linalg.norm(Y, 'fro')
    _, _, _ = bsearch(P, [0., normY*normY])
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
