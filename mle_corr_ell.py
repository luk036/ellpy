# -*- coding: utf-8 -*-
#import cvxpy as cvx
from scipy.interpolate import BSpline
import numpy as np
from cutting_plane import cutting_plane_dc
from ell import ell
# from oracles.qmi_oracle import qmi_oracle
from oracles.lmi_oracle import lmi_oracle
from oracles.lmi0_oracle import lmi0_oracle


class mle_bspline_oracle:
    """
     Oracle for Quadratic Matrix Inequality
        F(x).T * F(x) <= B
     where
        F(x) = F0 - (F1 * x1 + F2 * x2 + ...)
    """

    def __init__(self, Sig, Y):
        self.mle_poly = mle_poly_oracle(Sig, Y)

    def __call__(self, x, t):
        n = len(x)
        g = np.zeros(n)

        for i in range(n - 1):
            fj = x[i + 1] - x[i]
            if fj > 0.:
                g[i] = -1.
                g[i + 1] = 1.
                return (g, fj), 0

        return self.mle_poly(x, t)


class mle_poly_oracle:
    def __init__(self, Sig, Y):
        self.Y = Y
        self.Sig = Sig
        self.lmi0 = lmi0_oracle(Sig)
        self.lmi = lmi_oracle(Sig, 2.*Y)

    def __call__(self, x, t):
        cut, flag = self.lmi(x)
        if flag == 0:
            return cut, t

        cut, flag = self.lmi0(x)
        if flag == 0:
            return cut, t

        R = self.lmi0.Q.R
        invR = np.linalg.inv(R)
        S = invR.dot(invR.T)
        SY = S.dot(self.Y)
        m = len(self.Y)
        diag = np.diag(R)
        f1 = 2.*np.sum(np.log(diag))
        # f1 += sum(S[k].dot(self.Y[k]) for k in range(m))
        f1 += np.trace(SY)

        f = f1 - t
        if f < 0:
            t = f1
            f = 0.

        n = len(x)

        g = np.zeros(n)
        for i in range(n):
            SFsi = S.dot(self.Sig[i])
            # g[i] = sum(S[k].dot(self.Sig[k]) for k in range(m))
            g[i] = np.trace(SFsi)
            g[i] -= sum(SY[k, :].dot(SFsi[:, k]) for k in range(m))

        return (g, f), t


def mle_corr_poly(Y, s, m):
    _ = np.linalg.cholesky(Y) # test if Y is SPD.

    n = len(s)
    D1 = construct_distance_matrix(s)

    D = np.ones((n, n))
    Sig = [D]
    for _ in range(m - 1):
        D = np.multiply(D, D1)
        Sig += [D]
    Sig.reverse()
    # normY = np.linalg.norm(Y, 'fro')
    a = np.ones(m)

    # P = poly_oracle(Sig, Y, a)
    # _, _, _ = bsearch(P, [0., normY*normY])
    # a = P.x_best
    P = mle_poly_oracle(Sig, Y)
    E = ell(100., a)
    a, _, num_iters, flag, status = cutting_plane_dc(
        P, E, float('inf'))
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

    # normY = np.linalg.norm(Y, 'fro')
    c = np.ones(m)
    # P = bspline_oracle(Sig, Y, c)
    # _, _, _ = bsearch(P, [0., normY*normY])
    P = mle_bspline_oracle(Sig, Y)
    E = ell(100., c)
    c, _, num_iters, flag, status = cutting_plane_dc(
        P, E, float('inf'))

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
