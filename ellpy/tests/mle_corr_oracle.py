# -*- coding: utf-8 -*-
#import cvxpy as cvx
from scipy.interpolate import BSpline
import numpy as np
from ellpy.cutting_plane import cutting_plane_dc, Options
from ellpy.ell import ell
# from oracles.qmi_oracle import qmi_oracle
from ellpy.oracles.lmi2_oracle import lmi2_oracle
from ellpy.oracles.lmi0_oracle import lmi0_oracle
from ellpy.oracles.lmi3_oracle import lmi3_oracle
from ellpy.tests.lsq_corr_oracle import construct_distance_matrix, generate_bspline_info


class mle_poly_oracle:
    def __init__(self, Sig, Y):
        self.Y = Y
        self.Sig = Sig
        self.lmi0 = lmi0_oracle(Sig)
        # self.lmi = lmi_oracle(Sig, 2.*Y)
        self.lmi2 = lmi2_oracle(Sig, 2.*Y)

    def __call__(self, x, t):
        cut, feasible = self.lmi2(x)
        if not feasible:
            return cut, t

        cut, feasible = self.lmi0(x)
        if not feasible:
            return cut, t

        R = self.lmi0.Q.R
        invR = np.linalg.inv(R)
        S = (invR).dot(invR.T)
        # S = np.linalg.inv(self.lmi0.A)
        SY = S.dot(self.Y)
        diag = np.diag(R)
        # f = log(det(Sig)) + trace(inv(Sig)*Y)
        f1 = 2.*np.sum(np.log(diag))
        # f1 += sum(S[k].dot(self.Y[k]) for k in range(m))
        f1 += np.trace(SY)

        f = f1 - t
        if f < 0:
            t = f1
            f = 0.

        n = len(x)
        m = len(self.Y)
        g = np.zeros(n)
        for i in range(n):
            SFsi = S.dot(self.Sig[i])
            # g[i] = sum(S[k].dot(self.Sig[k]) for k in range(m))
            g[i] = np.trace(SFsi)
            g[i] -= sum(SFsi[k, :].dot(SY[:, k]) for k in range(m))

        return (g, f), t


def mle_corr_core(Y, m, P):
    x = np.zeros(m)
    x[0] = 10.
    E = ell(50., x)
    # E.use_parallel_cut = False
    options = Options()
    options.max_it = 2000
    options.tol = 1e-8

    x_best, _, num_iters, feasible, status = cutting_plane_dc(
        P, E, float('inf'), options)
    print(num_iters, feasible, status)
    return x_best, num_iters, feasible


def mle_corr_poly(Y, s, m):
    _ = np.linalg.cholesky(Y)  # test if Y is SPD.
    n = len(s)
    D1 = construct_distance_matrix(s)
    D = np.ones((n, n))
    Sig = [D]
    for _ in range(m - 1):
        D = np.multiply(D, D1)
        Sig += [D]
    Sig.reverse()
    P = mle_poly_oracle(Sig, Y)
    a, num_iters, feasible = mle_corr_core(Y, m, P)
    return np.poly1d(a), num_iters, feasible


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
                return (g, fj), t

        return self.mle_poly(x, t)


def mle_corr_bspline(Y, s, m):
    Sig, t, k = generate_bspline_info(s, m)
    P = mle_bspline_oracle(Sig, Y)
    c, num_iters, feasible = mle_corr_core(Y, m, P)
    return BSpline(t, c, k), num_iters, feasible
