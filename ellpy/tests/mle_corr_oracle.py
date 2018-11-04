# -*- coding: utf-8 -*-
#import cvxpy as cvx
import numpy as np
from ellpy.cutting_plane import cutting_plane_dc
from ellpy.ell import ell
from ellpy.oracles.lmi0_oracle import lmi0_oracle
from ellpy.oracles.lmi_oracle import lmi_oracle
from ellpy.tests.corr_oracle import generate_bspline_info, corr_poly, corr_bspline, mono_oracle


class mle_oracle:
    def __init__(self, Sig, Y):
        self.Y = Y
        self.Sig = Sig
        self.lmi0 = lmi0_oracle(Sig)
        self.lmi = lmi_oracle(Sig, 2*Y)
        # self.lmi2 = lmi2_oracle(Sig, 2*Y)

    def __call__(self, x, t):
        cut, feasible = self.lmi(x)
        if not feasible:
            return cut, t

        cut, feasible = self.lmi0(x)
        if not feasible:
            return cut, t

        R = self.lmi0.Q.R
        invR = np.linalg.inv(R)
        S = (invR).dot(invR.T)
        SY = S.dot(self.Y)
        diag = np.diag(R)
        f1 = 2*np.sum(np.log(diag)) + np.trace(SY)

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


class mono_decreasing_oracle:
    def __init__(self, basis):
        self.basis = basis

    def __call__(self, x, t):
        # monotonic decreasing constraint
        cut, feasible = mono_oracle(x)
        if not feasible:
            return cut, t
        return self.basis(x, t)


def mle_corr_core(Y, m, P):
    x = np.zeros(m)
    x[0] = 1.
    E = ell(50., x)
    # E.use_parallel_cut = False
    # options = Options()
    # options.max_it = 2000
    # options.tol = 1e-8
    x_best, _, num_iters, feasible, status = cutting_plane_dc(
        P, E, float('inf'))
    print(num_iters, feasible, status)
    return x_best, num_iters, feasible


def mle_corr_poly(Y, s, m):
    _ = np.linalg.cholesky(Y)  # test if Y is SPD.
    return corr_poly(Y, s, m, mle_oracle, mle_corr_core)


def mle_corr_bspline(Y, s, m):
    _ = np.linalg.cholesky(Y)  # test if Y is SPD.
    return corr_bspline(Y, s, m, mle_oracle, mle_corr_core)
