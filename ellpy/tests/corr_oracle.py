# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import BSpline
from ellpy.oracles.qmi_oracle import qmi_oracle
from ellpy.oracles.lmi0_oracle import lmi0_oracle
from ellpy.cutting_plane import bsearch, bsearch_adaptor, cutting_plane_dc
from ellpy.ell import ell


def create_2d_isotropic(nx=10, ny=8, N=3000):
    n = nx*ny
    s_end = [10., 8.]
    sdkern = 0.3  # width of kernel
    var = 2.     # standard derivation
    tau = 0.00001    # standard derivation of white noise
    np.random.seed(5)

    # create sites s
    sx = np.linspace(0, s_end[0], nx)
    sy = np.linspace(0, s_end[1], ny)
    xx, yy = np.meshgrid(sx, sy)
    s = np.vstack([xx.flatten(), yy.flatten()]).T

    Sig = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            d = np.array(s[j]) - np.array(s[i])
            Sig[i, j] = np.exp(-sdkern * np.sqrt(np.dot(d, d)))
            Sig[j, i] = Sig[i, j]

    A = np.linalg.cholesky(Sig)
    Ys = np.zeros((n, N))

    for k in range(N):
        x = var * np.random.randn(n)
        y = A.dot(x) + tau*np.random.randn(n)
        Ys[:, k] = y

    Y = np.cov(Ys, bias=True)
    return Y, s


def construct_distance_matrix(s):
    n = len(s)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            h = s[j] - s[i]
            d = np.sqrt(np.dot(h, h))
            D[i, j] = d
            D[j, i] = d
    return D


def corr_poly(Y, s, m, oracle, corr_core):
    n = len(s)
    D1 = construct_distance_matrix(s)
    D = np.ones((n, n))
    Sig = [D]
    for _ in range(m - 1):
        D = np.multiply(D, D1)
        Sig += [D]
    Sig.reverse()
    P = oracle(Sig, Y)
    a, num_iters, feasible = corr_core(Y, m, P)
    return np.poly1d(a), num_iters, feasible


def mono_oracle(x):
    # monotonic decreasing constraint
    n = len(x)
    g = np.zeros(n)
    for i in range(n - 1):
        fj = x[i + 1] - x[i]
        if fj > 0:
            g[i] = -1.
            g[i + 1] = 1.
            return (g, fj), False
    return None, True


class mono_decreasing_oracle2:
    def __init__(self, basis):
        self.basis = basis

    def __call__(self, x, t):
        # monotonic decreasing constraint
        n = len(x)
        g = np.zeros(n)
        cut, feasible = mono_oracle(x[:-1])
        if not feasible:
            g1, fj = cut
            g[:-1] = g1
            g[-1] = 0.
            return (g, fj), t
        return self.basis(x, t)


def corr_bspline(Y, s, m, oracle, corr_core):
    Sig, t, k = generate_bspline_info(s, m)
    Pb = oracle(Sig, Y)
    P = mono_decreasing_oracle2(Pb)
    c, num_iters, feasible = corr_core(Y, m, P)
    return BSpline(t, c, k), num_iters, feasible


def generate_bspline_info(s, m):
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
    return Sig, t, k
