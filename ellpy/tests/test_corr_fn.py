# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import BSpline
from ellpy.cutting_plane import bsearch, bsearch_adaptor, cutting_plane_dc
from ellpy.ell import ell
from ellpy.oracles.qmi_oracle import qmi_oracle

# a fake dataset to make the bumps with
nx = 10   # number of points
ny = 8
n = nx*ny
s_end = [10., 8.]
sdkern = 0.1  # width of kernel
var = 2.     # standard derivation
tau = 0.00001    # standard derivation of white noise
N = 200  # number of samples
np.random.seed(5)


# create sites s
sx = np.linspace(0, s_end[0], nx)
sy = np.linspace(0, s_end[1], ny)
xx, yy = np.meshgrid(sx, sy)
s = np.vstack([xx.flatten(), yy.flatten()]).T

Sig = np.ones((n, n))
for i in range(n):
    for j in range(i, n):
        d = np.array(s[j]) - np.array(s[i])
        Sig[i, j] = np.exp(-sdkern * np.dot(d, d))
        Sig[j, i] = Sig[i, j]

A = np.linalg.cholesky(Sig)
Ys = np.zeros((n, N))

for k in range(N):
    x = var * np.random.randn(n)
    y = A.dot(x) + tau*np.random.randn(n)
    Ys[:, k] = y

Y = np.cov(Ys, bias=True)


class poly_oracle2:
    def __init__(self, F, F0):
        self.F0 = F0
        self.qmi = qmi_oracle(F, F0)

    def update(self, t):
        self.qmi.update(t)

    def __call__(self, x, t):
        n = len(x)
        g = np.zeros(n)
        g[-1] = 1
        tc = x[-1]
        fj = tc - t
        if fj > 0.:
            return (g, fj), t

        self.qmi.update(tc)
        cut, feasible = self.qmi(x[:-1])
        if not feasible:
            g1, fj = cut
            g[:-1] = g1
            v = self.qmi.Q.witness()
            g[-1] = -v.dot(v)
            return (g, fj), t

        return (g, 0.), tc


def lsq_corr_poly2(Y, s, m):
    n = len(s)
    D1 = construct_distance_matrix(s)

    D = np.ones((n, n))
    Sig = [D]
    for _ in range(m - 1):
        D = np.multiply(D, D1)
        Sig += [D]
    Sig.reverse()

    P = poly_oracle2(Sig, Y)
    normY = np.linalg.norm(Y, 'fro')
    val = 256*np.ones(m + 1)
    normY2 = 32*normY*normY
    val[-1] = normY2*normY2
    x = np.zeros(m + 1)
    E = ell(val, x)
    x_best, _, num_iters, feasible, _ = cutting_plane_dc(P, E, normY2)

    print(num_iters, feasible)
    assert feasible
    assert num_iters >= 638 and num_iters <= 657

    a = x_best[:-1]
    print(a)
    return np.poly1d(a)


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

    Q = qmi_oracle(Sig, Y)
    E = ell(64., a)
    P = bsearch_adaptor(Q, E)
    normY = np.linalg.norm(Y, 'fro')
    _, niter, feasible = bsearch(P, [0., normY*normY])

    print(niter, feasible)
    assert feasible
    assert niter == 40

    a = P.x_best
    print(a)
    return np.poly1d(a)


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
                return (g, fj), False

        # if x[-1] < 0.:
        #     g[-1] = -1.
        #     return (g, -x[-1]), False

        return self.qmi(x)


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
    normY = np.linalg.norm(Y, 'fro')

    Q = bsp_oracle(Sig, Y)
    E = ell(64., c)
    P = bsearch_adaptor(Q, E)

    _, niter, feasible = bsearch(P, [0., normY*normY])

    print(niter, feasible)
    assert feasible
    assert niter == 40

    c = P.x_best

    return BSpline(t, c, k)


class bsp_oracle2:
    def __init__(self, F, F0):
        self.F0 = F0
        self.qmi = qmi_oracle(F, F0)

    def update(self, t):
        self.qmi.update(t)

    def __call__(self, x, t):
        n = len(x)
        g = np.zeros(n)

        for i in range(n - 2):
            fj = x[i + 1] - x[i]
            if fj > 0.:
                g[i] = -1.
                g[i + 1] = 1.
                return (g, fj), False

        # if x[-1] < 0.:
        #     g[-1] = -1.
        #     return (g, -x[-1]), False

        g[-1] = 1
        tc = x[-1]
        fj = tc - t
        if fj > 0.:
            return (g, fj), t

        self.qmi.update(tc)
        cut, feasible = self.qmi(x[:-1])
        if not feasible:
            g1, fj = cut
            g[:-1] = g1
            v = self.qmi.Q.witness()
            g[-1] = -v.dot(v)
            return (g, fj), t

        return (g, 0.), tc


def lsq_corr_bspline2(Y, s, m):
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

    P = bsp_oracle2(Sig, Y)
    normY = np.linalg.norm(Y, 'fro')
    val = 256*np.ones(m + 1)
    normY2 = 32*normY*normY
    val[-1] = normY2*normY2
    x = np.zeros(m + 1)
    E = ell(val, x)
    x_best, _, num_iters, feasible, _ = cutting_plane_dc(P, E, normY2)

    print(num_iters, feasible)
    assert feasible
    assert num_iters == 98

    c = x_best[:-1]

    return BSpline(t, c, k)


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


def test_corr_fn():
    lsq_corr_bspline2(Y, s, 4)
    lsq_corr_poly2(Y, s, 4)
