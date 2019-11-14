# -*- coding: utf-8 -*-
from typing import List, Tuple, Union

import numpy as np
from scipy.interpolate import BSpline

from .halton_n import halton_n

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


def create_2d_sites(nx=10, ny=8) -> Arr:
    """Create a 2d sites object

    Keyword Arguments:
        nx {int} -- [description] (default: {10})
        ny {int} -- [description] (default: {8})

    Returns:
        Arr -- location of sites
    """
    n = nx * ny
    s_end = [10., 8.]
    s = np.array([(s_end[0] * x, s_end[1] * y)
                  for x, y in halton_n(n, 2, [2, 3])])
    return s


def create_2d_isotropic(s: Arr, N=3000) -> Arr:
    """Create a 2d isotropic object

    Arguments:
        s {Arr} -- location of sites

    Keyword Arguments:
        N {int} -- [description] (default: {3000})

    Returns:
        Arr -- Biased covariance matrix
    """
    n = s.shape[0]
    sdkern = 0.12  # width of kernel
    var = 2.  # standard derivation
    tau = 0.00001  # standard derivation of white noise
    np.random.seed(5)

    Sig = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            d = np.array(s[j]) - np.array(s[i])
            Sig[i, j] = np.exp(-sdkern * d @ d)
            Sig[j, i] = Sig[i, j]

    A = np.linalg.cholesky(Sig)
    Y = np.zeros((n, n))

    for _ in range(N):
        x = var * np.random.randn(n)
        y = A.dot(x) + tau * np.random.randn(n)
        Y += np.outer(y, y)

    Y /= N
    return Y


def construct_distance_matrix(s: Arr) -> Arr:
    """Construct a distance matrix object

    Arguments:
        s {Arr} -- location of sites

    Returns:
        [type] -- [description]
    """
    n = len(s)
    D1 = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            h = s[j] - s[i]
            d = np.sqrt(h @ h)
            D1[i, j] = d
            D1[j, i] = d
    return D1


def construct_poly_matrix(s: Arr, m) -> List[Arr]:
    """Construct distance matrix for polynomial

    Arguments:
        s {Arr} -- location of sites
        m {int} -- degree of polynomial

    Returns:
        List[Arr] -- [description]
    """
    n = len(s)
    D1 = construct_distance_matrix(s)
    D = np.ones((n, n))
    Sig = [D]
    for _ in range(m - 1):
        D = np.multiply(D, D1)
        Sig += [D]
    return Sig


def corr_poly(Y, s, m, oracle, corr_core):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        s {[type]} -- [description]
        m {[type]} -- [description]
        oracle {[type]} -- [description]
        corr_core {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    Sig = construct_poly_matrix(s, m)
    P = oracle(Sig, Y)
    a, num_iters, feasible = corr_core(Y, m, P)
    pa = np.ascontiguousarray(a[::-1])
    return np.poly1d(pa), num_iters, feasible


def mono_oracle(x):
    """[summary]

    Arguments:
        x {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    # monotonic decreasing constraint
    n = len(x)
    g = np.zeros(n)
    for i in range(n - 1):
        fj = x[i + 1] - x[i]
        if fj > 0:
            g[i] = -1.
            g[i + 1] = 1.
            return g, fj


class mono_decreasing_oracle2:
    """[summary]

    Returns:
        [type] -- [description]
    """
    def __init__(self, basis):
        """[summary]

        Arguments:
            basis {[type]} -- [description]
        """
        self.basis = basis

    def __call__(self, x: Arr, t: float) -> Tuple[Cut, float]:
        """[summary]

        Arguments:
            x {Arr} -- [description]
            t {float} -- the best-so-far optimal value

        Returns:
            Tuple[Cut, float] -- [description]
        """
        # monotonic decreasing constraint
        n = len(x)
        g = np.zeros(n)
        cut = mono_oracle(x[:-1])
        if cut:
            g1, fj = cut
            g[:-1] = g1
            g[-1] = 0.
            return (g, fj), t
        return self.basis(x, t)


def corr_bspline(Y, s, m, oracle, corr_core):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        s {[type]} -- [description]
        m {[type]} -- [description]
        oracle {[type]} -- [description]
        corr_core {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    Sig, t, k = generate_bspline_info(s, m)
    Pb = oracle(Sig, Y)
    P = mono_decreasing_oracle2(Pb)
    c, num_iters, feasible = corr_core(Y, m, P)
    return BSpline(t, c, k), num_iters, feasible


def generate_bspline_info(s, m):
    """[summary]

    Arguments:
        s {[type]} -- [description]
        m {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    k = 2  # quadratic bspline
    h = s[-1] - s[0]
    d = np.sqrt(h @ h)
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
