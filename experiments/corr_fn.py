#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from ellpy.cutting_plane import cutting_plane_dc
from ellpy.ell import ell
from ellpy.oracles.corr_oracle import (
    corr_bspline,
    corr_poly,
    create_2d_isotropic,
    create_2d_sites
)
from ellpy.oracles.lsq_corr_oracle import lsq_oracle
from ellpy.oracles.mle_corr_oracle import mle_oracle


def lsq_corr_core2(Y, m, P):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        m {[type]} -- [description]
        P {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    normY = np.linalg.norm(Y, 'fro')
    normY2 = 32 * normY * normY
    val = 256 * np.ones(m + 1)
    val[-1] = normY2 * normY2
    x = np.zeros(m + 1)  # cannot all zeros
    x[0] = 1.
    x[-1] = normY2 / 2
    E = ell(val, x)
    xb, ell_info = cutting_plane_dc(P, E, float('inf'))
    return xb[:-1], ell_info.num_iters, ell_info.feasible


def lsq_corr_poly2(Y, s, m):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        s {[type]} -- [description]
        m {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    return corr_poly(Y, s, m, lsq_oracle, lsq_corr_core2)


def mle_corr_core(Y, m, P):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        m {[type]} -- [description]
        P {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    x = np.zeros(m)
    x[0] = 1.
    E = ell(50., x)
    # E.use_parallel_cut = False
    # options = Options()
    # options.max_it = 2000
    # options.tol = 1e-8
    xb, ell_info = cutting_plane_dc(P, E, float('inf'))
    # print(num_iters, feasible, status)
    return xb, ell_info.num_iters, ell_info.feasible


def mle_corr_poly(Y, s, m):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        s {[type]} -- [description]
        m {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    _ = np.linalg.cholesky(Y)  # test if Y is SPD.
    return corr_poly(Y, s, m, mle_oracle, mle_corr_core)


def mle_corr_bspline(Y, s, m):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        s {[type]} -- [description]
        m {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    _ = np.linalg.cholesky(Y)  # test if Y is SPD.
    return corr_bspline(Y, s, m, mle_oracle, mle_corr_core)


def lsq_corr_bspline2(Y, s, m):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        s {[type]} -- [description]
        m {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    return corr_bspline(Y, s, m, lsq_oracle, lsq_corr_core2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # import matplotlib.pylab as lab
    s = create_2d_sites(10, 8)
    Y = create_2d_isotropic(s, 1000)
    print('start ell...')
    spl, num_iters, _ = lsq_corr_bspline2(Y, s, 5)
    pol, num_iters, _ = lsq_corr_poly2(Y, s, 5)
    # pol, num_iters, _ = mle_corr_poly(Y, s, 4)
    print(pol)
    print(num_iters)
    # print('start cvx...')
    # splcvx = lsq_corr_bspline(Y, s, 4)
    # print(num_iters)
    # polcvx  = lsq_corr_poly(Y, s, 5)

    # h = s[-1] - s[0]
    d = np.sqrt(10**2 + 8**2)
    xs = np.linspace(0, d, 100)
    plt.plot(xs, spl(xs), 'g', label='BSpline')
    # plt.plot(xs, splcvx(xs), 'b', label='BSpline CVX')
    plt.plot(xs, np.polyval(pol, xs), 'r', label='Polynomial')
    # plt.plot(xs, np.polyval(polcvx, xs), 'r', label='Polynomial CVX')
    plt.legend(loc='best')
    plt.show()
