# -*- coding: utf-8 -*-
from pytest import approx

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

s = create_2d_sites(5, 4)
Y = create_2d_isotropic(s, 3000)


def lsq_corr_core2(Y, n, P):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        n {[type]} -- [description]
        P {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    normY = np.linalg.norm(Y, 'fro')
    normY2 = 32 * normY * normY
    val = 256 * np.ones(n + 1)
    val[-1] = normY2 * normY2
    x = np.zeros(n + 1)  # cannot all zeros
    x[0] = 1.
    x[-1] = normY2 / 2
    E = ell(val, x)
    xb, ell_info = cutting_plane_dc(P, E, float('inf'))
    return xb[:-1], ell_info.num_iters, ell_info.feasible


def lsq_corr_poly2(Y, s, n):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        s {[type]} -- [description]
        n {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    return corr_poly(Y, s, n, lsq_oracle, lsq_corr_core2)


def mle_corr_core(Y, n, P):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        n {[type]} -- [description]
        P {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    x = np.zeros(n)
    x[0] = 1.
    E = ell(50., x)
    # E.use_parallel_cut = False
    # options = Options()
    # options.max_it = 2000
    # options.tol = 1e-8
    xb, ell_info = cutting_plane_dc(P, E, float('inf'))
    # print(num_iters, feasible, status)
    return xb, ell_info.num_iters, ell_info.feasible


def mle_corr_poly(Y, s, n):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        s {[type]} -- [description]
        n {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    _ = np.linalg.cholesky(Y)  # test if Y is SPD.
    return corr_poly(Y, s, n, mle_oracle, mle_corr_core)


def mle_corr_bspline(Y, s, n):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        s {[type]} -- [description]
        n {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    _ = np.linalg.cholesky(Y)  # test if Y is SPD.
    return corr_bspline(Y, s, n, mle_oracle, mle_corr_core)


def lsq_corr_bspline2(Y, s, n):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        s {[type]} -- [description]
        n {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    return corr_bspline(Y, s, n, lsq_oracle, lsq_corr_core2)


def test_data():
    """[summary]
    """
    # assert Y[2,3] == approx(1.9365965488224368)
    assert s[6, 0] == approx(3.75)
    # D1 = construct_distance_matrix(s)
    # assert D1[2, 4] == approx(5.0)


def test_corr_fn():
    """[summary]
    """
    _, num_iters, feasible = lsq_corr_bspline2(Y, s, 4)
    assert feasible
    assert num_iters <= 480

    _, num_iters, feasible = lsq_corr_poly2(Y, s, 4)
    assert feasible
    assert num_iters <= 570

    _, num_iters, feasible = mle_corr_bspline(Y, s, 4)
    assert feasible
    assert num_iters <= 178

    _, num_iters, feasible = mle_corr_poly(Y, s, 4)
    assert feasible
    assert num_iters <= 234
