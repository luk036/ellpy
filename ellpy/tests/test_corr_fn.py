# -*- coding: utf-8 -*-
from pytest import approx
from ellpy.tests.corr_oracle import create_2d_isotropic, construct_distance_matrix
from ellpy.tests.lsq_corr_oracle import lsq_corr_bspline2, lsq_corr_poly2
from ellpy.tests.mle_corr_oracle import mle_corr_bspline, mle_corr_poly

Y, s = create_2d_isotropic(5, 4, 3000)


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
    assert num_iters <= 162

    _, num_iters, feasible = mle_corr_poly(Y, s, 4)
    assert feasible
    assert num_iters <= 220
