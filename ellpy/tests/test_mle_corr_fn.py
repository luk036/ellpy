# -*- coding: utf-8 -*-
import numpy as np
# from ellpy.tests.lsq_corr_oracle import lsq_corr_bspline2, lsq_corr_poly2
from ellpy.tests.mle_corr_oracle import mle_corr_bspline, mle_corr_poly
from ellpy.tests.lsq_corr_oracle import create_2d_isotropic
from scipy.interpolate import BSpline


def test_corr_fn():
    Y, s = create_2d_isotropic(10, 8, 3000)
    _, num_iters, _ = mle_corr_bspline(Y, s, 6)
    assert num_iters >= 292
    # _, num_iters, _ = mle_corr_poly(Y, s, 6)
    # assert num_iters >= 629 and num_iters <= 657
