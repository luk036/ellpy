# -*- coding: utf-8 -*-
import numpy as np
from ellpy.tests.lsq_corr_oracle import lsq_corr_bspline2, lsq_corr_poly2
from ellpy.tests.lsq_corr_oracle import create_2d_isotropic
from scipy.interpolate import BSpline


def test_corr_fn():
    Y, s = create_2d_isotropic(5, 4)
    _, num_iters, feasible = lsq_corr_bspline2(Y, s, 4)
    assert feasible
    assert num_iters <= 115
    _, num_iters, feasible = lsq_corr_poly2(Y, s, 4)
    assert feasible
    assert num_iters <= 570
