# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from ellpy.cutting_plane import cutting_plane_dc
from ellpy.ell import ell
from .test_example2 import my_oracle2


def my_oracle(z, t):
    """[summary]

    Arguments:
        z {[type]} -- [description]
        t {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    cut, feasible = my_oracle2(z)
    if not feasible:
        return cut, t
    x, y = z
    # objective: maximize x + y
    f0 = x + y
    fj = t - f0
    if fj < 0:
        fj = 0.
        t = f0
    return (-1.*np.array([1., 1.]), fj), t


def test_example1():
    """[summary]
    """
    x0 = np.array([0., 0.])  # initial x0
    E = ell(10., x0)
    P = my_oracle
    xb, fb, niter, feasible, status = cutting_plane_dc(P, E, -100.)
    assert feasible

    fmt = '{:f} {} {} {}'
    print(fmt.format(fb, niter, feasible, status))
    print(xb)
