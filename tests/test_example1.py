# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellpy.cutting_plane import cutting_plane_dc
from ellpy.ell import ell


def my_oracle2(z):
    """[summary]

    Arguments:
        z {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    x, y = z

    # constraint 1: x + y <= 3
    fj = x + y - 3
    if fj > 0:
        return (np.array([1., 1.]), fj), False

    # constraint 2: x - y >= 1
    fj = -x + y + 1
    if fj > 0:
        return (np.array([-1., 1.]), fj), False

    return None, True


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
    return (-1. * np.array([1., 1.]), fj), t


def test_case_feasible():
    """[summary]
    """
    x0 = np.array([0., 0.])  # initial x0
    E = ell(10., x0)
    P = my_oracle
    ell_info = cutting_plane_dc(P, E, float('-inf'))
    assert ell_info.feasible

    # fmt = '{:f} {} {} {}'
    # print(fmt.format(fb, niter, feasible, status))
    # print(xb)


def test_case_infeasible():
    """[summary]
    """
    x0 = np.array([100., 100.])  # initial x0
    E = ell(10., x0)
    P = my_oracle
    ell_info = cutting_plane_dc(P, E, float('-inf'))
    assert not ell_info.feasible
    assert ell_info.status == 1  # no sol'n
