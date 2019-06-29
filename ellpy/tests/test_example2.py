# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from ellpy.cutting_plane import cutting_plane_feas
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


def test_case_feasible():
    """[summary]
    """
    x0 = np.array([0., 0.])  # initial guess
    E = ell(10., x0)
    P = my_oracle2
    ell_info = cutting_plane_feas(P, E)
    assert ell_info.feasible
    assert ell_info.status == 0
    print(ell_info.num_iters, ell_info.status)
    print(ell_info.val)


def test_case_infeasible():
    """[summary]
    """
    x0 = np.array([100., 100.])  # wrong initial guess
    E = ell(10., x0)
    P = my_oracle2
    ell_info = cutting_plane_feas(P, E)
    assert ell_info.status == 1  # no sol'n
    assert ell_info.num_iters == 1  # small
    assert not ell_info.feasible
