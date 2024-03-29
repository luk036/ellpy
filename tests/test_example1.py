# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellpy.cutting_plane import CUTStatus, cutting_plane_dc
from ellpy.ell import ell


def my_oracle2(z):
    """[summary]

    Arguments:
        z ([type]): [description]

    Returns:
        [type]: [description]
    """
    x, y = z

    # constraint 1: x + y <= 3
    fj = x + y - 3
    if fj > 0:
        return np.array([1.0, 1.0]), fj

    # constraint 2: x - y >= 1
    fj = -x + y + 1
    if fj > 0:
        return np.array([-1.0, 1.0]), fj


def my_oracle(z, t: float):
    """[summary]

    Arguments:
        z ([type]): [description]
        t (float): the best-so-far optimal value

    Returns:
        [type]: [description]
    """
    cut = my_oracle2(z)
    if cut:
        return cut, t
    x, y = z
    # objective: maximize x + y
    f0 = x + y
    fj = t - f0
    if fj < 0:
        fj = 0.0
        t = f0
    return (-1.0 * np.array([1.0, 1.0]), fj), t


def test_case_feasible():
    """[summary]"""
    x0 = np.array([0.0, 0.0])  # initial x0
    E = ell(10.0, x0)
    P = my_oracle
    _, _, ell_info = cutting_plane_dc(P, E, float("-inf"))
    assert ell_info.feasible

    # fmt = '{:f} {} {} {}'
    # print(fmt.format(fb, niter, feasible, status))
    # print(xb)


def test_case_infeasible1():
    """[summary]"""
    x0 = np.array([100.0, 100.0])  # wrong initial guess,
    E = ell(10.0, x0)  # or ellipsoid is too small
    P = my_oracle
    _, _, ell_info = cutting_plane_dc(P, E, float("-inf"))
    assert not ell_info.feasible
    assert ell_info.status == CUTStatus.nosoln  # no sol'n


def test_case_infeasible2():
    """[summary]"""
    x0 = np.array([0.0, 0.0])  # initial x0
    E = ell(10.0, x0)
    P = my_oracle
    _, _, ell_info = cutting_plane_dc(P, E, 100)  # wrong initial best-so-far
    assert not ell_info.feasible
