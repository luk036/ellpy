# -*- coding: utf-8 -*-
from __future__ import print_function

import math

from pytest import approx

import numpy as np

from ellpy.cutting_plane import CUTStatus, cutting_plane_dc
from ellpy.ell import ell


def my_quasicvx_oracle(z, t: float):
    """[summary]

    Arguments:
        z ([type]): [description]
        t (float): the best-so-far optimal value

    Returns:
        [type]: [description]
    """
    sqrtx, ly = z

    # constraint 1: exp(x) <= y, or sqrtx**2 <= ly
    fj = sqrtx * sqrtx - ly
    if fj > 0:
        return (np.array([2*sqrtx, -1.]), fj), None

    # constraint 3: x > 0
    # if x <= 0.:
    #     return (np.array([-1., 0.]), -x), None

    # objective: minimize -sqrt(x) / y
    tmp2 = math.exp(ly)
    tmp3 = t * tmp2
    fj = -sqrtx + tmp3
    if fj < 0.:  # feasible
        t = sqrtx / tmp2
        return (np.array([-1., sqrtx]), 0), t

    return (np.array([-1., tmp3]), fj), None


def test_case_feasible():
    """[summary]
    """
    x0 = np.array([0., 0.])  # initial x0
    E = ell(10., x0)
    P = my_quasicvx_oracle
    xb, fb, ell_info = cutting_plane_dc(P, E, 0.)
    assert ell_info.feasible
    assert fb == approx(0.4288673396685956)
    assert xb[0]*xb[0] == approx(0.5046900657538383)
    assert math.exp(xb[1]) == approx(1.6564805414665902)


def test_case_infeasible1():
    """[summary]
    """
    x0 = np.array([100., 100.])  # wrong initial guess,
    E = ell(10., x0)  # or ellipsoid is too small
    P = my_quasicvx_oracle
    _, _, ell_info = cutting_plane_dc(P, E, 0.)
    assert not ell_info.feasible
    assert ell_info.status == CUTStatus.nosoln  # no sol'n


def test_case_infeasible2():
    """[summary]
    """
    x0 = np.array([0., 0.])  # initial x0
    E = ell(10., x0)
    P = my_quasicvx_oracle
    _, _, ell_info = cutting_plane_dc(P, E, 100)  # wrong initial best-so-far
    assert not ell_info.feasible
