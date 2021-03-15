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
    x, y = z

    # constraint 1: exp(x) <= y
    tmp = math.exp(x)
    fj = tmp - y
    if fj > 0:
        return (np.array([tmp, -1.]), fj), None

    # constraint 2: y > 0
    if y <= 0.:
        return (np.array([0., -1.]), -y), None

    # constraint 3: x > 0
    if x <= 0.:
        return (np.array([-1., 0.]), -x), None

    # objective: minimize -sqrt(x) / y
    tmp2 = math.sqrt(x)
    fj = -tmp2 - t * y
    if fj < 0.:  # feasible
        t = -tmp2 / y
        return (np.array([-0.5/tmp2, -t]), 0), t

    return (np.array([-0.5/tmp2, -t]), fj), None


def test_case_feasible():
    """[summary]
    """
    x0 = np.array([1., 1.])  # initial x0
    E = ell(10., x0)
    P = my_quasicvx_oracle
    xb, fb, ell_info = cutting_plane_dc(P, E, 0.)
    assert ell_info.feasible
    assert fb == approx(-0.4288673396685956)
    assert xb[0] == approx(0.501315956)
    assert xb[1] == approx(1.650886769)


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
    x0 = np.array([1., 1.])  # initial x0
    E = ell(10., x0)
    P = my_quasicvx_oracle
    _, _, ell_info = cutting_plane_dc(P, E, -100)  # wrong initial best-so-far
    assert not ell_info.feasible
