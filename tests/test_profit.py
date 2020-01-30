# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellpy.cutting_plane import cutting_plane_dc, cutting_plane_q
from ellpy.ell import ell
from ellpy.oracles.profit_oracle import profit_oracle, profit_q_oracle, profit_rb_oracle

p, A, k = 20., 40., 30.5
params = p, A, k
alpha, beta = 0.1, 0.4
v1, v2 = 10., 35.
a = np.array([alpha, beta])
v = np.array([v1, v2])
r = np.array([100., 100.])  # initial ellipsoid (sphere)


def test_profit():
    E = ell(r, np.array([0., 0.]))
    P = profit_oracle(params, a, v)
    _, _, ell_info = cutting_plane_dc(P, E, 0.)
    assert ell_info.feasible
    assert ell_info.num_iters == 37


def test_profit_rb():
    e1 = 0.003
    e2 = 0.007
    e3 = e4 = e5 = 1.
    E = ell(r, np.array([0., 0.]))
    P = profit_rb_oracle(params, a, v, (e1, e2, e3, e4, e5))
    _, _, ell_info = cutting_plane_dc(P, E, 0.)
    assert ell_info.feasible
    assert ell_info.num_iters == 42


def test_profit_q():
    E = ell(r, np.array([0., 0.]))
    P = profit_q_oracle(params, a, v)
    _, _, ell_info = cutting_plane_q(P, E, 0.)
    assert ell_info.feasible
    assert ell_info.num_iters == 28
    return ell_info.num_iters
