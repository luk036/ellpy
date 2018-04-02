# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from ..oracles.profit_oracle import *
from ..cutting_plane import *
from ..ell import *


def test_profit():
    p, A, k = 20. , 40. , 30.5
    alpha, beta = 0.1, 0.4
    v1, v2 = 10. , 35.
    y0 = np.array([0. , 0.])  # initial x0
    r = np.array([100. , 100.])  # initial ellipsoid (sphere)
    fmt = '{:f} {} {} {}'

    E = ell(r, y0)
    P = profit_oracle(p, A, alpha, beta, v1, v2, k)
    yb1, fb, iter, flag, status = cutting_plane_dc(P, E, 0. , 200, 1e-4)
    print(fmt.format(fb, iter, flag, status))
    assert flag == 1
    assert iter == 37

    ui = 1.0
    e1 = 0.003
    e2 = 0.007
    e3 = 1.0

    E = ell(r, y0)
    P = profit_rb_oracle(p, A, alpha, beta, v1, v2, k, ui, e1, e2, e3)
    yb1, fb, iter, flag, status = cutting_plane_dc(P, E, 0. , 200, 1e-4)
    print(fmt.format(fb, iter, flag, status))
    assert flag == 1
    assert iter == 42

    E = ell(r, y0)
    P = profit_q_oracle(p, A, alpha, beta, v1, v2, k)
    yb1, fb, iter, flag, status = cutting_plane_q(P, E, 0. , 200, 1e-4)
    print(fmt.format(fb, iter, flag, status))
    assert flag == 1
    assert iter == 28
