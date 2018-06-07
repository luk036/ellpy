# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from ellpy.cutting_plane import cutting_plane_feas
from ellpy.ell import ell


def my_oracle2(z):
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


def test_example2():
    x0 = np.array([0., 0.])  # initial x0
    E = ell(10., x0)
    P = my_oracle2
    xb, niter, feasible, status = cutting_plane_feas(P, E)
    assert feasible
    print(niter, feasible, status)
    print(xb)
