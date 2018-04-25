# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from ..cutting_plane import cutting_plane_dc
from ..ell import ell
from .test_example2 import my_oracle2


def my_oracle(z, t):
    g, h, flag = my_oracle2(z)
    if flag == 0:
        return g, h, t

    x, y = z

    # objective: maximize x + y
    f0 = x + y
    fj = t - f0
    if fj < 0.:
        fj = 0.
        t = f0
    return -1.*np.array([1., 1.]), fj, t


# if __name__ == "__main__":
def test_example1():
    x0 = np.array([0., 0.])  # initial x0
    E = ell(10., x0)
    P = my_oracle
    xb, fb, niter, flag, status = cutting_plane_dc(P, E, -100., 200, 1e-4)
    assert flag == 1

    fmt = '{:f} {} {} {}'
    print(fmt.format(fb, niter, flag, status))
    print(xb)
    #assert niter == 115
