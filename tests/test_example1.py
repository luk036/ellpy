# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from ..cutting_plane import cutting_plane_dc
from ..ell import ell


def my_oracle(z, t):
    x, y = z

    # constraint 1: x + y <= 3
    fj = x + y - 3
    if fj > 0:
        return np.array([1., 1.]), fj, t

    # constraint 2: x - y >= 1
    fj = -x + y + 1
    if fj > 0:
        return np.array([-1., 1.]), fj, t

    # objective: maximize x + y
    f0 = x + y
    fj = t - f0
    if fj < 0.:
        fj = 0.
        t = f0
    return -1.*np.array([1., 1.]), fj, t


#if __name__ == "__main__":
def test_example1():
    x0 = np.array([0., 0.])  # initial x0
    E = ell(10., x0)
    P = my_oracle
    xb, fb, niter, flag, status = cutting_plane_dc(P, E, -100., 200, 1e-4)

    fmt = '{:f} {} {} {}'
    print(fmt.format(fb, niter, flag, status))
    print(xb)
    assert flag == 1
    #assert niter == 115
