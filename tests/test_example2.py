# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from ..cutting_plane import cutting_plane_feas
from ..ell import ell


def my_oracle2(z):
    x, y = z

    # constraint 1: x + y <= 3
    fj = x + y - 3
    if fj > 0:
        return (np.array([1., 1.]), fj), 0

    # constraint 2: x - y >= 1
    fj = -x + y + 1
    if fj > 0:
        return (np.array([-1., 1.]), fj), 0

    return None, 1


# if __name__ == "__main__":
def test_example2():
    x0 = np.array([0., 0.])  # initial x0
    E = ell(10., x0)
    P = my_oracle2
    xb, niter, flag, status = cutting_plane_feas(P, E, 200, 1e-4)
    assert flag == 1
    print(niter, flag, status)
    print(xb)
