#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from corr_fn_cvx import lsq_corr_bspline

from ellpy.oracles.corr_oracle import create_2d_isotropic
# from scipy.interpolate import BSpline
# from ellpy.tests.lsq_corr_oracle import lsq_corr_bspline2, lsq_corr_poly2
from ellpy.tests.mle_corr_oracle import mle_corr_poly

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # import matplotlib.pylab as lab
    Y, s = create_2d_isotropic(6, 4, 3000)
    print('start ell...')
    # spl, num_iters, _ = lsq_corr_bspline2(Y, s, 5)
    pol, num_iters, _ = mle_corr_poly(Y, s, 4)
    print(pol)
    print(num_iters)
    print('start cvx...')
    splcvx = lsq_corr_bspline(Y, s, 4)
    # pol, num_iters, _ = lsq_corr_poly2(Y, s, 5)
    # print(num_iters)
    # polcvx  = lsq_corr_poly(Y, s, 5)

    h = s[-1] - s[0]
    d = np.sqrt(h @ h)
    xs = np.linspace(0, d, 100)
    # plt.plot(xs, spl(xs), 'g', label='BSpline')
    plt.plot(xs, splcvx(xs), 'b', label='BSpline CVX')
    plt.plot(xs, np.polyval(pol, xs), 'r', label='Polynomial')
    # plt.plot(xs, np.polyval(polcvx, xs), 'r', label='Polynomial CVX')
    plt.legend(loc='best')
    plt.show()
