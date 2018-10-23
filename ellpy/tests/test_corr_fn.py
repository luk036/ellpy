# -*- coding: utf-8 -*-
import numpy as np
from ellpy.tests.lsq_corr_oracle import lsq_corr_bspline2, lsq_corr_poly2
from scipy.interpolate import BSpline

# a fake dataset to make the bumps with
nx = 10   # number of points
ny = 8
n = nx*ny
s_end = [10., 8.]
sdkern = 0.1  # width of kernel
var = 2.     # standard derivation
tau = 0.00001    # standard derivation of white noise
N = 200  # number of samples
np.random.seed(5)


# create sites s
sx = np.linspace(0, s_end[0], nx)
sy = np.linspace(0, s_end[1], ny)
xx, yy = np.meshgrid(sx, sy)
s = np.vstack([xx.flatten(), yy.flatten()]).T

Sig = np.ones((n, n))
for i in range(n):
    for j in range(i, n):
        d = np.array(s[j]) - np.array(s[i])
        Sig[i, j] = np.exp(-sdkern * np.dot(d, d))
        Sig[j, i] = Sig[i, j]

A = np.linalg.cholesky(Sig)
Ys = np.zeros((n, N))

for k in range(N):
    x = var * np.random.randn(n)
    y = A.dot(x) + tau*np.random.randn(n)
    Ys[:, k] = y

Y = np.cov(Ys, bias=True)


def test_corr_fn():
    _, num_iters, _ = lsq_corr_bspline2(Y, s, 4)
    assert num_iters == 98
    _, num_iters, _ = lsq_corr_poly2(Y, s, 4)
    assert num_iters >= 631 and num_iters <= 657
