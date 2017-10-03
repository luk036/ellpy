# -*- coding: utf-8 -*-
from __future__ import print_function
 
from pprint import pprint
from scipy import linalg, sparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from corr_fn import *

# a fake dataset to make the bumps with
n = 4   # number of points
s_begin = 1.
s_end = 10.
sdkern = 0.1  # width of kernel
var = 2.0     # standard derivation
tau = 0.00001    # standard derivation of white noise
N = 60  # number of samples

# create sites s
dist = s_end - s_begin
s = np.linspace(s_begin, s_end, n)

Sig = np.ones((n,n))
for i in range(n):
  for j in range(i+1,n):
    d = s[j] - s[i]
    Sig[i,j] = np.exp(-sdkern * (d*d) )
    Sig[j,i] = Sig[i,j]

##[U,Lambda,V] = svd(Sig)
##A = U*sqrt(Lambda)
A = linalg.sqrtm(Sig)
Ys = np.zeros((n,N))
# ym = np.random.randn(n)
for k in range(N):
    x = var * np.random.randn(n)
    y = A.dot(x) + tau*np.random.randn(n)
    Ys[:,k] = y

Y = np.cov(Ys, bias=True)

print(Y)

a_lsq = lsq_corr_poly(Y, s)
#Sig_mle = mle_corr_mtx(Y, s)

print(a_lsq)
#print(Sig_mle)
# x = np.linspace(-3, 3, 50)
# y = np.exp(-x**2) + 0.1 * np.random.randn(50)
# plt.plot(x, y, 'ro', ms=5)
# spl = UnivariateSpline(x, y)
# xs = np.linspace(-3, 3, 1000)
# plt.plot(xs, spl(xs), 'g', lw=3)
# spl.set_smoothing_factor(0.5)
# plt.plot(xs, spl(xs), 'b', lw=3)
# plt.show()