# -*- coding: utf-8 -*-
from __future__ import print_function
 
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline
from corr_fn import *

# a fake dataset to make the bumps with
n = 20   # number of points
s_end = 10.
sdkern = 0.16  # width of kernel
var = 2.0     # standard derivation
tau = 0.00001    # standard derivation of white noise
N = 300  # number of samples

# create sites s
s = np.linspace(0, s_end, n)

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

pol = lsq_corr_poly(Y, s, 5)
spl = lsq_corr_bspline(Y, s, 5)

xs = np.linspace(0, s_end, 100)
plt.plot(xs, np.polyval(pol, xs), 'g', label='Polynomial')
plt.plot(xs, spl(xs), 'r', label='BSpline')
plt.legend(loc='best')
plt.show()
