# -*- coding: utf-8 -*-
from __future__ import print_function
 
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.pylab as lab
import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline
from corr_fn import *

# a fake dataset to make the bumps with
nx = 10   # number of points
ny = 8
n = nx*ny
s_begin = [1., 1.]
s_end = [10., 8]
sdkern = 8  # width of kernel
var = 2.0     # standard derivation
tau = 0.00001    # standard derivation of white noise
N = 100  # number of samples

# create sites s
sx = np.linspace(s_begin[0], s_end[0], nx)
sy = np.linspace(s_begin[1], s_end[1], ny)
xx, yy = np.meshgrid(sx, sy)
s = zip(xx.flatten(), yy.flatten())

Sig = np.ones((n,n))
for i in range(n):
  for j in range(i,n):
    d = np.array(s[j]) - np.array(s[i])
    Sig[i,j] = np.exp(-sdkern * np.dot(d,d) )
    Sig[j,i] = Sig[i,j]

A = linalg.sqrtm(Sig)
Ys = np.zeros((n,N))
# ym = np.random.randn(n)
for k in range(N):
    x = var * np.random.randn(n)
    y = A.dot(x) + tau*np.random.randn(n)
    Ys[:,k] = y

Y = np.cov(Ys, bias=True)

plt.subplot(2,2,1)
lab.contourf(xx,yy,np.reshape( Ys[:,1],(ny, nx) ), cmap='Greens')
plt.subplot(2,2,2)
lab.contourf(xx,yy,np.reshape( Ys[:,3],(ny, nx) ), cmap='Greens')
plt.subplot(2,2,3)
lab.contourf(xx,yy,np.reshape( Ys[:,5],(ny, nx) ), cmap='Greens')
plt.subplot(2,2,4)
lab.contourf(xx,yy,np.reshape( Ys[:,7],(ny, nx) ), cmap='Greens')
plt.show()


pol = lsq_corr_poly(Y, s)
spl = lsq_corr_bspline(Y, s)
h = np.array(s[-1]) - np.array(s[0])
d = np.sqrt(np.dot(h,h))
xs = np.linspace(0, d, 100)
plt.plot(xs, np.polyval(pol, xs), 'g', label='Polynomial')
plt.plot(xs, spl(xs), 'r', label='BSpline')
plt.legend(loc='best')
plt.show()
