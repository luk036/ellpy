# -*- coding: utf-8 -*-
import cvxpy as cvx
from scipy.interpolate import BSpline
import numpy as np

#function [sp, tau2, Sig]
def lsq_corr_poly(Y, s, m):
  n = len(s)
  a = cvx.Variable(m)
  D1 = np.zeros((n,n))
  for i in range(n):
    D1[i,i] = 0.
    for j in range(i+1,n):
      h = s[j] - s[i]
      d = np.sqrt(np.dot(h,h))
      D1[i,j] = d
      D1[j,i] = d
  # D2 = np.multiply(D1, D1)
  # D3 = np.multiply(D2, D1)
  # D0 = np.ones((n,n))
  # Sig = a[3] + D1*a[2] + D2*a[1] + D3*a[0]
  Sig = a[-1]
  D = np.ones((n,n))
  for i in range(m-1):
    D = np.multiply(D, D1)
    Sig += D*a[m-2-i]
  constraints = [ Sig >> 0 ]
  prob = cvx.Problem(cvx.Minimize(cvx.norm(Sig - Y, 'fro')), constraints)
  prob.solve(solver=cvx.CVXOPT)
  if prob.status != cvx.OPTIMAL:
    raise Exception('CVXPY Error')
  return np.poly1d(np.array(a.value).flatten())
#  return prob.is_dcp()


def lsq_corr_bspline(Y,s,m):
  k = 2
  h = s[-1] - s[0]
  d = np.sqrt(np.dot(h,h))
  t = np.linspace(0, d*1.2, m+k+1)
  spls = []
  for i in range(m):
    coeff = np.zeros(m)
    coeff[i] = 1
    spls += [ BSpline(t, coeff, k) ]

  n = len(s)
  c = cvx.Variable(m)
  D = np.zeros((n,n))
  for i in range(n):
    for j in range(i+1,n):
      h = s[j] - s[i]
      d = np.sqrt(np.dot(h,h))
      D[i,j] = d
      D[j,i] = d

  # Sig = spls[0](D)*c[0] + spls[1](D)*c[1] + spls[2](D)*c[2] + spls[3](D)*c[3]
  Sig = np.zeros((n,n))
  for i in range(m):
    Sig += spls[i](D)*c[i]
  # constraints += [ Sig[i,j] == cvx.sum_entries(cvx.mul_elemwise(splval, c))]
  constraints = [ Sig >> 0 ]
  for i in range(m-1):
    constraints += [ c[i] >= c[i+1] ]
  constraints += [ c[-1] >= 0.0 ]

  prob = cvx.Problem(cvx.Minimize(cvx.norm(Sig - Y, 'fro')), constraints)
  prob.solve(solver=cvx.CVXOPT)
  if prob.status != cvx.OPTIMAL:
    raise Exception('CVXPY Error')
  return BSpline(t, np.array(c.value).flatten(), k)
