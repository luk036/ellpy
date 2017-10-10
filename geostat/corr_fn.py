# -*- coding: utf-8 -*-
import cvxpy as cvx
from scipy.interpolate import BSpline
import numpy as np

#function [sp, tau2, Sig]
def lsq_corr_poly(Y, s):
  n = len(s)
  a = cvx.Variable(4)
  Sig = cvx.Semidef(n)
  constraints = []
  for i in range(n):
    constraints += [ Sig[i,i] == a[3] ]
    for j in range(i+1,n):
      d = s[j] - s[i]
      constraints += [ Sig[i,j] == a[3] + d * (a[2] + d * (a[1] + d * a[0] ) ) ]

  prob = cvx.Problem(cvx.Minimize(cvx.norm(Sig - Y, 'fro')), constraints)
  prob.solve(solver=cvx.CVXOPT)
  if prob.status != cvx.OPTIMAL:
    raise Exception('CVXPY Error')
  return np.poly1d(np.array(a.value).flatten())
#  return prob.is_dcp()


def lsq_corr_bspline(Y,s):
  k = 2
  m = 4
  t = np.linspace(0, (s[-1] - s[0])*1.2, m+k+1)
  spls = []
  for i in range(m):
    coeff = np.zeros(m)
    coeff[i] = 1
    spls += [ BSpline(t, coeff, k) ]

  n = len(s)
  Sig = cvx.Semidef(n)
  constraints = []
  c = cvx.Variable(4)
  for i in range(n):
    for j in range(i,n):
      d = s[j] - s[i]
      splval = np.zeros(m)
      for l in range(m):
        splval[l] = spls[l](d)
      constraints += [ Sig[i,j] == cvx.sum_entries(cvx.mul_elemwise(splval, c))]
  for i in range(m-1):
    constraints += [ c[i] >= c[i+1] ]
  constraints += [ c[-1] >= 0.0 ]

  prob = cvx.Problem(cvx.Minimize(cvx.norm(Sig - Y, 'fro')), constraints)
  prob.solve(solver=cvx.CVXOPT)
  if prob.status != cvx.OPTIMAL:
    raise Exception('CVXPY Error')
  return BSpline(t, np.array(c.value).flatten(), k)
