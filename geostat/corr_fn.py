# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from cvxpy import *

#function [sp, tau2, Sig]
def lsq_corr_fn(Y, s):
  n = len(s)
  a = Variable(3)
  Sig = Semidef(n)
  constraints = []
  for i in range(n):
    constraints += [ Sig[i,i] == a[0] ]
    for j in range(i+1,n):
      d = s[j] - s[i]
      constraints += [ Sig[i,j] == a[0] + d * (a[1] + d * a[2]) ]

  prob = Problem(Minimize(norm(Sig - Y, 'fro')), constraints)
  prob.solve(solver=CVXOPT)
  if prob.status != OPTIMAL:
    raise Exception('CVXPY Error')
  return a.value
#  return prob.is_dcp()