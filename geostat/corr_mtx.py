# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from cvxpy import *

#function [sp, tau2, Sig]
def lsq_corr_mtx(Y, s):
  n = len(s)
  Sig = Semidef(n)
  prob = Problem(Minimize(norm(Sig - Y, 'fro')))
  prob.solve()
  if prob.status != OPTIMAL:
    raise Exception('CVXPY Error')
  return Sig.value

def mle_corr_mtx(Y, s):
  n = len(s)
  S = Semidef(n)
  prob = Problem(Maximize(log_det(S) - trace(S*Y)))
  prob.solve()
  if prob.status != OPTIMAL:
    raise Exception('CVXPY Error')
  return linalg.inv(S.value)