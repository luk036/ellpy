# -*- coding: utf-8 -*-
#import cvxpy as cvx
from scipy.interpolate import BSpline
import numpy as np
from cutting_plane import cutting_plane_feas, bsearch
from ell import ell
from problem import Problem
# from oracles.qmi_oracle import qmi_oracle
from oracles.lmi_oracle import lmi_oracle


# class fitting_oracle:
#     def __init__(self, F, F0, x, max_it=1000, tol=1e-8):
#         n = F0.shape[0]
#         self.P = qmi_oracle(F, F0, np.eye(n))
#         self.x_best = x
#         self.max_it = max_it
#         self.tol = tol

#     def __call__(self, t):
#         x = self.x_best.copy()
#         E = ell(100, x)
#         n = self.P.F0.shape[0]
#         self.P.B = np.eye(n) * t  # update B
#         x, _, flag, _ = cutting_plane_feas(self.P, E, self.max_it, self.tol)
#         if flag == 1:
#             self.x_best = x.copy()
#             return True
#         return False


class fitting_oracle:
    def __init__(self, Sig, Y, x, max_it=1000, tol=1e-8):
        n = Y.shape[0]
        nx = len(x)
        F = []
        for k in range(nx):
            Fk = np.vstack([np.hstack([np.zeros((n,n)), Sig[k]]), np.hstack([Sig[k].T, np.zeros((n,n))])])
            F += [ Fk ]
        self.F0 = np.vstack([np.hstack([np.eye(n), Y]), np.hstack([Y.T, np.eye(n)])])
        self.P = lmi_oracle(F, self.F0)
        self.x_best = x
        self.max_it = max_it
        self.tol = tol

    def __call__(self, t):
        x = self.x_best.copy()
        E = ell(100, x)
        n = self.P.F0.shape[0]
        self.P.B = np.eye(n) * t - self.F0 # <- update B
        x, _, flag, _ = cutting_plane_feas(self.P, E, self.max_it, self.tol)
        if flag == 1:
            self.x_best = x.copy()
            return True
        return False


def lsq_corr_poly(Y, s, m):
    n = len(s)
    a = np.zeros(m)
    D1 = np.zeros((n, n))
    for i in range(n):
        D1[i, i] = 0.
        for j in range(i + 1, n):
            h = s[j] - s[i]
            d = np.sqrt(np.dot(h, h))
            D1[i, j] = d
            D1[j, i] = d
    # D2 = np.multiply(D1, D1)
    # D3 = np.multiply(D2, D1)
    # D0 = np.ones((n,n))
    # Sig = a[3] + D1*a[2] + D2*a[1] + D3*a[0]
    D = np.ones((n, n))
    Sig = [D]
    for i in range(m - 1):
        D = np.multiply(D, D1)
        Sig += [D]
    Sig.reverse()

    # constraints = [Sig >> 0]
    # prob = cvx.Problem(cvx.Minimize(cvx.norm(Sig - Y, 'fro')), constraints)
    # prob.solve(solver=cvx.CVXOPT)
    # prob.solve()
    ## E = ell(100., a)
    P = fitting_oracle(Sig, Y, a)
    t = np.linalg.norm(Y, 'fro')
    u, niter, flag = bsearch(P, [0., 2.*t*t])
    a = P.x_best
    return np.poly1d(a)
#  return prob.is_dcp()


# def lsq_corr_bspline(Y, s, m):
#     k = 2
#     h = s[-1] - s[0]
#     d = np.sqrt(np.dot(h, h))
#     t = np.linspace(0, d * 1.2, m + k + 1)
#     spls = []
#     for i in range(m):
#         coeff = np.zeros(m)
#         coeff[i] = 1
#         spls += [BSpline(t, coeff, k)]

#     n = len(s)
#     c = cvx.Variable(m)
#     D = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i + 1, n):
#             h = s[j] - s[i]
#             d = np.sqrt(np.dot(h, h))
#             D[i, j] = d
#             D[j, i] = d

#     # Sig = spls[0](D)*c[0] + spls[1](D)*c[1] + spls[2](D)*c[2] +
#     # spls[3](D)*c[3]
#     Sig = np.zeros((n, n))
#     for i in range(m):
#         Sig += spls[i](D) * c[i]
#     # constraints += [ Sig[i,j] == cvx.sum_entries(cvx.mul_elemwise(splval,
#     # c))]
#     constraints = [Sig >> 0]
#     for i in range(m - 1):
#         constraints += [c[i] >= c[i + 1]]
#     constraints += [c[-1] >= 0.]

#     prob = cvx.Problem(cvx.Minimize(cvx.norm(Sig - Y, 'fro')), constraints)
#     prob.solve(solver=cvx.CVXOPT)
#     # prob.solve()
#     if prob.status != cvx.OPTIMAL:
#         raise Exception('CVXPY Error')
#     return BSpline(t, np.array(c.value).flatten(), k)
