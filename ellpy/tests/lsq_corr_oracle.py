# -*- coding: utf-8 -*-
import numpy as np
from ellpy.oracles.qmi_oracle import qmi_oracle
from ellpy.oracles.lmi0_oracle import lmi0_oracle
from ellpy.cutting_plane import cutting_plane_dc
from ellpy.ell import ell
from ellpy.tests.corr_oracle import corr_poly, corr_bspline


class lsq_oracle:
    """[summary]

    Returns:
        [type] -- [description]
    """

    def __init__(self, F, F0):
        """[summary]

        Arguments:
            F {[type]} -- [description]
            F0 {[type]} -- [description]
        """
        self.qmi = qmi_oracle(F, F0)
        self.lmi0 = lmi0_oracle(F)

    def __call__(self, x, t):
        """[summary]

        Arguments:
            x {[type]} -- [description]
            t {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        n = len(x)
        g = np.zeros(n)

        cut, feasible = self.lmi0(x[:-1])
        if not feasible:
            g1, fj = cut
            g[:-1] = g1
            g[-1] = 0.
            return (g, fj), t

        self.qmi.update(x[-1])
        cut, feasible = self.qmi(x[:-1])
        if not feasible:
            g1, fj = cut
            g[:-1] = g1
            self.qmi.Q.witness()
            # n = self.qmi.Q.p[-1] + 1
            p = self.qmi.Q.p
            v = self.qmi.Q.v[p]
            g[-1] = -v.dot(v)
            return (g, fj), t

        g[-1] = 1
        tc = x[-1]
        fj = tc - t
        if fj > 0:
            return (g, fj), t
        return (g, 0.), tc


def lsq_corr_core2(Y, m, P):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        m {[type]} -- [description]
        P {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    normY = np.linalg.norm(Y, 'fro')
    normY2 = 32*normY*normY
    val = 256*np.ones(m + 1)
    val[-1] = normY2*normY2
    x = np.zeros(m + 1)  # cannot all zeros
    x[0] = 1.
    x[-1] = normY2/2
    E = ell(val, x)
    ell_info = cutting_plane_dc(P, E, float('inf'))
    return ell_info.val[:-1], ell_info.num_iters, ell_info.feasible


def lsq_corr_poly2(Y, s, m):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        s {[type]} -- [description]
        m {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    return corr_poly(Y, s, m, lsq_oracle, lsq_corr_core2)


def lsq_corr_bspline2(Y, s, m):
    """[summary]

    Arguments:
        Y {[type]} -- [description]
        s {[type]} -- [description]
        m {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    return corr_bspline(Y, s, m, lsq_oracle, lsq_corr_core2)


# class bsp_oracle:
#     def __init__(self, F, F0):
#         self.qmi = qmi_oracle(F, F0)

#     def update(self, t):
#         self.qmi.update(t)

#     def __call__(self, x):
#         cut, feasible = mono_oracle(x)
#         if not feasible:
#             return cut, False
#         return self.qmi(x)


# def lsq_corr_poly(Y, s, m):
#     n = len(s)
#     D1 = construct_distance_matrix(s)
#     D = np.ones((n, n))
#     Sig = [D]
#     for _ in range(m - 1):
#         D = np.multiply(D, D1)
#         Sig += [D]
#     # Sig.reverse()
#     Q = qmi_oracle(Sig, Y)

#     niter, a = lsq_corr_core(m, Y, Q)
#     assert niter == 40
#     pa = np.ascontiguousarray(a[::-1])
#     return np.poly1d(pa)


# def lsq_corr_bspline(Y, s, m):
#     Sig, t, k = generate_bspline_info(s, m)
#     Q = bsp_oracle(Sig, Y)
#     niter, c = lsq_corr_core(m, Y, Q)
#     assert niter == 40
#     return BSpline(t, c, k)


# def lsq_corr_core(m, Y, Q):
#     c = np.ones(m)  # cannot all zeros
#     normY = np.linalg.norm(Y, 'fro')
#     E = ell(64., c)
#     P = bsearch_adaptor(Q, E)
#     _, niter, feasible = bsearch(P, [0., normY*normY])
#     print(niter, feasible)
#     assert feasible
#     c = P.x_best
#     return niter, c
