# -*- coding: utf-8 -*-
from typing import Union

import cvxpy as cvx
import numpy as np
from scipy.interpolate import BSpline

from ellpy.oracles.corr_oracle import (
    construct_distance_matrix,
    create_2d_isotropic,
    create_2d_sites,
)

Arr = Union[np.ndarray]


def lsq_corr_poly(Y: Arr, s: Arr, n: int):
    """[summary]

    Arguments:
        Y (Arr): [description]
        s (Arr): [description]
        n (int): [description]

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """
    N = len(s)
    a = cvx.Variable(n)
    D1 = construct_distance_matrix(s)
    # D2 = np.multiply(D1, D1)
    # D3 = np.multiply(D2, D1)
    # D0 = np.ones((N,N))
    # Sig = a[3] + D1*a[2] + D2*a[1] + D3*a[0]
    Sig = a[-1]
    D = np.ones((N, N))
    for i in range(n - 1):
        D = np.multiply(D, D1)
        Sig += D * a[n - 2 - i]
    constraints = [Sig >> 0]
    prob = cvx.Problem(cvx.Minimize(cvx.norm(Sig - Y, "fro")), constraints)
    prob.solve(solver=cvx.CVXOPT)
    # prob.solve()
    if prob.status != cvx.OPTIMAL:
        raise Exception("CVXPY Error")
    return np.poly1d(np.array(a.value).flatten())


#  return prob.is_dcp()


def lsq_corr_bspline(Y: Arr, s: Arr, n: int):
    """[summary]

    Arguments:
        Y (Arr): [description]
        s (Arr): [description]
        n (int): [description]

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """
    k = 2
    h = s[-1] - s[0]
    d = np.sqrt(h @ h)
    t = np.linspace(0, d * 1.2, n + k + 1)
    spls = []
    for i in range(n):
        coeff = np.zeros(n)
        coeff[i] = 1
        spls += [BSpline(t, coeff, k)]

    N = len(s)
    c = cvx.Variable(n)
    D = construct_distance_matrix(s)

    # Sig = spls[0](D)*c[0] + spls[1](D)*c[1] + spls[2](D)*c[2] +
    # spls[3](D)*c[3]
    Sig = np.zeros((N, N))
    for i in range(n):
        Sig += spls[i](D) * c[i]
    # constraints += [ Sig[i,j] == cvx.sum_entries(cvx.mul_elemwise(splval,
    # c))]
    constraints = [Sig >> 0]
    for i in range(n - 1):
        constraints += [c[i] >= c[i + 1]]
    constraints += [c[-1] >= 0.0]

    prob = cvx.Problem(cvx.Minimize(cvx.norm(Sig - Y, "fro")), constraints)
    prob.solve(solver=cvx.CVXOPT)
    # prob.solve()
    if prob.status != cvx.OPTIMAL:
        raise Exception("CVXPY Error")
    return BSpline(t, np.array(c.value).flatten(), k)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # import matplotlib.pylab as lab
    s = create_2d_sites(10, 8)
    Y = create_2d_isotropic(s, 1000)
    # print('start ell...')
    # spl, num_iters, _ = lsq_corr_bspline(Y, s, 5)
    # pol, num_iters, _ = lsq_corr_poly(Y, s, 5)
    # # pol, num_iters, _ = mle_corr_poly(Y, s, 4)
    # print(pol)
    # print(num_iters)
    print("start cvx...")
    splcvx = lsq_corr_bspline(Y, s, 5)
    polcvx = lsq_corr_poly(Y, s, 5)

    # h = s[-1] - s[0]
    d = np.sqrt(10**2 + 8**2)
    xs = np.linspace(0, d, 100)
    # plt.plot(xs, spl(xs), 'g', label='BSpline')
    plt.plot(xs, splcvx(xs), "b", label="BSpline CVX")
    # plt.plot(xs, np.polyval(pol, xs), 'r', label='Polynomial')
    plt.plot(xs, np.polyval(polcvx, xs), "r", label="Polynomial CVX")
    plt.legend(loc="best")
    plt.show()
