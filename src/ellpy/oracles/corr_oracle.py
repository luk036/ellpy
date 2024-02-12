# -*- coding: utf-8 -*-
from typing import List, Tuple, Union

import numpy as np
from pylds.low_discr_seq import halton

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


def create_2d_sites(nx=10, ny=8) -> Arr:
    """Create a 2d sites object

    Keyword Arguments:
        nx (int): [description] (default: {10})
        ny (int): [description] (default: {8})

    Returns:
        Arr: location of sites
    """
    n = nx * ny
    s_end = np.array([10.0, 8.0])
    hgen = halton([2, 3])
    s = s_end * np.array([hgen() for _ in range(n)])
    return s


def create_2d_isotropic(s: Arr, N=3000) -> Arr:
    """Create a 2d isotropic object

    Arguments:
        s (Arr): location of sites

    Keyword Arguments:
        N (int): [description] (default: {3000})

    Returns:
        Arr: Biased covariance matrix
    """
    n = s.shape[0]
    sdkern = 0.12  # width of kernel
    var = 2.0  # standard derivation
    tau = 0.00001  # standard derivation of white noise
    np.random.seed(5)

    Sig = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            d = np.array(s[j]) - np.array(s[i])
            Sig[i, j] = np.exp(-sdkern * (d @ d))
            Sig[j, i] = Sig[i, j]

    A = np.linalg.cholesky(Sig)
    Y = np.zeros((n, n))

    for _ in range(N):
        x = var * np.random.randn(n)
        y = A @ x + tau * np.random.randn(n)
        Y += np.outer(y, y)

    Y /= N
    return Y


def construct_distance_matrix(s: Arr) -> Arr:
    """Construct a distance matrix object

    Arguments:
        s (Arr): location of sites

    Returns:
        [type]: [description]
    """
    n = len(s)
    D1 = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            h = s[j] - s[i]
            d = np.sqrt(h @ h)
            D1[i, j] = d
            D1[j, i] = d
    return D1


def construct_poly_matrix(s: Arr, m) -> List[Arr]:
    """Construct distance matrix for polynomial

    Arguments:
        s (Arr): location of sites
        m (int): degree of polynomial

    Returns:
        List[Arr]: [description]
    """
    n = len(s)
    D1 = construct_distance_matrix(s)
    D = np.ones((n, n))
    Sig = [D]
    for _ in range(m - 1):
        D = np.multiply(D, D1)
        Sig += [D]
    return Sig


def corr_poly(Y, s, m, oracle, corr_core):
    """[summary]

    Arguments:
        Y ([type]): [description]
        s ([type]): [description]
        m ([type]): [description]
        oracle ([type]): [description]
        corr_core ([type]): [description]

    Returns:
        [type]: [description]
    """
    Sig = construct_poly_matrix(s, m)
    P = oracle(Sig, Y)
    a, num_iters, feasible = corr_core(Y, m, P)
    pa = np.ascontiguousarray(a[::-1])
    return np.poly1d(pa), num_iters, feasible
