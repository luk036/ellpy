# -*- coding: utf-8 -*-
from __future__ import print_function

from pprint import pprint
import numpy as np


def chol_ext(A):
    '''
     If $A$ is positive definite, then $p$ is zero.
     If it is not, then $p$ is a positive integer,
     such that $v = R^{-1} e_p$ is a certificate vector
     to make $v'*A[:p,:p]*v < 0$
    '''
    p = 0
    n = len(A)
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            d = A[j, i] - np.dot(R[:j, i], R[:j, j])
            if i == j:
                if d < 0.:
                    p = i + 1
                    R[j, i] = np.sqrt(-d)
                    return R[:p, :p], p
                else:
                    R[j, i] = np.sqrt(d)
            else:
                R[j, i] = 1.0 / R[j, j] * d
    return R, p


def witness(R, p):
    assert p > 0
    v = np.zeros(p)
    v[-1] = 1.0 / R[-1, -1]
    for i in range(p - 2, -1, -1):
        s = np.dot(R[i, i + 1:], v[i + 1:])
        v[i] = -s / R[i, i]
    return v


def print_case(l1):
    m1 = np.array(l1)
    R, p = chol_ext(m1)
    pprint(R)
    if p > 0:
        v = witness(R, p)
        print(np.dot(v, m1[:p, :p].dot(v)))


if __name__ == "__main__":
    l1 = [[25., 15., -5.],
          [15., 18.,  0.],
          [-5.,  0., 11.]]
    print_case(l1)

    l2 = [[18., 22.,  54.,  42.],
          [22., -70.,  86.,  62.],
          [54., 86., -174., 134.],
          [42., 62., 134., -106.]]
    print_case(l2)
