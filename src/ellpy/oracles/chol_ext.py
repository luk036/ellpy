# -*- coding: utf-8 -*-
import math
from typing import Union

import numpy as np

Arr = Union[np.ndarray]


class chol_ext:
    """Cholesky factorization for LMI

       - LDL^T square-root-free version
       - Option allow semidefinite
       - A matrix $A in R^{m x m}$ is positive definite iff v' A v > 0
           for all v in R^n.
       - O(p^3) per iteration, independent of N

        Member variables:
            p (int, int): the rows where the process starts and stops
            v (Arr): witness
            n (int): dimension
    """
    __slots__ = ('p', 'v', '_n', '_T', 'allow_semidefinite')

    def __init__(self, N: int):
        """Construct a new chol ext object

        Arguments:
            N (int): dimension
        """
        self.allow_semidefinite = False
        self.p = (0, 0)
        self.v: Arr = np.zeros(N)

        self._n: int = N
        self._T: Arr = np.zeros((N, N))

    def factorize(self, A: Arr):
        """Perform Cholesky Factorization

        Arguments:
            A (np.array): Symmetric Matrix

         If $A$ is positive definite, then $p$ is zero.
         If it is not, then $p$ is a positive integer,
         such that $v = R^{-1} e_p$ is a certificate vector
         to make $v'*A[:p,:p]*v < 0$
        """
        self.factor(lambda i, j: A[i, j])

    def factor(self, getA):
        """Perform Cholesky Factorization (square-root free version)

        Arguments:
            getA (callable): function to access symmetric matrix
        """
        T = self._T
        start = 0  # range start
        self.p = (0, 0)
        for i in range(self._n):
            d = getA(i, start)
            for j in range(start, i):
                T[i, j] = d
                T[j, i] = d / T[j, j]
                d = getA(i, j+1) - np.dot(T[start:j+1, i], T[j+1, start:j+1])
            T[i, i] = d
            if d > 0.:
                continue
            if d < 0. or not self.allow_semidefinite:
                self.p = start, i + 1
                break
            start = i + 1  # T[i, i] == 0, restart at i+1

    def is_spd(self):
        """Is $A$ symmetric positive definite (spd)

        Returns:
            bool: True if $A$ is a spd
        """
        return self.p[1] == 0

    def witness(self):
        """witness that certifies $A$ is not symmetric positive definite (spd)
            (square-root-free version)

           evidence: v' A v = -ep

        Raises:
            AssertionError: $A$ indeeds a spd matrix

        Returns:
            array: v
            float: ep
        """
        if self.is_spd():
            raise AssertionError()
        start, n = self.p
        m = n - 1
        self.v[m] = 1.
        for i in range(m, start, -1):
            self.v[i - 1] = -(self._T[i - 1, i:n] @ self.v[i:n])
        return -self._T[m, m]

    def sym_quad(self, A: Arr):
        """[summary]

        Arguments:
            v ([type]): [description]
            A ([type]): [description]

        Returns:
            [type]: [description]
        """
        s, n = self.p
        v = self.v[s:n]
        return v @ A[s:n, s:n] @ v

    def sqrt(self) -> Arr:
        if not self.is_spd():
            raise AssertionError()
        M = np.zeros((self._n, self._n))
        for i in range(self._n):
            M[i, i] = math.sqrt(self._T[i, i])
            for j in range(i + 1, self._n):
                M[i, j] = self._T[i, j] * M[i, i]
        return M
