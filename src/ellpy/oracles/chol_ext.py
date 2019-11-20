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
       - O(p^2 n) per iteration, independent of N

        Member variables:
            p (integer): the rows where the process starts and stops
            v (Arr): witness
            n (integer): dimension
    """
    __slots__ = ('p', 'v', 'n', 'T', 'allow_semidefinite')

    def __init__(self, N: int):
        """initialization

        Arguments:
            N (integer): dimension
        """
        self.allow_semidefinite = False
        self.p = (0, 0)

        self.v = np.zeros(N)
        self.n = N
        self.T = np.zeros((N, N))

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
        T = self.T
        start = 0  # range start
        self.p = (0, 0)
        for i in range(self.n):
            for j in range(start, i + 1):
                d = getA(i, j) - np.dot(T[start:j, i], T[j, start:j])
                T[i, j] = d
                if i != j:
                    T[j, i] = d / T[j, j]
            if T[i, i] > 0.:
                continue
            if T[i, i] < 0. or not self.allow_semidefinite:
                self.p = start, i + 1
                break
            start = i + 1  # T[i, i] == 0, restart at i+1

    def is_spd(self):
        """Is $A$ symmetric positive definite (spd)

        Returns:
            bool: True if $A$ is a spd
        """
        return self.p == (0, 0)

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
            self.v[i - 1] = -(self.T[i - 1, i:n] @ self.v[i:n])
        return -self.T[m, m]

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

    def sqrt(self):
        if not self.is_spd():
            raise AssertionError()
        n = self.n
        M = np.zeros((n, n))
        for i in range(n):
            M[i, i] = math.sqrt(self.T[i, i])
            for j in range(i + 1, n):
                M[i, j] = self.T[i, j] * M[i, i]
        return M
