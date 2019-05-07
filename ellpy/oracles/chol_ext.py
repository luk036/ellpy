# -*- coding: utf-8 -*-
import numpy as np
import math


class chol_ext:
    """Cholesky factorization for LMI 

       - LDLT square-root-free version
       - Strictly positive, i.e. reject zero
       - A matrix $A$ is positive definite iff v^T A v > 0 for all v in R^n.
    """
    p = None

    def __init__(self, N):
        """initialization

        Arguments:
            N {integer} -- dimension
        """
        self.T = np.zeros((N, N))
        self.n = N

    def factorize(self, A):
        """Perform Cholesky Factorization

        Arguments:
            A {np.array} -- Symmetric Matrix

         If $A$ is positive definite, then $p$ is zero.
         If it is not, then $p$ is a positive integer,
         such that $v = R^{-1} e_p$ is a certificate vector
         to make $v'*A[:p,:p]*v < 0$
        """
        self.factor(lambda i, j: A[i, j])

    def factor(self, getA):
        """Perform Cholesky Factorization (square-root free version)

        Arguments:
            getA {function} -- function to access symmetric matrix
        """
        T = self.T
        self.p = self.n
        for i in range(self.n):
            for j in range(i+1):
                d = getA(i, j) - np.dot(T[:j, i], T[j, :j])
                T[i, j] = d
                if i != j:
                    T[j, i] = d / T[j, j]
            if d <= 0.:  # strictly positive, reject zero also
                self.p = i
                break

    def is_spd(self):
        """Is $A$ symmetric positive definite (spd)

        Returns:
            bool -- True if $A$ is a spd
        """
        return self.p == self.n

    def witness(self):
        """witness that certifies $A$ is not symmetric positive definite (spd)
            (square-root-free version)

           evidence $v^T A v = ep$

        Raises:
            AssertionError -- $A$ indeeds a spd matrix

        Returns:
            array -- v
            float -- ep
        """
        if self.is_spd():
            raise AssertionError()
        p = self.p
        n = p + 1
        v = np.zeros(n)
        v[p] = 1.

        for i in range(p, 0, -1):
            v[i-1] = -np.dot(self.T[i-1, i:n], v[i:n])
        return v, -self.T[p, p]

    def sqrt(self):
        if not self.is_spd():
            raise AssertionError()
        n = self.n
        M = np.zeros((n, n))
        for i in range(self.n):
            M[i, i] = math.sqrt(self.T[i, i])
            for j in range(i+1, n):
                M[i, j] = self.T[i, j] * M[i, i]
        return M

    def sym_quad(self, v, A):
        """[summary]

        Arguments:
            v {[type]} -- [description]
            A {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        n = self.p + 1
        return v.dot(A[:n, :n].dot(v))
