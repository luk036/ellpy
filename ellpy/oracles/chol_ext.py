# -*- coding: utf-8 -*-
import numpy as np
import math


class chol_ext:
    """chol_ext Cholesky factorization for LMI """
    sqrt_free = True
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
        for i in range(self.n):
            for j in range(i+1):
                d = getA(i, j) - np.dot(T[:j, i], T[j, :j])
                T[i, j] = d
                if i != j:
                    T[j, i] = d / T[j, j]
            if d <= 0.:  # strictly positive
                self.p = i
                return
        self.p = self.n

    # def factor3(self, getA):
    #     """Perform Cholesky Factorization (Lazy evaluation)

    #     Arguments:
    #         getA {function} -- function to access symmetric matrix
    #     """
    #     self.p = 0
    #     self.sqrt_free = False

    #     for i in range(self.n):
    #         for j in range(i+1):
    #             d = getA(i, j) - np.dot(self.T[:j, i], self.T[:j, j])
    #             if i != j:
    #                 self.T[j, i] = d / self.T[j, j]
    #         if d <= 0.:  # strictly positive
    #             self.p = i + 1
    #             self.T[i, i] = math.sqrt(-d)
    #             break
    #         else:
    #             self.T[i, i] = math.sqrt(d)

    def is_spd(self):
        """Is $A$ symmetric positive definite (spd)

        Returns:
            bool -- True if $A$ is a spd
        """
        return self.p == self.n

    def witness(self):
        """witness that certifies $A$ is not symmetric positive definite (spd)
            (square-root-free version)

        Raises:
            AssertionError -- $A$ indeeds a spd matrix

        Returns:
            array, float -- v, ep
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

    # def witness3(self):
    #     """witness that certifies $A$ is not symmetric positive definite (spd)

    #     Raises:
    #         AssertionError -- $A$ indeeds a spd matrix

    #     Returns:
    #         array, float -- v, ep
    #     """
    #     if self.sqrt_free:
    #         raise AssertionError()
    #     if self.is_spd():
    #         raise AssertionError()

    #     p = self.p
    #     v = np.zeros(p)
    #     # r = self.T[p - 1, p - 1]
    #     # ep = 0. if r == 0 else 1.
    #     # v[p - 1] = 1. if r == 0 else 1. / r
    #     v[p - 1] = 1.

    #     for i in range(p - 2, -1, -1):
    #         s = np.dot(self.T[i, i+1:p], v[i+1:p])
    #         v[i] = -(s / self.T[i, i])

    #     ep = self.T[p - 1, p - 1]
    #     return v, ep*ep

    def sqrt(self):
        if not self.is_spd():
            raise AssertionError()

        if not self.sqrt_free:
            return self.T

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
