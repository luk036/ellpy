# -*- coding: utf-8 -*-
import numpy as np
import math


class chol_ext:
    """chol_ext Cholesky factorization for LMI """
    sqrt_free = True
    p = 0

    def __init__(self, N):
        """initialization

        Arguments:
            N {integer} -- dimension
        """
        self.R = np.zeros((N, N))
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
        self.p = 0

        for i in range(self.n):
            for j in range(i+1):
                d = getA(i, j) - np.dot(self.R[:j, i], self.R[j, :j])
                if i != j:
                    self.R[i, j] = d
                    self.R[j, i] = d / self.R[j, j]
            if d <= 0.:  # strictly positive
                self.p = i + 1
                self.R[i, i] = -d
                break
            else:
                self.R[i, i] = d

    # def factor3(self, getA):
    #     """Perform Cholesky Factorization (Lazy evaluation)

    #     Arguments:
    #         getA {function} -- function to access symmetric matrix
    #     """
    #     self.p = 0
    #     self.sqrt_free = False

    #     for i in range(self.n):
    #         for j in range(i+1):
    #             d = getA(i, j) - np.dot(self.R[:j, i], self.R[:j, j])
    #             if i != j:
    #                 self.R[j, i] = d / self.R[j, j]
    #         if d <= 0.:  # strictly positive
    #             self.p = i + 1
    #             self.R[i, i] = math.sqrt(-d)
    #             break
    #         else:
    #             self.R[i, i] = math.sqrt(d)

    def is_spd(self):
        """Is $A$ symmetric positive definite (spd)

        Returns:
            bool -- True if $A$ is a spd
        """
        return self.p == 0

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
        v = np.zeros(p)
        # r = self.R[p - 1, p - 1]
        # ep = 0. if r == 0 else 1.
        # v[p - 1] = 1. if r == 0 else 1. / math.sqrt(r)
        v[p - 1] = 1.

        for i in range(p - 2, -1, -1):
            v[i] = -np.dot(self.R[i, i+1:p], v[i+1:p])
        return v, self.R[p - 1, p - 1]

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
    #     # r = self.R[p - 1, p - 1]
    #     # ep = 0. if r == 0 else 1.
    #     # v[p - 1] = 1. if r == 0 else 1. / r
    #     v[p - 1] = 1.

    #     for i in range(p - 2, -1, -1):
    #         s = np.dot(self.R[i, i+1:p], v[i+1:p])
    #         v[i] = -(s / self.R[i, i])

    #     ep = self.R[p - 1, p - 1]
    #     return v, ep*ep

    def sqrt(self):
        if not self.is_spd():
            raise AssertionError()

        if not self.sqrt_free:
            return self.R

        n = self.n
        M = np.zeros((n, n))

        for i in range(self.n):
            M[i, i] = math.sqrt(self.R[i, i])
            for j in range(i+1, n):
                M[i, j] = self.R[i, j] * M[i, i]

        return M

    def sym_quad(self, v, A):
        """[summary]

        Arguments:
            v {[type]} -- [description]
            A {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        p = self.p
        return v.dot(A[:p, :p].dot(v))
