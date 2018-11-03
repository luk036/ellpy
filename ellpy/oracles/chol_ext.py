# -*- coding: utf-8 -*-
import numpy as np
import math


class chol_ext:
    def __init__(self, N):
        self.R = np.zeros((N, N))
        self.p = 0
        # self.v = np.zeros(N)
        # self.d = np.zeros(N)

    # def factorize2(self, A):
    #     '''
    #     (square-root-free version)
    #      If $A$ is positive definite, then $p$ is zero.
    #      If it is not, then $p$ is a positive integer,
    #      such that $v = R^{-1} e_p$ is a certificate vector
    #      to make $v'*A[:p,:p]*v < 0$
    #     '''
    #     # N = len(A)
    #     # self.p = 0
    #     self.p = 0
    #     R = self.R
    #     d = self.d
    #     N = len(R)
    #     for i in range(N):
    #         for j in range(i+1):
    #             d[i] = A[i, j] - np.dot(d[:j], R[:j, i] * R[:j, j])
    #             if i != j:
    #                 R[j, i] = d[i] / d[j]
    #         if d[i] <= 0:  # strictly positive???
    #             self.p = i + 1
    #             break

    def factorize(self, A):
        '''
         If $A$ is positive definite, then $p$ is zero.
         If it is not, then $p$ is a positive integer,
         such that $v = R^{-1} e_p$ is a certificate vector
         to make $v'*A[:p,:p]*v < 0$
        '''
        return self.factor(lambda i, j: A[i, j])

        # # N = len(A)
        # # self.p = 0
        # self.p = 0
        # R = self.R
        # N = len(R)
        # for i in range(N):
        #     for j in range(i+1):
        #         d = A[i, j] - np.dot(R[:j, i], R[:j, j])
        #         if i != j:
        #             R[j, i] = d / R[j, j]
        #     if d <= 0:  # strictly positive???
        #         self.p = i + 1
        #         R[i, i] = math.sqrt(-d)
        #         break
        #     R[i, i] = math.sqrt(d)

    # def factor2(self, getA):
    #     '''(lazy evalution of A)
    #        (square-root-free version)
    #     '''
    #     # N = len(A)
    #     # self.p = 0
    #     self.p = 0
    #     R = self.R
    #     d = self.d
    #     N = len(R)
    #     for i in range(N):
    #         for j in range(i+1):
    #             d[i] = getA(i, j) - np.dot(d[:j], R[:j, i]*R[:j, j])
    #             if i != j:
    #                 R[j, i] = d[i] / d[j]
    #         if d[i] <= 0:  # strictly positive???
    #             self.p = i + 1
    #             break

    def factor(self, getA):
        '''(lazy evalution of A)'''
        # N = len(A)
        # self.p = 0
        self.p = 0
        R = self.R
        N = len(R)
        for i in range(N):
            for j in range(i+1):
                d = getA(i, j) - np.dot(R[:j, i], R[:j, j])
                if i != j:
                    R[j, i] = d / R[j, j]
            if d <= 0:  # strictly positive???
                self.p = i + 1
                R[i, i] = math.sqrt(-d)
                break
            R[i, i] = math.sqrt(d)

    def is_spd(self):
        """is symmetric positive definite"""
        return self.p == 0

    # def witness2(self):
    #     '''
    #     (square-root-free version)
    #     '''
    #     if self.is_spd():
    #         raise AssertionError()
    #     p = self.p
    #     v = np.zeros(p)
    #     v[p-1] = 1. / math.sqrt(-self.d[p-1])
    #     for i in range(p - 2, -1, -1):
    #         v[i] = -np.dot(self.R[i, i+1:p], v[i+1:p])
    #     return v

    def witness(self):
        if self.is_spd():
            raise AssertionError()
        p = self.p
        v = np.zeros(p)
        r = self.R[p - 1, p - 1]
        v[p - 1] = 1. if r == 0 else 1. / r
        ep = 0. if r == 0 else 1.
        for i in range(p - 2, -1, -1):
            s = np.dot(self.R[i, i+1:p], v[i+1:p])
            v[i] = -s / self.R[i, i]
        return v, ep

    def sym_quad(self, v, F):
        # v = self.witness()
        p = self.p
        return v.dot(F[:p, :p].dot(v))
