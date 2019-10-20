# -*- coding: utf-8 -*-
import math

import numpy as np


class ell:
    _use_parallel_cut = True
    _no_defer_trick = False

    _n: int  # dimension
    c1: float
    kappa: float
    rho: float
    sigma: float
    delta: float
    tsq: float

    def __init__(self, val, x):
        """ell = { x | (x - xc)' * P^-1 * (x - xc) <= 1 }

        Arguments:
            val {[type]} -- [description]
            x {ndarray} -- [description]
        """
        self._n = n = len(x)
        self.c1 = float(n * n) / (n * n - 1)
        self._xc = x.copy()
        if np.isscalar(val):
            self.Q = np.eye(n)
            self.kappa = val
        else:
            self.Q = np.diag(val)
            self.kappa = 1.

    def copy(self):
        """[summary]

        Returns:
            ell -- [description]
        """
        E = ell(self.kappa, self.xc)
        E.Q = self.Q.copy()
        E.c1 = self.c1
        E._use_parallel_cut = self._use_parallel_cut
        E._no_defer_trick = self._no_defer_trick
        return E

    @property
    def xc(self):
        """[summary]

        Returns:
            [type] -- [description]
        """
        return self._xc

    @xc.setter
    def xc(self, x):
        """[summary]

        Arguments:
            x {[type]} -- [description]
        """
        self._xc = x

    @property
    def use_parallel_cut(self) -> bool:
        """[summary]

        Returns:
            bool -- [description]
        """
        return self._use_parallel_cut

    @use_parallel_cut.setter
    def use_parallel_cut(self, b: bool):
        """[summary]

        Arguments:
            b {bool} -- [description]
        """
        self._use_parallel_cut = b

    def update(self, cut):
        """[summary]

        Arguments:
            cut {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        return self.update_core(self.__calc_ll, cut)

    def update_core(self, calc_ell, cut):
        """Update ellipsoid core function using the cut

                g' * (x - xc) + beta <= 0

            Note: At most one square-root per iteration.

        Arguments:
            calc_ell {[type]} -- [description]
            cut {[type]} -- [description]

        Returns:
            status -- 0: success
            tau -- "volumn" of ellipsoid
        """
        g, beta = cut
        Qg = self.Q.dot(g)  # n^2 multiplications
        omega = g.dot(Qg)  # n^2 multiplications
        self.tsq = self.kappa * omega
        # if self.tsq <= 0.: # unlikely
        #     return 4, 0.
        status = calc_ell(beta)
        if status != 0:
            return status, self.tsq
        # rho, sigma, delta = params
        self._xc -= (self.rho / omega) * Qg
        self.Q -= (self.sigma / omega) * np.outer(Qg, Qg)  # n*(n+1)/2
        self.kappa *= self.delta
        if self._no_defer_trick:
            self.Q *= self.kappa
            self.kappa = 1.
        return status, self.tsq

    def __calc_ll(self, beta) -> int:
        """parallel or deep cut

        Arguments:
            beta {[type]} -- [description]

        Returns:
            int -- [description]
        """
        if np.isscalar(beta):
            return self.__calc_dc(beta)
        if len(beta) < 2:  # unlikely
            return self.__calc_dc(beta[0])
        return self.__calc_ll_core(beta[0], beta[1])

    def __calc_ll_core(self, b0: float, b1: float) -> int:
        """[summary]

        Arguments:
            b0 {float} -- [description]
            b1 {float} -- [description]

        Returns:
            int -- [description]
        """
        b1sq = b1**2
        if b1sq > self.tsq or not self.use_parallel_cut:
            return self.__calc_dc(b0)
        if b1 < b0:  # unlikely
            return 1  # no sol'n
        if b0 == 0:
            self.__calc_ll_cc(b1, b1sq)
            return 0

        n = self._n
        b0b1 = b0 * b1
        if n * b0b1 < -self.tsq:  # unlikely
            return 3  # no effect

        # parallel cut
        b0sq = b0**2
        t0 = self.tsq - b0sq
        t1 = self.tsq - b1sq
        bav = (b0 + b1) / 2
        xi = math.sqrt(4 * t0 * t1 + (n * (b1sq - b0sq))**2)
        self.sigma = (n + (self.tsq - b0b1 - xi / 2) / (2 * bav**2)) / (n + 1)
        self.rho = self.sigma * bav
        self.delta = self.c1 * (t0 + t1 + xi / n) / (2 * self.tsq)
        return 0

    def __calc_ll_cc(self, b1: float, b1sq: float):
        """parallel central cut

        Arguments:
            b1 {float} -- [description]
            b1sq {float} -- [description]
        """
        n = self._n
        xi = math.sqrt(4 * self.tsq * (self.tsq - b1sq) + (n * b1sq)**2)
        self.sigma = (n + (2 * self.tsq - xi) / b1sq) / (n + 1)
        self.rho = self.sigma * b1 / 2
        self.delta = self.c1 * (self.tsq - (b1sq - xi / n) / 2) / self.tsq

    def __calc_dc(self, beta: float) -> int:
        """Deep cut

        Arguments:
            beta {float} -- [description]

        Returns:
            int -- [description]
        """
        tau = math.sqrt(self.tsq)
        if beta > tau:
            return 1  # no sol'n
        if beta == 0.:
            self.__calc_cc(tau)
            return 0
        n = self._n
        gamma = tau + n * beta
        if gamma < 0.:
            return 3  # no effect, unlikely

        self.rho = gamma / (n + 1)
        self.sigma = 2 * self.rho / (tau + beta)
        self.delta = self.c1 * (self.tsq - beta**2) / self.tsq
        return 0

    def __calc_cc(self, tau):
        """Central cut

        Arguments:
            tau {float} -- [description]
        """
        np1 = self._n + 1
        self.sigma = 2. / np1
        self.rho = tau / np1
        self.delta = self.c1


class ell1d:
    _r: float
    _xc: float

    def __init__(self, I):
        """[summary]

        Arguments:
            I {[type]} -- [description]
        """
        l, u = I
        self._r = (u - l) / 2
        self._xc = l + self._r

    def copy(self):
        """[summary]

        Returns:
            [type] -- [description]
        """
        E = ell1d([self._xc - self._r, self._xc + self._r])
        return E

    @property
    def xc(self) -> float:
        """[summary]

        Returns:
            float -- [description]
        """
        return self._xc

    @xc.setter
    def xc(self, x: float):
        """[summary]

        Arguments:
            x {float} -- [description]
        """
        self._xc = x

    def update(self, cut):
        """Update ellipsoid core function using the cut
                g' * (x - xc) + beta <= 0

        Arguments:
            g {floay} -- cut
            beta {array or scalar} -- [description]

        Returns:
            status -- 0: success
            tau -- "volumn" of ellipsoid
        """
        g, beta = cut
        tau = abs(self._r * g)
        tsq = tau**2
        if beta == 0:
            self._r /= 2
            self._xc += -self._r if g > 0. else self._r
            return 0, tsq
        if beta > tau:
            return 1, tsq  # no sol'n
        if beta < -tau:  # unlikely
            return 3, tsq  # no effect

        bound = self._xc - beta / g
        upper = bound if g > 0 else self._xc + self._r
        lower = self._xc - self._r if g > 0. else bound
        self._r = (upper - lower) / 2
        self._xc = lower + self._r
        return 0, tsq
