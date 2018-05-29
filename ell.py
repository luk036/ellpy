# -*- coding: utf-8 -*-
import numpy as np
import math


class ell:

    def __init__(self, val, x):
        '''ell = { x | (x - xc)' * P^-1 * (x - xc) <= 1 }'''
        self.use_parallel = True
        n = len(x)
        self.c1 = float(n * n) / (n * n - 1.)
        self._xc = x.copy()
        if np.isscalar(val):
            self.Q = np.eye(n)
            self.kappa = val
        else:
            self.Q = np.diag(val)
            self.kappa = 1.

    @property
    def xc(self):
        return self._xc

    def copy(self):
        E = ell(0, self.xc.copy())
        E.Q = self.Q.copy()
        E.c1 = self.c1
        E.kappa = self.kappa
        E.use_parallel = self.use_parallel
        return E

    def update_core(self, calc_ell, cut):
        """Update ellipsoid core function using the cut
                g' * (x - xc) + beta <= 0

        Arguments:
            calc_ell {[type]} -- [description]
            g {array} -- cut
            beta {array or scalar} -- [description]

        Returns:
            status -- 0: success
            tau -- "volumn" of ellipsoid
        """
        g, beta = cut
        Qg = self.Q.dot(g)
        tsq = g.dot(Qg)
        if tsq <= 0.:
            return 4, 0.
        tau = np.sqrt(self.kappa * tsq)
        if tau < 0.00000001:
            return 2, tau
        if beta == 0:
            status, params = self.calc_cc()
        else:
            alpha = beta / tau
            status, params = calc_ell(alpha)
            if status != 0:
                return status, tau
        rho, sigma, delta = params
        self._xc -= (self.kappa * rho / tau) * Qg
        self.Q -= (sigma / tsq) * np.outer(Qg, Qg)
        # self.Q *= delta
        self.kappa *= delta
        return status, tau

    def calc_cc(self):
        '''central cut'''
        n = len(self._xc)
        rho = 1. / (n + 1)
        sigma = 2. * rho
        delta = self.c1
        return 0, (rho, sigma, delta)

    def calc_dc(self, alpha):
        '''deep cut'''
        if alpha == 0.:
            return self.calc_cc()
        n = len(self._xc)
        # status, rho, sigma, delta = 0, 0., 0., 0.
        if alpha > 1.:
            return 1, None  # no sol'n
        elif n * alpha < -1.:
            return 3, None  # no effect
        else:
            rho = (1. + n * alpha) / (n + 1)
            sigma = 2. * rho / (1. + alpha)
            delta = self.c1 * (1. - alpha * alpha)
        return 0, (rho, sigma, delta)

    def calc_ll(self, alpha):
        '''parallel or deep cut'''
        if np.isscalar(alpha):
            return self.calc_dc(alpha)
        # parallel cut
        a0, a1 = alpha
        if a1 >= 1. or not self.use_parallel:
            return self.calc_dc(a0)
        n = len(self._xc)
        # status, rho, sigma, delta = 0, 0., 0., 0.
        aprod = a0 * a1
        if a0 > a1:
            return 1, None  # no sol'n
        elif n * aprod < -1.:
            return 3, None  # no effect
        else:
            asq = alpha * alpha
            asum = a0 + a1
            asq0, asq1 = asq
            asqdiff = asq1 - asq0
            xi = math.sqrt(4. * (1. - asq0) * (
                1. - asq1) + (n * asqdiff)**2)
            sigma = (n + (2. * (1. + aprod - xi / 2.)
                          / (asum * asum))) / (n + 1)
            rho = asum * sigma / 2.
            delta = self.c1 * (1. - (asq0 + asq1 - xi / n) / 2.)
        return 0, (rho, sigma, delta)

    def update(self, cut):
        return self.update_core(self.calc_ll, cut)


class ell1d:

    def __init__(self, I):
        l, u = I
        self.r = (u - l)/2
        self._xc = l + self.r

    @property
    def xc(self):
        return self._xc

    def copy(self):
        E = ell1d([self._xc - self.r,
                   self._xc + self.r])
        return E

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
        tau = abs(self.r * g)
        if beta == 0.:
            self.r /= 2
            if g > 0.:
                self._xc -= self.r
            else:
                self._xc += self.r
            return 0, tau
        if beta > tau:
            return 1, tau  # no sol'n
        if beta < -tau:
            return 3, tau  # no effect
        bound = self._xc - beta / g
        if g > 0.:
            u = bound
            l = self._xc - self.r
        else:
            l = bound
            u = self._xc + self.r
        self.r = (u - l)/2
        self._xc = l + self.r
        return 0, tau
