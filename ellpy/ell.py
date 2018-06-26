# -*- coding: utf-8 -*-
import numpy as np
import math


class ell:

    def __init__(self, val, x):
        '''ell = { x | (x - xc)' * P^-1 * (x - xc) <= 1 }'''
        self._use_parallel = True
        n = len(x)
        self.c1 = float(n * n) / (n * n - 1.)
        self._xc = x.copy()
        if np.isscalar(val):
            self.Q = np.eye(n)
            self.kappa = val
        else:
            self.Q = np.diag(val)
            self.kappa = 1.

    def copy(self):
        E = ell(0, self.xc.copy())
        E.Q = self.Q.copy()
        E.c1 = self.c1
        E.kappa = self.kappa
        E._use_parallel = self._use_parallel
        return E

    @property
    def xc(self):
        return self._xc

    @xc.setter
    def xc(self, x):
        self._xc = x

    @property
    def use_parallel(self):
        return self._use_parallel

    @use_parallel.setter
    def use_parallel(self, b):
        self._use_parallel = b

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
        omega = g.dot(Qg)
        tsq = self.kappa * omega
        if tsq <= 0.:
            return 4, 0.
        tau = np.sqrt(tsq)
        # if tau < 0.00000001:
        #     return 2, tau
        alpha = beta / tau
        status, params = calc_ell(alpha)
        if status != 0:
            return status, tsq
        rho, sigma, delta = params
        self._xc -= (self.kappa * rho / tau) * Qg
        self.Q -= (sigma / omega) * np.outer(Qg, Qg)
        self.kappa *= delta
        return status, tsq

    def calc_ll(self, alpha):
        '''parallel or deep cut'''
        if np.isscalar(alpha):
            return self.calc_dc(alpha)
        # parallel cut
        a0 = alpha[0]
        if len(alpha) < 2:
            return self.calc_dc(a0)
        a1 = alpha[1]
        if a1 >= 1. or not self.use_parallel:
            return self.calc_dc(a0)
        n = len(self._xc)
        status = 0
        params = None

        if a0 > a1:
            status = 1  # no sol'n
        elif n*a0*a1 < -1.:
            status = 3  # no effect
        elif a0 == 0:
            params = self.calc_ll_cc(a1, n)
        else:
            params = self.calc_ll_general(a0, a1, n)
        return status, params

    def update(self, cut):
        return self.update_core(self.calc_ll, cut)

    def calc_cc(self):
        '''central cut'''
        n = len(self._xc)
        rho = 1. / (n + 1)
        sigma = 2. * rho
        delta = self.c1
        return rho, sigma, delta

    def calc_dc(self, alpha):
        '''deep cut'''
        if alpha == 0.:
            return 0, self.calc_cc()
        n = len(self._xc)
        status = 0
        params = None
        if alpha > 1.:
            status = 1  # no sol'n
        elif n * alpha < -1.:
            status = 3  # no effect
        else:
            rho = (1. + n * alpha) / (n + 1)
            sigma = 2. * rho / (1. + alpha)
            delta = self.c1 * (1. - alpha * alpha)
            params = (rho, sigma, delta)
        return status, params

    def calc_ll_cc(self, a1, n):
        """Situation when feasible cut."""
        asq1 = a1**2
        xi = math.sqrt((n*asq1)**2 - 4.*asq1 + 4.)
        sigma = (n + (2. - xi) / asq1) / (n + 1)
        rho = a1 * sigma / 2.
        delta = self.c1*(1 - (asq1 - xi / n)/2.)
        return rho, sigma, delta

    def calc_ll_general(self, a0, a1, n):
        asum = a0 + a1
        asq0, asq1 = a0*a0, a1*a1
        asqdiff = asq1 - asq0
        xi = math.sqrt(4. * (1. - asq0) * (
            1. - asq1) + (n*asqdiff)**2)
        sigma = (n + (2. * (1. + a0*a1 - xi/2.)
                      / (asum**2))) / (n + 1)
        rho = asum * sigma / 2.
        delta = self.c1 * (1. - (asq0 + asq1 - xi/n) / 2.)
        return rho, sigma, delta


class ell1d:

    def __init__(self, I):
        l, u = I
        self.r = (u - l)/2
        self._xc = l + self.r

    def copy(self):
        E = ell1d([self._xc - self.r,
                   self._xc + self.r])
        return E

    @property
    def xc(self):
        return self._xc

    @xc.setter
    def xc(self, x):
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
        tau = abs(self.r * g)
        tsq = tau*tau
        if beta == 0.:
            self.r /= 2
            if g > 0.:
                self._xc -= self.r
            else:
                self._xc += self.r
            return 0, tsq
        if beta > tau:
            return 1, tsq  # no sol'n
        if beta < -tau:
            return 3, tsq  # no effect

        bound = self._xc - beta / g
        if g > 0.:
            u = bound
            l = self._xc - self.r
        else:
            l = bound
            u = self._xc + self.r
        self.r = (u - l)/2
        self._xc = l + self.r
        return 0, tsq
