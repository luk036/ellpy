# -*- coding: utf-8 -*-
import numpy as np
import math


class ell:
    _use_parallel = True

    def __init__(self, val, x):
        '''ell = { x | (x - xc)' * P^-1 * (x - xc) <= 1 }'''
        self._n = n = len(x)
        self.c1 = float(n*n) / (n*n - 1)
        self._xc = x.copy()
        if np.isscalar(val):
            self.Q = np.eye(n)
            self.kappa = val
        else:
            self.Q = np.diag(val)
            self.kappa = 1.

    def copy(self):
        E = ell(self.kappa, self.xc)
        E.Q = self.Q.copy()
        E.c1 = self.c1
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
        status, params = calc_ell(beta, tsq)
        if status != 0:
            return status, tsq
        rho, sigma, delta = params
        self._xc -= (rho / omega) * Qg
        self.Q -= (sigma / omega) * np.outer(Qg, Qg)
        self.kappa *= delta
        return status, tsq

    def calc_ll(self, beta, tsq):
        '''parallel or deep cut'''
        if np.isscalar(beta):
            return self.calc_dc(beta, tsq)

        b0 = beta[0]
        if len(beta) < 2:
            return self.calc_dc(b0, tsq)
        return self.calc_ll_core(b0, beta[1], tsq)

    def calc_ll_core(self, b0, b1, tsq):
        t1 = tsq - b1*b1
        if t1 < 0. or not self.use_parallel:
            return self.calc_dc(b0, tsq)

        l = b1 - b0
        if l < 0:
            return 1, None  # no sol'n

        n = self._n
        p = b0*b1
        if n*p < -tsq:
            return 3, None  # no effect

        params = None

        # parallel cut
        if b0 == 0:
            params = self.calc_ll_cc(b1, t1, tsq)
        else:
            t0 = tsq - b0*b0
            bav = (b0 + b1)/2
            xi = math.sqrt(t0*t1 + (n*bav*l)**2)
            sigma = (n + (tsq - p - xi)/(2*bav*bav)) / (n + 1)
            rho = sigma * bav
            delta = self.c1 * ((t0 + t1)/2 + xi/n) / tsq
            params = rho, sigma, delta

        return 0, params

    def update(self, cut):
        return self.update_core(self.calc_ll, cut)

    def calc_cc(self, tsq):
        '''central cut'''
        np1 = self._n + 1
        sigma = 2. / np1
        rho = math.sqrt(tsq) / np1
        delta = self.c1
        return rho, sigma, delta

    def calc_dc(self, b0, tsq):
        '''deep cut'''
        if b0 == 0.:
            return 0, self.calc_cc(tsq)

        t0 = tsq - b0*b0
        if t0 < 0.:
            return 1, None    # no sol'n

        n = self._n
        tau = math.sqrt(tsq)
        gamma = tau + n * b0
        if gamma < 0.:
            return 3, None  # no effect

        rho = gamma / (n + 1)
        sigma = 2. * rho / (tau + b0)
        delta = self.c1 * t0/tsq
        params = (rho, sigma, delta)
        return 0, params

    def calc_ll_cc(self, b1, t1, tsq):
        """Situation when feasible cut."""
        n = self._n
        hsq1 = tsq - t1
        xi = math.sqrt(tsq*t1 + (n*hsq1/2)**2)
        sigma = (n + 2*(tsq - xi) / hsq1) / (n + 1)
        rho = sigma * b1 / 2
        delta = self.c1 * (tsq - hsq1/2 - xi/n) / tsq
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
        tsq = tau**2
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
