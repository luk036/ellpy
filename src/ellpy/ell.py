# -*- coding: utf-8 -*-
import math
from typing import Tuple, Union

import numpy as np

from .cutting_plane import CUTStatus

Arr = Union[np.ndarray]


class ell:
    """Ellipsoid Search Space

            ell = {x | (x − xc)' Q^−1 (x − xc) ​≤ κ}

    Returns:
        [type] -- [description]
    """
    __slots__ = ('_n', '_c1', '_kappa', '_rho', '_sigma', '_delta', '_tsq',
                 '_xc', '_Q', 'use_parallel_cut', 'no_defer_trick')

    def __init__(self, val: Union[Arr, float], x: Arr):
        """Construct a new ell object

        Arguments:
            val (Union[Arr, float]): [description]
            x (Arr): [description]
        """
        self.use_parallel_cut = True
        self.no_defer_trick = False

        self._n = n = len(x)
        self._c1 = float(n * n) / (n * n - 1)
        self._xc = x
        if np.isscalar(val):
            self._Q = np.eye(n)
            self._kappa = val
        else:
            self._Q = np.diag(val)
            self._kappa = 1.

    def copy(self):
        """[summary]

        Returns:
            ell: [description]
        """
        E = ell(self._kappa, self.xc)
        E._Q = self._Q.copy()
        # E._c1 = self._c1
        E.use_parallel_cut = self.use_parallel_cut
        E.no_defer_trick = self.no_defer_trick
        return E

    @property
    def xc(self):
        """copy the whole array anyway

        Returns:
            [type]: [description]
        """
        return self._xc

    @xc.setter
    def xc(self, x: Arr):
        """Set the xc object

        Arguments:
            x ([type]): [description]
        """
        self._xc = x

    # @property
    # def use_parallel_cut(self) -> bool:
    #     """[summary]

    #     Returns:
    #         bool: [description]
    #     """
    #     return self._use_parallel_cut

    # @use_parallel_cut.setter
    # def use_parallel_cut(self, b: bool):
    #     """[summary]

    #     Arguments:
    #         b (bool): [description]
    #     """
    #     self._use_parallel_cut = b

    def update(self, cut) -> Tuple[int, float]:
        """Update ellipsoid core function using the cut(s)

        Arguments:
            cut: cutting-plane

        Returns:
            Tuple[int, float]: [description]
        """
        return self.update_core(self._calc_ll, cut)

    def update_core(self, calc_ell, cut):
        """Update ellipsoid core function using the cut(s)

                g' * (x − xc) + beta <= 0

            Note: At most one square-root per iteration.

        Arguments:
            calc_ell ([type]): [description]
            cut (float): [description]

        Returns:
            status: 0: success
            tau: "volumn" of ellipsoid
        """
        g, beta = cut
        Qg = self._Q.dot(g)  # n^2 multiplications
        omega = g.dot(Qg)  # n^2 multiplications
        self._tsq = self._kappa * omega
        status = calc_ell(beta)
        if status != CUTStatus.success:
            return status, self._tsq

        self._xc -= (self._rho / omega) * Qg
        self._Q -= (self._sigma / omega) * np.outer(Qg, Qg)  # n*(n+1)/2
        self._kappa *= self._delta

        if self.no_defer_trick:
            self._Q *= self._kappa
            self._kappa = 1.
        return status, self._tsq

    def _calc_ll(self, beta) -> int:
        """parallel or deep cut

        Arguments:
            beta ([type]): [description]

        Returns:
            int: [description]
        """
        if np.isscalar(beta):
            return self._calc_dc(beta)
        if len(beta) < 2:  # unlikely
            return self._calc_dc(beta[0])
        return self._calc_ll_core(beta[0], beta[1])

    def _calc_ll_core(self, b0: float, b1: float) -> int:
        """Calculate new ellipsoid under Parallel Cut

                g' (x − xc​) + β0 ​≤ 0
                g' (x − xc​) + β1 ​≥ 0

        Arguments:
            b0 (float): [description]
            b1 (float): [description]

        Returns:
            int: [description]
        """
        b1sq = b1**2
        if b1sq > self._tsq or not self.use_parallel_cut:
            return self._calc_dc(b0)
        if b1 < b0:  # unlikely
            return CUTStatus.nosoln  # no sol'n
        if b0 == 0:
            self._calc_ll_cc(b1, b1sq)
            return CUTStatus.success

        n = self._n
        b0b1 = b0 * b1
        if n * b0b1 < -self._tsq:  # unlikely
            return CUTStatus.noeffect  # no effect

        # parallel cut
        t0 = self._tsq - b0 * b0
        t1 = self._tsq - b1sq
        bav = (b0 + b1) / 2
        xi = math.sqrt(t0 * t1 + (n * bav * (b1 - b0))**2)
        self._sigma = (n + (self._tsq - b0b1 - xi) / (2 * bav**2)) / (n + 1)
        self._rho = self._sigma * bav
        self._delta = self._c1 * ((t0 + t1) / 2 + xi / n) / self._tsq
        return CUTStatus.success

    def _calc_ll_cc(self, b1: float, b1sq: float):
        """Calculate new ellipsoid under Parallel Cut, one of them is central

                g' (x − xc​) ​≤ 0
                g' (x − xc​) + β1 ​≥ 0

        Arguments:
            b1 (float): [description]
            b1sq (float): [description]
        """
        n = self._n
        xi = math.sqrt(self._tsq * (self._tsq - b1sq) + (n * b1sq / 2)**2)
        self._sigma = (n + 2 * (self._tsq - xi) / b1sq) / (n + 1)
        self._rho = self._sigma * b1 / 2
        self._delta = self._c1 * (self._tsq - b1sq / 2 + xi / n) / self._tsq

    def _calc_dc(self, beta: float) -> int:
        """Calculate new ellipsoid under Deep Cut

                g' (x − xc​) + β ​≤ 0

        Arguments:
            beta (float): [description]

        Returns:
            int: [description]
        """
        tau = math.sqrt(self._tsq)
        if beta > tau:
            return CUTStatus.nosoln  # no sol'n
        if beta == 0.:
            self._calc_cc(tau)
            return CUTStatus.success
        n = self._n
        gamma = tau + n * beta
        if gamma < 0.:
            return CUTStatus.noeffect  # no effect, unlikely

        self._rho = gamma / (n + 1)
        self._sigma = 2 * self._rho / (tau + beta)
        self._delta = self._c1 * (self._tsq - beta**2) / self._tsq
        return CUTStatus.success

    def _calc_cc(self, tau: float):
        """Calculate new ellipsoid under Central Cut

        Arguments:
            tau (float): [description]
        """
        np1 = self._n + 1
        self._sigma = 2. / np1
        self._rho = tau / np1
        self._delta = self._c1


class ell1d:
    __slots__ = ('_r', '_xc')

    def __init__(self, I):
        """[summary]

        Arguments:
            I ([type]): [description]
        """
        l, u = I
        self._r = (u - l) / 2
        self._xc = l + self._r

    def copy(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        E = ell1d([self._xc - self._r, self._xc + self._r])
        return E

    @property
    def xc(self):
        """[summary]

        Returns:
            float: [description]
        """
        return self._xc

    @xc.setter
    def xc(self, x):
        """[summary]

        Arguments:
            x (float): [description]
        """
        self._xc = x

    def update(self, cut):
        """Update ellipsoid core function using the cut
                g' * (x - xc) + beta <= 0

        Arguments:
            g (floay): cut
            beta (array or scalar): [description]

        Returns:
            status: 0: success
            tau: "volumn" of ellipsoid
        """
        g, beta = cut
        # TODO handle g == 0
        tau = abs(self._r * g)
        tsq = tau**2
        if beta == 0:
            self._r /= 2
            self._xc += -self._r if g > 0 else self._r
            return CUTStatus.success, tsq
        if beta > tau:
            return CUTStatus.nosoln, tsq  # no sol'n
        if beta < -tau:  # unlikely
            return CUTStatus.noeffect, tsq  # no effect

        bound = self._xc - beta / g
        upper = bound if g > 0 else self._xc + self._r
        lower = self._xc - self._r if g > 0 else bound
        self._r = (upper - lower) / 2
        self._xc = lower + self._r
        return CUTStatus.success, tsq
