# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np

# np.ndarray = np.ndarray
Cut = Tuple[np.ndarray, float]


class profit_oracle:
    """[summary]

    Returns:
        [type] -- [description]
    """
    def __init__(self, params, a, v):
        """[summary]

        Arguments:
            params {[type]} -- [description]
            a {[type]} -- [description]
            v {[type]} -- [description]
        """
        p, A, k = params
        self.log_pA = np.log(p * A)
        self.log_k = np.log(k)
        self.v = v
        self.a = a

    def __call__(self, y: np.ndarray, t: float) -> Tuple[Cut, float]:
        """[summary]

        Arguments:
            y {[type]} -- [description]
            t {float} -- [description]

        Returns:
            [type] -- [description]
        """
        fj = y[0] - self.log_k  # constraint
        if fj > 0.:
            g = np.array([1., 0.])
            return (g, fj), t

        log_Cobb = self.log_pA + self.a @ y
        q = self.v * np.exp(y)
        vx = q[0] + q[1]
        te = t + vx
        fj = np.log(te) - log_Cobb

        if fj < 0.:  # feasible
            te = np.exp(log_Cobb)
            t = te - vx
            fj = 0.
        g = q / te - self.a
        return (g, fj), t


class profit_rb_oracle:
    """[summary]

    Returns:
        [type] -- [description]
    """
    def __init__(self, params, a, v, vparams):
        """[summary]

        Arguments:
            params {[type]} -- [description]
            a {[type]} -- [description]
            v {[type]} -- [description]
            vparams {[type]} -- [description]
        """
        e1, e2, e3, e4, e5 = vparams
        self.a = a
        self.e = [e1, e2]
        p, A, k = params
        params_rb = p - e3, A, k - e4
        self.P = profit_oracle(params_rb, a, v + e5)

    def __call__(self, y: np.ndarray, t: float) -> Tuple[Cut, float]:
        """[summary]

        Arguments:
            y {np.ndarray} -- [description]
            t {float} -- [description]

        Returns:
            Tuple[Cut, float] -- [description]
        """
        a_rb = self.a.copy()
        for i in [0, 1]:
            a_rb[i] += self.e[i] if y[i] <= 0 else -self.e[i]
        self.P.a = a_rb
        return self.P(y, t)


class profit_q_oracle:
    """[summary]

    Raises:
        AssertionError -- [description]

    Returns:
        [type] -- [description]
    """
    def __init__(self, params, a, v):
        """[summary]

        Arguments:
            params {[type]} -- [description]
            a {[type]} -- [description]
            v {[type]} -- [description]
        """
        self.P = profit_oracle(params, a, v)

    def __call__(self, y, t, retry):
        """[summary]

        Arguments:
            y {[type]} -- [description]
            t {float} -- [description]
            retry {[type]} -- [description]

        Raises:
            AssertionError -- [description]

        Returns:
            [type] -- [description]
        """
        x = np.round(np.exp(y))
        if x[0] == 0:
            x[0] = 1
        if x[1] == 0:
            x[1] = 1
        yd = np.log(x)
        (g, h), t = self.P(yd, t)
        h += g.dot(yd - y)
        return (g, h), yd, t, 1
