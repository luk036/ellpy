# -*- coding: utf-8 -*-
import numpy as np


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

    def __call__(self, y, t):
        """[summary]

        Arguments:
            y {[type]} -- [description]
            t {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        fj = y[0] - self.log_k  # constraint
        if fj > 0:
            g = np.array([1., 0.])
            return (g, fj), t

        log_Cobb = self.log_pA + np.dot(self.a, y)
        x = np.exp(y)
        vx = np.dot(self.v, x)
        te = t + vx
        fj = np.log(te) - log_Cobb

        if fj < 0:
            te = np.exp(log_Cobb)
            t = te - vx
            fj = 0.
        g = (self.v * x) / te - self.a
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
        p, A, k = params
        e1, e2, e3, e4, e5 = vparams
        params_rb = p - e3, A, k - e4
        self.a = a
        self.e = [e1, e2]
        self.P = profit_oracle(params_rb, a, v + e5)

    def __call__(self, y, t):
        """[summary]

        Arguments:
            y {[type]} -- [description]
            t {[type]} -- [description]

        Returns:
            [type] -- [description]
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
            t {[type]} -- [description]
            retry {[type]} -- [description]

        Raises:
            AssertionError -- [description]

        Returns:
            [type] -- [description]
        """
        x = np.round(np.exp(y))
        if x[0] == 0 or x[1] == 0:
            raise AssertionError()
        yd = np.log(x)
        (g, h), t = self.P(yd, t)
        h += g.dot(yd - y)
        return (g, h), yd, t, 1
