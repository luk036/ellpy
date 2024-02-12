# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Union

import numpy as np

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


class profit_oracle:
    """Oracle for a profit maximization problem.

    This example is taken from [Aliabadi and Salahi, 2013]

        max     p(A x1^α x2^β) − v1*x1 − v2*x2
        s.t.    x1 ≤ k

    where:

        p(A x1^α x2^β): Cobb-Douglas production function
        p: the market price per unit
        A: the scale of production
        α, β: the output elasticities
        x: input quantity
        v: output price
        k: a given constant that restricts the quantity of x1
    """

    def __init__(self, params: Tuple[float, float, float], a: Arr, v: Arr):
        """[summary]

        Arguments:
            params (Tuple[float, float, float]): p, A, k
            a (Arr): the output elasticities
            v (Arr): output price
        """
        p, A, k = params
        self.log_pA = np.log(p * A)
        self.log_k = np.log(k)
        self.v = v
        self.a = a

    def __call__(self, y: Arr, t: float) -> Tuple[Cut, Optional[float]]:
        """Make object callable for cutting_plane_dc()

        Arguments:
            y (Arr): input quantity (in log scale)
            t (float): the best-so-far optimal value

        Returns:
            Tuple[Cut, float]: Cut and the updated best-so-far value

        See also:
            cutting_plane_dc
        """
        fj = y[0] - self.log_k  # constraint
        if fj > 0.0:
            g = np.array([1.0, 0.0])
            return (g, fj), None

        log_Cobb = self.log_pA + self.a @ y
        q = self.v * np.exp(y)
        vx = q[0] + q[1]
        te = t + vx
        fj = np.log(te) - log_Cobb

        if fj < 0.0:  # feasible
            te = np.exp(log_Cobb)
            g = q / te - self.a
            return (g, 0.0), te - vx

        g = q / te - self.a
        return (g, fj), None


class profit_rb_oracle:
    """Oracle for a robust profit maximization problem.

    This example is taken from [Aliabadi and Salahi, 2013]:

        max     p'(A x1^α' x2^β') - v1'*x1 - v2'*x2
        s.t.    x1 ≤ k'

    where:

        α' = α ± e1
        β' = β ± e2
        p' = p ± e3
        k' = k ± e4
        v' = v ± e5

    See also:
        profit_oracle
    """

    def __init__(
        self,
        params: Tuple[float, float, float],
        a: Arr,
        v: Arr,
        vparams: Tuple[float, float, float, float, float],
    ):
        """[summary]

        Arguments:
            params (Tuple[float, float, float]): p, A, k
            a (Arr): the output elasticities
            v (Arr): output price
            vparams (Tuple): paramters for uncertainty
        """
        e1, e2, e3, e4, e5 = vparams
        self.a = a
        self.e = [e1, e2]
        p, A, k = params
        params_rb = p - e3, A, k - e4
        self.P = profit_oracle(params_rb, a, v + e5)

    def __call__(self, y: Arr, t: float) -> Tuple[Cut, Optional[float]]:
        """Make object callable for cutting_plane_dc()

        Arguments:
            y (Arr): input quantity (in log scale)
            t (float): the best-so-far optimal value

        Returns:
            Tuple[Cut, float]: Cut and the updated best-so-far value

        See also:
            cutting_plane_dc
        """
        a_rb = self.a.copy()
        for i in [0, 1]:
            a_rb[i] += -self.e[i] if y[i] > 0.0 else self.e[i]
        self.P.a = a_rb
        return self.P(y, t)


class profit_q_oracle:
    """Oracle for a decrete profit maximization problem.

        max     p(A x1^α x2^β) - v1*x1 - v2*x2
        s.t.    x1 ≤ k

    where:

        p(A x1^α x2^β): Cobb-Douglas production function
        p: the market price per unit
        A: the scale of production
        α, β: the output elasticities
        x: input quantity (must be integer value)
        v: output price
        k: a given constant that restricts the quantity of x1

    Raises:
        AssertionError: [description]

    See also:
        profit_oracle
    """

    yd = None

    def __init__(self, params, a, v):
        """[summary]

        Arguments:
            params (Tuple[float, float, float]): p, A, k
            a (Arr): the output elasticities
            v (Arr): output price
        """
        self.P = profit_oracle(params, a, v)

    def __call__(self, y, t, retry):
        """Make object callable for cutting_plane_q()

        Arguments:
            y (Arr): input quantity (in log scale)
            t (float): the best-so-far optimal value
            retry ([type]): unused

        Raises:
            AssertionError: [description]

        Returns:
            Tuple: Cut, t, and the actual evaluation point

        See also:
            cutting_plane_q
        """
        if not retry:
            x = np.round(np.exp(y))
            if x[0] == 0:
                x[0] = 1
            if x[1] == 0:
                x[1] = 1
            self.yd = np.log(x)

        (g, h), t = self.P(self.yd, t)
        h += g @ (self.yd - y)
        return (g, h), self.yd, t, not retry
