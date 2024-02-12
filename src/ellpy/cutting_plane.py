# -*- coding: utf-8 -*-
from enum import Enum
from typing import Any, Callable, Tuple


class CUTStatus(Enum):
    success = 0
    nosoln = 1
    smallenough = 2
    noeffect = 3


class Options:
    max_it: int = 2000  # maximum number of iterations
    tol: float = 1e-8  # error tolerance


class CInfo:
    def __init__(self, feasible: bool, num_iters: int, status: CUTStatus):
        """Construct a new CInfo object

        Arguments:
            feasible (bool): [description]
            num_iters (int): [description]
            status (int): [description]
        """
        self.feasible: bool = feasible
        self.num_iters: int = num_iters
        self.status: CUTStatus = status


def cutting_plane_feas(Omega: Callable[[Any], Any], S, options=Options()) -> CInfo:
    """Find a point in a convex set (defined through a cutting-plane oracle).

    Description:
        A function f(x) is *convex* if there always exist a g(x)
        such that f(z) >= f(x) + g(x)' * (z - x), forall z, x in dom f.
        Note that dom f does not need to be a convex set in our definition.
        The affine function g' (x - xc) + beta is called a cutting-plane,
        or a ``cut'' for short.
        This algorithm solves the following feasibility problem:

                find x
                s.t. f(x) <= 0,

        A *separation oracle* asserts that an evalution point x0 is feasible,
        or provide a cut that separates the feasible region and x0.

    Arguments:
        Omega ([type]): perform assessment on x0
        S ([type]): Initial search space known to contain x*

    Keyword Arguments:
        options ([type]): [description] (default: {Options()})

    Returns:
        x: solution vector
        niter: number of iterations performed
    """
    feasible = False
    status = CUTStatus.success
    for niter in range(options.max_it):
        cut = Omega(S.xc)  # query the oracle at S.xc
        if cut is None:  # feasible sol'n obtained
            feasible = True
            break
        cutstatus, tsq = S.update(cut)  # update S
        if cutstatus != CUTStatus.success:
            status = cutstatus
            break
        if tsq < options.tol:
            status = CUTStatus.smallenough
            break
    return CInfo(feasible, niter + 1, status)


def cutting_plane_dc(
    Omega: Callable[[Any, Any], Any], S, t, options=Options()
) -> Tuple[Any, Any, CInfo]:
    """Cutting-plane method for solving convex optimization problem

    Arguments:
        Omega ([type]): perform assessment on x0
        S ([type]): Search Space containing x*
        t (float): initial best-so-far value

    Keyword Arguments:
        options ([type]): [description] (default: {Options()})

    Returns:
        x_best (Any): solution vector
        t: final best-so-far value
        ret {CInfo}
    """
    t_orig = t  # const
    x_best = None
    status = CUTStatus.success

    for niter in range(options.max_it):
        cut, t1 = Omega(S.xc, t)
        if t1 is not None:  # better t obtained
            t = t1
            x_best = S.xc
        cutstatus, tsq = S.update(cut)
        if cutstatus != CUTStatus.success:
            status = cutstatus
            break
        if tsq < options.tol:
            status = CUTStatus.smallenough
            break
    ret = CInfo(t != t_orig, niter + 1, status)
    return x_best, t, ret


def cutting_plane_q(Omega, S, t, options=Options()):
    """Cutting-plane method for solving convex discrete optimization problem

    Arguments:
        Omega ([type]): perform assessment on x0
        S ([type]): Search Space containing x*
        t (float): initial best-so-far value

    Keyword Arguments:
        options ([type]): [description] (default: {Options()})

    Returns:
        x_best (float): solution vector
        t (float): best-so-far optimal value
        niter ([type]): number of iterations performed
    """
    # x_last = S.xc
    t_orig = t  # const
    x_best = None
    status = CUTStatus.nosoln

    for niter in range(options.max_it):
        retry = status == CUTStatus.noeffect
        cut, x0, t1, more_alt = Omega(S.xc, t, retry)
        if t1 is not None:  # better t obtained
            t = t1
            x_best = x0.copy()
        status, tsq = S.update(cut)
        if status == CUTStatus.noeffect:
            if not more_alt:  # no more alternative cut
                break
        if status == CUTStatus.nosoln:
            break
        if tsq < options.tol:
            status = CUTStatus.smallenough
            break

    ret = CInfo(t != t_orig, niter + 1, status)
    return x_best, t, ret


def bsearch(
    Omega: Callable[[Any], bool], Interval: Tuple, options=Options()
) -> Tuple[Any, CInfo]:
    """[summary]

    Arguments:
        Omega ([type]): [description]
        I ([type]): interval (initial search space)

    Keyword Arguments:
        options ([type]): [description] (default: {Options()})

    Returns:
        [type]: [description]
    """
    # assume monotone
    # feasible = False
    lower, upper = Interval
    T = type(upper)  # T could be `int` or `Fraction`
    u_orig = upper
    status = CUTStatus.success

    for niter in range(options.max_it):
        tau = (upper - lower) / 2
        if tau < options.tol:
            status = CUTStatus.smallenough
            break
        t = T(lower + tau)
        if Omega(t):  # feasible sol'n obtained
            upper = t
        else:
            lower = t

    ret = CInfo(upper != u_orig, niter, status)
    return upper, ret


class bsearch_adaptor:
    def __init__(self, P, S, options=Options()):
        """[summary]

        Arguments:
            P ([type]): [description]
            S ([type]): [description]

        Keyword Arguments:
            options ([type]): [description] (default: {Options()})
        """
        self.P = P
        self.S = S
        self.options = options

    @property
    def x_best(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.S.xc

    def __call__(self, t):
        """[summary]

        Arguments:
            t (float): the best-so-far optimal value

        Returns:
            [type]: [description]
        """
        S = self.S.copy()
        self.P.update(t)
        ell_info = cutting_plane_feas(self.P, S, self.options)
        if ell_info.feasible:
            self.S.xc = S.xc
        return ell_info.feasible
