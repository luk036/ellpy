# -*- coding: utf-8 -*-


class Options:
    max_it = 2000
    tol = 1e-8


class CInfo:
    value = None
    val = None

    def __init__(self, feasible, num_iters, status):
        self.feasible = feasible
        self.num_iters = num_iters
        self.status = status


def cutting_plane_feas(Omega, S, options=Options()):
    """Find a point in a convex set (defined through a cutting-plane oracle).

    Description:
        A function f(x) is *convex* if there always exist a g(x)
        such that f(z) >= f(x) + g(x)^T * (z - x), forall z, x in dom f.
        Note that dom f does not need to be a convex set in our definition.
        The affine function g^T (x - xc) + beta is called a cutting-plane,
        or a ``cut'' for short.
        This algorithm solves the following feasibility problem:

                find x
                s.t. f(x) <= 0,

        A *separation oracle* asserts that an evalution point x0 is feasible,
        or provide a cut that separates the feasible region and x0.

    Arguments:
        Omega {[type]} -- perform assessment on x0
        S {[type]} -- Initial search space known to contain x*

    Keyword Arguments:
        options {[type]} -- [description] (default: {Options()})

    Returns:
        x -- solution vector
        niter -- number of iterations performed
    """
    feasible = False
    status = 0
    for niter in range(options.max_it):
        cut = Omega(S.xc)  # query the oracle at S.xc
        if cut is None:  # feasible sol'n obtained
            feasible = True
            break
        status, tsq = S.update(cut)  # update S
        if status != 0:
            break
        if tsq < options.tol:
            status = 2
            break

    return CInfo(feasible, niter + 1, status)


def cutting_plane_dc(Omega, S, t, options=Options()):
    """Cutting-plane method for solving convex optimization problem

    Arguments:
        Omega {[type]} -- perform assessment on x0
        S {[type]} -- Search Space containing x*
        t {[type]} -- initial best-so-far value

    Keyword Arguments:
        options {[type]} -- [description] (default: {Options()})

    Returns:
        x_best {[type]} -- solution vector
        t {[type]} -- best-so-far optimal value
        niter {[type]} -- number of iterations performed
    """
    feasible = False  # no sol'n
    x_best = S.xc
    for niter in range(options.max_it):
        cut, t1 = Omega(S.xc, t)
        if t != t1:  # best t obtained
            feasible = True
            t = t1
            x_best = S.xc
        status, tsq = S.update(cut)
        if status != 0:
            break
        if tsq < options.tol:
            status = 2
            break

    ret = CInfo(feasible, niter + 1, status)
    ret.val = x_best
    ret.value = t
    return ret


def cutting_plane_q(Omega, S, t, options=Options()):
    """Cutting-plane method for solving convex discrete optimization problem

    Arguments:
        Omega {[type]} -- perform assessment on x0
        S {[type]} -- Search Space containing x*
        t {[type]} -- initial best-so-far value

    Keyword Arguments:
        options {[type]} -- [description] (default: {Options()})

    Returns:
        x_best {[type]} -- solution vector
        t {[type]} -- best-so-far optimal value
        niter {[type]} -- number of iterations performed
    """
    '''
    Cutting-plane method for solving convex discrete optimization problem
    input
             Omega        perform assessment on x0
             S(xc)         Search space containing x*
             t             best-so-far optimal sol'n
             max_it        maximum number of iterations
             tol           error tolerance
    output
             x             solution vector
             niter         number of iterations performed
    '''
    feasible = False  # no sol'n
    # x_last = S.xc
    x_best = S.xc
    status = 1  # new
    for niter in range(options.max_it):
        cut, x0, t1, loop = Omega(S.xc, t, 0 if status != 3 else 1)
        g, h = cut
        # if status != 3:
        #     if loop == 1:  # discrete sol'n
        #         h += g.dot(x0 - S.xc)
        # else:  # can't cut in the previous iteration
        if status == 3:
            if loop == 0:  # no more alternative cut
                break
            # h += g.dot(x0 - S.xc)
        if t != t1:  # best t obtained
            feasible = True
            t = t1
            x_best = x0.copy()

        status, tsq = S.update((g, h))
        if status == 1:
            break
        if tsq < options.tol:
            status = 2
            break

    ret = CInfo(feasible, niter + 1, status)
    ret.val = x_best
    ret.value = t
    return ret


def bsearch(Omega, I, options=Options()):
    """[summary]

    Arguments:
        Omega {[type]} -- [description]
        I {[type]} -- interval (initial search space)

    Keyword Arguments:
        options {[type]} -- [description] (default: {Options()})

    Returns:
        [type] -- [description]
    """
    # assume monotone
    # feasible = False
    lower, upper = I
    u_orig = upper
    for niter in range(options.max_it):
        t = lower + (upper - lower) / 2
        if Omega(t):  # feasible sol'n obtained
            # feasible = True
            upper = t
        else:
            lower = t
        tau = (upper - lower) / 2
        if tau < options.tol:
            break

    feasible = (upper != u_orig)
    ret = CInfo(feasible, niter + 1, None)
    ret.value = upper
    return ret


class bsearch_adaptor:
    def __init__(self, P, S, options=Options()):
        """[summary]

        Arguments:
            P {[type]} -- [description]
            S {[type]} -- [description]

        Keyword Arguments:
            options {[type]} -- [description] (default: {Options()})
        """
        self.P = P
        self.S = S
        self.options = options

    @property
    def x_best(self):
        """[summary]

        Returns:
            [type] -- [description]
        """
        return self.S.xc

    def __call__(self, t):
        """[summary]

        Arguments:
            t {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        S = self.S.copy()
        self.P.update(t)
        ell_info = cutting_plane_feas(self.P, S, self.options)
        if ell_info.feasible:
            self.S.xc = S.xc
            return True
        return False
