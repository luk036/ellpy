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


def cutting_plane_feas(evaluate, S, options=Options()):
    """Cutting-plane method for solving convex feasibility problem

    Arguments:
        evaluate {[type]} -- perform assessment on x0
        S {[type]} -- Search Space containing x*

    Keyword Arguments:
        options {[type]} -- [description] (default: {Options()})

    Returns:
        x -- solution vector
        niter -- number of iterations performed
    """
    feasible = False
    status = 0
    for niter in range(1, options.max_it):
        cut, feasible = evaluate(S.xc)
        if feasible:  # feasible sol'n obtained
            break
        status, tsq = S.update(cut)
        if status != 0:
            break
        if tsq < options.tol:
            status = 2
            break

    return CInfo(feasible, niter, status)


def cutting_plane_dc(evaluate, S, t, options=Options()):
    """Cutting-plane method for solving convex optimization problem

    Arguments:
        evaluate {[type]} -- perform assessment on x0
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
    for niter in range(1, options.max_it):
        cut, t1 = evaluate(S.xc, t)
        if t != t1:  # best t obtained
            feasible = True
            t = t1
            x_best = S.xc
        status, tsq = S.update(cut)
        if status == 1:
            break
        if tsq < options.tol:
            status = 2
            break

    # return x_best, t, niter, feasible, status
    ret = CInfo(feasible, niter, status)
    ret.val = x_best
    ret.value = t
    return ret


def cutting_plane_q(evaluate, S, t, options=Options()):
    """Cutting-plane method for solving convex discrete optimization problem

    Arguments:
        evaluate {[type]} -- perform assessment on x0
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
             oracle        perform assessment on x0
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
    for niter in range(1, options.max_it):
        cut, x0, t1, loop = evaluate(
            S.xc, t, 0 if status != 3 else 1)
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

    ret = CInfo(feasible, niter, status)
    ret.val = x_best
    ret.value = t
    return ret


def bsearch(evaluate, I, options=Options()):
    """[summary]

    Arguments:
        evaluate {[type]} -- [description]
        I {[type]} -- [description]

    Keyword Arguments:
        options {[type]} -- [description] (default: {Options()})

    Returns:
        [type] -- [description]
    """
    # assume monotone
    feasible = False
    l, u = I
    t = l + (u - l)/2
    for niter in range(1, options.max_it):
        if evaluate(t):  # feasible sol'n obtained
            feasible = True
            u = t
        else:
            l = t
        tau = (u - l)/2
        t = l + tau
        if tau < options.tol:
            break
    
    ret = CInfo(feasible, niter, None)
    ret.value = u
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