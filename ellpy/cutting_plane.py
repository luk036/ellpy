# -*- coding: utf-8 -*-


class Options:
    max_it = 2000
    tol = 1e-8


def bsearch(evaluate, I, options=Options()):
    # assume monotone
    feasible = False
    l, u = I
    t = l + (u - l)/2
    for niter in range(options.max_it):
        if evaluate(t):  # feasible sol'n obtained
            feasible = True
            u = t
        else:
            l = t
        tau = (u - l)/2
        t = l + tau
        if tau < options.tol:
            break
    return u, niter+1, feasible


class bsearch_adaptor:
    def __init__(self, P, S, options=Options()):
        self.P = P
        self.S = S
        self.options = options

    @property
    def x_best(self):
        return self.S.xc

    def __call__(self, t):
        S = self.S.copy()
        self.P.update(t)
        x, _, feasible, _ = cutting_plane_feas(
            self.P, S, self.options)
        if feasible:
            self.S._xc = x.copy()
            return True
        return False


def cutting_plane_feas(evaluate, S, options=Options()):
    '''
    Cutting-plane method for solving convex feasibility problem
    input
             evaluate      perform assessment on x0
             S(xc)         Search Space containing x*
             t             best-so-far optimal sol'n
             max_it        maximum number of iterations
             tol           error tolerance
    output
             x             solution vector
             niter         number of iterations performed
    '''
    feasible = False
    status = 0
    for niter in range(options.max_it):
        cut, feasible = evaluate(S.xc)
        if feasible:  # feasible sol'n obtained
            break
        status, tsq = S.update(cut)
        if status != 0:
            break
        if tsq < options.tol:
            status = 2
            break
    return S.xc, niter+1, feasible, status


def cutting_plane_dc(evaluate, S, t, options=Options()):
    '''
    Cutting-plane method for solving convex optimization problem
    input
             evaluate      perform assessment on x0
             S(xc)         Search Space containing x*
             t             initial best-so-far value
             max_it        maximum number of iterations
             tol           error tolerance
    output
             x_best        solution vector
             t             best-so-far optimal value
             niter         number of iterations performed
    '''
    feasible = False  # no sol'n
    x_best = S.xc
    for niter in range(options.max_it):
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
    return x_best, t, niter+1, feasible, status


def cutting_plane_q(evaluate, S, t, options=Options()):
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
    for niter in range(options.max_it):
        cut, t1, loop = evaluate(
            S.xc, t, 0 if status != 3 else 1)
        g, h, x0 = cut
        if status != 3:
            if loop == 1:  # discrete sol'n
                h += g.dot(x0 - S.xc)
        else:  # can't cut in the previous iteration
            if loop == 0:  # no more alternative cut
                break
            h += g.dot(x0 - S.xc)
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
    return x_best, t, niter+1, feasible, status
