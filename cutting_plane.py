# -*- coding: utf-8 -*-
# import numpy as np  # Can move to below???


class Options:
    max_it = 2000
    tol = 1e-4


def bsearch(evaluate, I, options=Options()):
    # assume monotone
    flag = 0
    l, u = I
    t = l + (u - l)/2
    for niter in range(options.max_it):
        if evaluate(t):  # feasible sol'n obtained
            flag = 1
            u = t
        else:
            l = t
        tau = (u - l)/2
        t = l + tau
        if tau < options.tol:
            break
    return u, niter+1, flag


class bsearch_adaptor:
    def __init__(self, P, E, options=Options()):
        self.P = P
        self.E = E
        self.options = options

    @property
    def x_best(self):
        return self.E.xc

    def __call__(self, t):
        E = self.E.copy()
        self.P.update(t)
        x, _, flag, _ = cutting_plane_feas(
            self.P, E, self.options)
        if flag == 1:
            self.E._xc = x.copy()
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
    flag = 0
    status = 0
    for niter in range(options.max_it):
        cut, flag = evaluate(S.xc)
        if flag == 1:  # feasible sol'n obtained
            break
        status, tau = S.update(cut)
        if status != 0:
            break
        if tau < options.tol:
            status = 2
            break
    return S.xc, niter+1, flag, status


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
    flag = 0  # no sol'n
    x_best = S.xc
    for niter in range(options.max_it):
        cut, t1 = evaluate(S.xc, t)
        if t != t1:  # best t obtained
            flag = 1
            t = t1
            x_best = S.xc
        status, tau = S.update(cut)
        if status == 1:
            break
        if tau < options.tol:
            status = 2
            break
    return x_best, t, niter+1, flag, status


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
    flag = 0  # no sol'n
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
            flag = 1
            t = t1
            x_best = x0.copy()
        status, tau = S.update((g, h))
        if status == 1:
            break
        if tau < options.tol:
            status = 2
            break
    return x_best, t, niter+1, flag, status
