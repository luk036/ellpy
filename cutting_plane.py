# -*- coding: utf-8 -*-
# import numpy as np  # Can move to below???


def cutting_plane_feas(assess, S, max_it=1000, tol=1e-8):
    '''
    Cutting-plane method for solving convex feasibility problem
    input
             assess        perform assessment on x0
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
    for niter in range(1, max_it):
        g, h, flag = assess(S.xc)
        if flag == 1:  # feasible sol'n obtained
            break
        status, tau = S.update(g, h)
        if status != 0:
            break
        if tau < tol:
            status = 2
            break
    return S.xc, niter, flag, status


def cutting_plane_dc(assess, S, t, max_it=1000, tol=1e-8):
    '''
    Cutting-plane method for solving convex optimization problem
    input
             assess        perform assessment on x0
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
    for niter in range(1, max_it):
        g, h, t1 = assess(S.xc, t)
        if t != t1:  # best t obtained
            flag = 1
            t = t1
            x_best = S.xc
        status, tau = S.update(g, h)
        if status == 1:
            break
        if tau < tol:
            status = 2
            break
    return x_best, t, niter, flag, status


def cutting_plane_q(assess, S, t, max_it=1000, tol=1e-8):
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
    for niter in range(1, max_it):
        g, h, t1, x, loop = assess(S.xc, t, 0 if status != 3 else 1)
        if status != 3:
            if loop == 1:  # discrete sol'n
                h += g.dot(x - S.xc)
        else:  # can't cut in the previous iteration
            if loop == 0:  # no more alternative cut
                break
            h += g.dot(x - S.xc)
        if t != t1:  # best t obtained
            flag = 1
            t = t1
            x_best = x.copy()
        status, tau = S.update(g, h)
        if status == 1:
            break
        if tau < tol:
            status = 2
            break
    return x_best, t, niter, flag, status
