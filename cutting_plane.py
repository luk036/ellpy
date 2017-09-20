# -*- coding: utf-8 -*-
import math
import numpy as np

def cutting_plane_fea(assess, S, t, max_it=1000, tol=1e-8):
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
             iter          number of iterations performed
    '''
    flag = 0
    for iter in range(1, max_it):
        g, h, t1 = assess(S.xc, t)
        if t != t1: # feasible sol'n obtained
            flag = 1 
            break
        status, tau = S.update(g, h)
        if status == 1: break
        if tau < tol:
            status = 2
            break
    return S.xc, iter, flag, status


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
             iter          number of iterations performed
    '''
    flag = 0 # no sol'n
    for iter in range(1, max_it):
        g, h, t1 = assess(S.xc, t)
        if t != t1: # best t obtained
            flag, t = 1, t1
            x_best = np.array(S.xc)
        status, tau = S.update(g, h)
        if status == 1: break
        if tau < tol:
            status = 2
            break
    return x_best, t, iter, flag, status


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
             iter          number of iterations performed
    '''
    flag = 0 # no sol'n
    # x_last = np.array(S.xc)
    x_best = np.array(S.xc)
    status = 1 # new
    for iter in range(1, max_it):
        if status != 3:
            g, h, t1, x, loop = assess(S.xc, t, 0)
            if loop == 1: # discrete sol'n
                h += np.dot(g, x - S.xc)
        else: # can't cut in the previous iteration
            g, h, t1, x, loop = assess(S.xc, t, 1)
            if loop == 0: # no more alternative cut
                break  
            h += np.dot(g, x - S.xc)
        if t != t1: # best t obtained
            flag, t = 1, t1
            x_best = np.array(x)
        status, tau = S.update(g,h)
        if status == 1: break
        if tau < tol:
            status = 2
            break
    return x_best, t, iter, flag, status



