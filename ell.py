# -*- coding: utf-8 -*-
import numpy as np

class ell:
    def __init__(self, val, x):
        '''ell = { x | (x - xc)' * P^-1 * (x - xc) <= 1 }'''
        n = len(x)
        self.c1 = float(n*n)/(n*n-1.0)
        self.xc = np.array(x)
        if np.isscalar(val):
            self.P = val * np.eye(n)
        else:
            self.P = np.diag(val)

    def update_core(self, calc_ell, g, beta):
        Pg = self.P.dot(g)
        tsq = g.dot(Pg)
        tau = np.sqrt(tsq)
        alpha = beta/tau
        status, rho, sigma, delta = calc_ell(alpha)
        if status != 0: 
            return status, tau
        self.xc -= (rho/tau) * Pg
        self.P -= (sigma/tsq) * np.outer(Pg, Pg)
        self.P *= delta
        return status, tau

    def calc_cc(self):
        '''central cut'''
        n = len(self.xc)
        rho = 1.0/(n+1)
        sigma = 2.0*rho
        delta = self.c1
        return 0, rho, sigma, delta

    def calc_dc(self, alpha):
        '''deep cut'''
        if alpha == 0.0: 
            return self.calc_cc()
        n = len(self.xc)
        status, rho, sigma, delta = 0, 0.0, 0.0, 0.0
        if alpha > 1.: 
            status = 1  # no sol'n
        elif n*alpha < -1.: 
            status = 3  # no effect
        else:
            rho = (1.0+n*alpha)/(n+1)
            sigma = 2.0*rho/(1.0+alpha)
            delta = self.c1*(1.0-alpha*alpha)
        return status, rho, sigma, delta

    def calc_ll(self, alpha):
        '''parallel or deep cut'''
        if np.isscalar(alpha): 
            return self.calc_dc(alpha)
        # parallel cut
        a0, a1 = alpha
        if a1 >= 1.0:
            return self.calc_dc(a0)
        n = len(self.xc)
        status, rho, sigma, delta = 0, 0.0, 0.0, 0.0
        aprod = a0 * a1
        if a0 > a1: 
            status = 1 # no sol'n
        elif n*aprod < -1.0: 
            status = 3  # no effect
        else:
            asq = alpha * alpha
            asum = a0 + a1
            asqdiff = asq[1] - asq[0]
            xi = np.sqrt(4.0*(1.0-asq[0])*(1.0-asq[1]) + n*n*asqdiff*asqdiff)
            sigma = (n + (2.0*(1.0 + aprod - xi/2.0)/(asum*asum)))/(n+1)
            rho = asum * sigma/2.0
            delta = self.c1*(1.0 - (asq[0] + asq[1] - xi/n)/2.0)        
        return status, rho, sigma, delta

    def update(self, g, beta):
        return self.update_core(self.calc_ll, g, beta)