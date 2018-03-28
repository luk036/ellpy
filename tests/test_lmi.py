# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from ..oracles.lmi_oracle import *
from ..cutting_plane import *
from ..ell import *

class my_oracle:
	def __init__(self):
		self.c = np.array([1.,-1.,1.])
		F1 = np.array([ [[-7., -11.], [-11.,  3.]], 
                        [[ 7., -18.], [-18., 8.]], 
                        [[-2.,  -8.], [-8., 1.]] ] )
		B1 = np.array([[33.,-9.],[-9.,26.]])
		F2 = np.array([ [[-21., -11.,   0.], [-11.,  10.,   8.], [  0.,   8.,  5.]],
                        [[  0.,  10.,  16.], [ 10., -10., -10.], [ 16., -10.,  3.]],
                        [[ -5.,   2., -17.], [  2.,  -6.,   8.], [-17.,   8.,  6.]] ] )
		B2 = np.array([[ 14.,   9.,  40.], [  9.,  91.,  10.], [ 40.,  10., 15.]] )
		self.lmi1 = lmi_oracle(F1, B1)
		self.lmi2 = lmi_oracle(F2, B2)

	def __call__(self, x, t):
		f0 = np.dot(self.c, x)
		fj = f0 - t
		if fj > 0.0: 
			return self.c, fj, t

		g, fj, _, _ = self.lmi1.chk_spd(x)
		if fj > 0.:
			return g, fj, t

		g, fj, _, _ = self.lmi2.chk_spd(x)
		if fj > 0.:
			return g, fj, t
		return self.c, 0.0, f0


def test_lmi():
    x0 = np.array([0., 0., 0.])  # initial x0
    fmt = '{:f} {} {} {}'
 
    E = ell(10., x0)
    P = my_oracle()
    xb, fb, iter, flag, status = cutting_plane_dc(P, E, 100.0, 200, 1e-4)
    print(fmt.format(fb, iter, flag, status))
    print(xb)
    assert flag == 1
    assert iter == 115

