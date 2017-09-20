# -*- coding: utf-8 -*-
import numpy as np

class network_oracle:
	def __init__(self, G, edge_fun):
		self.G = G
		self.edge_fun = edge_fun

	def __call__(self, y, t):
		fj = y[0] - self.log_k; # constraint
		if fj > 0.0:
			g = np.array([1.0, 0.0]) 
			return g, fj, t
		log_Cobb = self.log_pA + np.dot(self.a, y)
		x = np.exp(y)
		vx = np.dot(self.v, x)
		te = t + vx
		fj = np.log(te) - log_Cobb
		if fj < 0.0:
			te = np.exp(log_Cobb)
			t = te - vx
			fj = 0.0
		g = (self.v * x)/te - self.a
		return g, fj, t


class network_rb_oracle:
	def __init__(self, p, A, alpha, beta, \
				v1, v2, k, ui, e1, e2, e3):
		self.uie1 = ui*e1
		self.uie2 = ui*e2
		self.log_pA = np.log((p-ui*e3)*A)
		self.log_k = np.log(k-ui*e3)
		self.v = np.array([v1+ui*e3, v2+ui*e3])
		self.a = np.array([alpha, beta])

	def __call__(self, y, t):
		fj = y[0] - self.log_k; # constraint
		if fj > 0.0:
			g = np.array([1.0, 0.0]) 
			return g, fj, t
		a_rb = np.array(self.a)		
		a_rb[0] += self.uie1 * (+1.0 if y[0] <= 0.0 else -1.0)
		a_rb[1] += self.uie2 * (+1.0 if y[1] <= 0.0 else -1.0)

		log_Cobb = self.log_pA + np.dot(a_rb, y)
		x = np.exp(y)
		vx = np.dot(self.v, x)
		te = t + vx
		fj = np.log(te) - log_Cobb
		if fj < 0.0:
			te = np.exp(log_Cobb)
			t = te - vx
			fj = 0.0
		g = (self.v * x)/te - a_rb
		return g, fj, t


