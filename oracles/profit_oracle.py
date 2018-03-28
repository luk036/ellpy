# -*- coding: utf-8 -*-
import numpy as np

class profit_oracle:
	def __init__(self, p, A, alpha, beta, v1, v2, k):
		self.log_pA = np.log(p*A)
		self.log_k = np.log(k)
		self.v = np.array([v1, v2])
		self.a = np.array([alpha, beta])

	def __call__(self, y, t):
		fj = y[0] - self.log_k # constraint
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


class profit_rb_oracle:
	def __init__(self, p, A, alpha, beta, \
				v1, v2, k, ui, e1, e2, e3):
		self.uie1 = ui*e1
		self.uie2 = ui*e2
		self.log_pA = np.log((p-ui*e3)*A)
		self.log_k = np.log(k-ui*e3)
		self.v = np.array([v1+ui*e3, v2+ui*e3])
		self.a = np.array([alpha, beta])

	def __call__(self, y, t):
		fj = y[0] - self.log_k # constraint
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


class profit_q_oracle(profit_oracle):
	def __init__(self, p, A, alpha, beta, v1, v2, k):
		profit_oracle.__init__(self, p, A, alpha, beta, v1, v2, k)

	def __call__(self, y, t, retry):
		x = np.round(np.exp(y))
		if x[0] == 0.0: 
			x[0] = 1.0
		if x[1] == 0.0: 
			x[1] = 1.0
		yd = np.log(x)
		g, fj, t = profit_oracle.__call__(self, yd, t)
		return g, fj, t, yd, 1
