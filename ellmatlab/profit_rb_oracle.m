classdef profit_rb_oracle < handle
    properties
		log_k, v, a, log_pA
		ui, e1, e2, e3
	end
	methods
		function obj = profit_rb_oracle(p, A, alpha, beta, v1, v2, k, ui, e1, e2, e3)
			obj.ui = ui;
			obj.e1 = e1;
			obj.e2 = e2;
			obj.e3 = e3;
			obj.log_pA = log((p - ui*e3)*A);
			obj.log_k = log(k - ui*e3);
			obj.v = [v1+ui*e3 v2+ui*e3]';
			obj.a = [alpha beta]';
		end
		
		function [g, fj), t] = assess(obj, y, t)
			fj = y(1) - obj.log_k; % constraint
			if (fj > 0), g = [1 0]'; return; end

			if (y(1) > 0) 
				alpha = obj.a(1) - obj.ui*obj.e1;
			else
				alpha = obj.a(1) + obj.ui*obj.e1;
			end
			if (y(2) > 0) 
				beta = obj.a(2) - obj.ui*obj.e2;
			else
				beta = obj.a(2) + obj.ui*obj.e2;
			end
			a_rb = [alpha beta]';
			
			log_Cobb = obj.log_pA + a_rb'*y;
			x = exp(y);
			vx = obj.v'*x;
			te = t + vx;
			fj = log(te) - log_Cobb;
			if (fj < 0),
				t = exp(log_Cobb) - vx;
				te = t + vx;
				fj = 0;
			end
			g = (obj.v .* x)/te - a_rb;			
		end
	end
end

