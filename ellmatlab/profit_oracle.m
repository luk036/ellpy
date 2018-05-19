classdef profit_oracle < handle
    properties
		log_k, v, a, log_pA
	end
	methods
		function obj = profit_oracle(p, A, k, alpha, beta, v1, v2)
			obj.log_pA = log(p*A);
			obj.log_k = log(k);
			obj.v = [v1 v2]';
			obj.a = [alpha beta]';
		end
		
		function [g, fj), t] = evaluate(obj, y, t)
			fj = y(1) - obj.log_k; % constraint
			if (fj > 0), g = [1 0]'; return; end
			log_Cobb = obj.log_pA + obj.a'*y;
			x = exp(y);
			te = t + obj.v'*x;
			fj = log(te) - log_Cobb;
			if (fj < 0),
				t = exp(log_Cobb) - obj.v'*x;
				te = t + obj.v'*x;
				fj = 0;
			end
			g = (obj.v .* x)/te - obj.a;			
		end
	end
end

