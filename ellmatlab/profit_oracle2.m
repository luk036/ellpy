classdef profit_oracle2 < handle
    properties
		log_k, v, a, log_pA
	end
	methods
		function obj = profit_oracle2(p, A, alpha, beta, v1, v2, k)
			obj.log_pA = log(p*A);
			obj.log_k = log(k);
			obj.v = [v1 v2 1]';
			obj.a = [alpha beta 0]';
		end
		
		function [g, fj), t] = evaluate(obj, y, t)
			fj = y(1) - obj.log_k; % constraint
			if (fj > 0), g = [1 0 0]'; return; end
			
			log_Cobb = obj.log_pA + obj.a'*y;
			x = exp(y);
			te = obj.v'*x;
			fj = log(te) - log_Cobb;
			g = (obj.v .* x)/te - obj.a;
			if (fj > 0), return; end
			
			fj = -y(3) - t;
			if fj < 0,
				fj = 0;
                t = -y(3);
			end
			g = [0 0 -1]';
		end
	end
end

