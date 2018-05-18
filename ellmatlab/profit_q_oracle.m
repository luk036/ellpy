classdef profit_q_oracle < handle
    properties
		log_k, v, a, log_pA, retry_mode, count, mark
	end
	methods
		function obj = profit_q_oracle(p, A, alpha, beta, v1, v2, k)
			obj.log_pA = log(p*A);
			obj.log_k = log(k);
			obj.v = [v1 v2]';
			obj.a = [alpha beta]';
			obj.retry_mode = 0;
			obj.count = 0;
			obj.mark = 0;
		end
		
		function [g, fj), t, y, loop] = evaluate(obj, y, t, retry)
			if retry == 1,  % no cut in previous iteration
			  if obj.retry_mode == 0, % the first time
			    obj.mark = obj.count; % mark the current position
			    obj.retry_mode = 1; % begin the "retry" mode
			  else
			    if obj.count == obj.mark, % return to the first position 
			      loop = 0; % tell the ellipsoid method that no more retry
				  return
				end
			  end
			else
			  obj.retry_mode = 0; % not in retry mode
			end
			  
			for loop = retry:1
			  for k = 1:2,
			    obj.count = obj.count + 1;
				if obj.count > 2, obj.count = 1; end % round robin
			    switch obj.count
				case 1
			      fj = y(1) - obj.log_k; % constraint
			      if (fj > 0), g = [1 0]'; return; end
				case 2
			      log_Cobb = obj.log_pA + obj.a'*y;
			      x = exp(y);
			      te = t + obj.v'*x;
			      fj = log(te) - log_Cobb;
			      if (fj >= 0), g = (obj.v .* x)/te - obj.a; return, end
		        end % switch
			  end
			  
			  if loop == 0,
			    x = round(exp(y));
			    y = log(x);
			  else
				t = exp(log_Cobb) - obj.v'*x;
				te = t + obj.v'*x;
				fj = 0;
			    g = (obj.v .* x)/te - obj.a;			
              end
	        end
		end
	end
end

