classdef lmi2_oracle < handle
    % Check constraint 0 < F(x) < U, F{0} = 0
	properties
	   Fs, U
	end
	methods
		function obj = lmi2_oracle(Fs, U)
            obj.Fs = Fs;
			obj.U = U;
        end

        % Return a parallel cut if constraints are not satisfied.
        % Otherwise return a triangular matrix R such that F(x) = R'*R 
		function [g, f, R] = chk_spd(obj, x)
			n = length(x); 
			Sig = obj.Fs{1}*x(1);
			for i=2:n,
				Sig = Sig + obj.Fs{i}*x(i);
            end
			g = zeros(n,1); f = -1;
			A = obj.U - Sig;
			[R, p] = chol_ext(A);
			if (p ~= 0),
				e = zeros(p,1);
				e(p) = 1;
				v = R\e;
				f1 = v'*Sig(1:p,1:p)*v;
				f2 = v'*obj.U(1:p,1:p)*v;
				f = [f1 - f2, f1];
                %f = f1 - 2*f2;
				for i=1:n,
					g(i) = v'*obj.Fs{i}(1:p,1:p)*v;
				end
				return;
			end

			[R, p] = chol_ext(Sig);
			if (p ~= 0),
				e = zeros(p,1);
				e(p) = 1;
				v = R\e;
				f1 = v'*Sig(1:p,1:p)*v;
				f2 = v'*obj.U(1:p,1:p)*v;
				f = [-f1, -f1 + f2];
                %f = -f1;
				for i=1:n,
					g(i) = -v'*obj.Fs{i}(1:p,1:p)*v;
				end
				return;
            end
        end
        
        function [g, fj), t] = evaluate(obj, x, t)
            [g, fj] = chk_spd(obj, x);
            if fj<0, t = t-1; end; % to record x
        end
	end
end
