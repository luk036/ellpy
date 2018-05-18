classdef mle_corr_oracle < handle
	properties
		Y
        C
        Fs
	end
	methods
		function obj = mle_corr_oracle(Y, s, nn)
			n = size(s,1);
			disp('constructing F...');
			% Convert to sparse matrices, make cvx solver more efficient???
			F{1} = ones(n,n); % constant term
			for i=1:n,
				for j=1:n,
					dist = norm(s(j,:) - s(i,:));
					F{2}(i,j) = dist;
				end
			end
			for i=3:nn,
				F{i} = F{i-1}.*F{2};
            end
            F = [F, eye(n)]; % for kappa
            obj.Fs = F;
            obj.Y = Y;
			obj.C = lmi2_oracle(F, 2*Y);
		end

		function [g, f, t] = evaluate(obj, x, t)
            if x(end) < 0, % kappa > 0
                f = -x(end); 
                g = zeros(size(x)); 
                g(end) = -1;
                return
            end
            
            % Solve:
			%   minimize      log(det(Sig)) - trace(Sig\Y)
			%   subject to    0 <= \sum_1^n F{i}*x(i) + F{n} <= 2*Y
            [g, f, R] = obj.C.chk_spd(x);
            if f > 0, return; end;
            
			% f = log(det(Sig)) + trace(Sig^{-1}*Y)
			% g(f) = Sig^{-1} + Sig^{-1} Y Sig^{-1}
			invR = inv(R);
			S = invR * invR';
			SY = S*obj.Y;
			f1 = 2*sum(log(diag(R))) + trace(SY);

            f = f1 - t;
            if (f < 0),
                t = f1;
                f = 0;
            end
            n = length(x);
			g = zeros(n,1);
			for i=1:n,
				SFsi = obj.Fs{i}*S;
				g(i) = trace(SFsi) - trace(SY*SFsi);
			end
		end
	end
end
