classdef lsq_corr_oracle < handle
	properties
        C1
        C2
	end
	methods
		function obj = lsq_corr_oracle(Y, s, nn)
            n = size(s,1);
            disp('constructing F...');
            % Convert to sparse matrices, make cvx solver more efficient???
            Fs{1} = ones(n,n); % constant term
            Fs{2} = zeros(n,n);
            for i=1:n,
              for j=i+1:n,
                  d = s(j,:)' - s(i,:)';
             	  Fs{2}(i,j) = norm(d);
             	  Fs{2}(j,i) = Fs{2}(i,j);
              end
            end
            % Fs{2} = Fs{2} .* Fs{2};
            for i=3:nn,
                Fs{i} = Fs{i-1}.*Fs{2};
            end
            Fs{nn+1} = speye(n); % for kappa
            
            Fs1 = Fs;
            Fs1{nn+2} = sparse(n,n);
            obj.C1 = lmi_oracle(Fs1);
            Fs2 = [Fs, -Y];
            obj.C2 = mtx_norm_q_oracle(Fs2);
		end

		function [g, fj), t] = evaluate(obj, x, t)
            g = zeros(size(x));
            if x(end) < 0, % kappa > 0
                fj = -x(end); 
                g(end) = -1;
                return
            end
            
            [g, fj] = obj.C1.chk_spd(x);
            if fj > 0, return; end;
            [g, fj] = obj.C2.chk_spd(x, t);
            if fj > 0, return; end;
            t = t-1; fj = -1; % feasible
        end
	end
end
