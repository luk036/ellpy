classdef lsq_corr_2d_oracle < handle
	properties
        C1
        C2
	end
	methods
		function obj = lsq_corr_2d_oracle(Y, s, nn, degree)
            n = size(s,1);
            disp('constructing F...');
            % Convert to sparse matrices, make cvx solver more efficient???
			F = zeros(n,n,nn);
			for i=1:n,
			  for j=1:n,
			    h = s(j,:) - s(i,:);
                if h(1) < 0, h = -h; end;
			    Cx = h(1).^(0:degree(1));
			    Cy = h(2).^(0:degree(2));
			    C = kron(Cy, Cx); % Kronecker product
			    F(i,j,:) = C;
			  end
			end

			% Convert to sparse matrices
			Fs = cell(1, nn);
			for i=1:nn
			  Fs{i} = F(:,:,i);
			end
			clear F;
            %Fs = [Fs, eye(n)]; % for kappa
            Fs = [Fs, zeros(n,n)];
            
            % Formulate matrix norm minimization as semidefinite programming
            % Reference: 
            %   L. Vandenberghe and S. Boyd. Semidefinite Programming, 
            %   SIAM Review, Vol. 38, No. 1., 1996, p.55
            %  min norm \| A(x) \| can be cast as
            %        min    t
            %        s.t.   [t*I    A(x)
            %                A(x)'   t*I ]
            I = speye(n*2);
            Zero = zeros(n,n);
            
            F{1} = I; % t
            for i=2:nn+2,
              F{i} = [Zero     Fs{i-1}
                      Fs{i-1}'  Zero];
            end
            F = [F, [Zero     -Y
                       -Y'     Zero]];
            %c = zeros(nn+1,1);
            %c(1) = 1;  % i.e. t
            Fs = [Fs, zeros(n,n)];
            
            obj.C1 = lmi_oracle(Fs);
            obj.C2 = lmi_oracle(F);
		end

		function [g, fj), t] = assess(obj, x, t)
            f0 = x(1);
            fj = f0 - t;
            g = zeros(size(x));
            if fj >= 0, 
                g(1) = 1;
                return;
            end
            if x(end) < 0, % kappa > 0
                fj = -x(end); 
                g(end) = -1;
                return
            end
            
            [g, fj] = obj.C1.chk_spd(x(2:end));
            if fj > 0, g=[0; g]; return; end;
            [g, fj] = obj.C2.chk_spd(x);
            if fj > 0, return; end;

            t = f0; fj = 0;
            g = zeros(size(x));
            g(1) = 1;
        end
	end
end
