classdef mtx_norm_q_oracle < handle
	properties
        dp, F
	end
	methods
		function obj = mtx_norm_q_oracle(Fs)
            % Formulate matrix norm minimization as quadratic matrix
            % Inequality
            % Reference: 
            %   L. Vandenberghe and S. Boyd. Semidefinite Programming, 
            %   SIAM Review, Vol. 38, No. 1., 1996, p.55
            %  min norm \| A(x) \| can be cast as
            %        min    t
            %        s.t.   A'(x)*A(x) <= t*I
			n = size(Fs{1},1);
            nc = length(Fs);
            p = 1:n*n;
            p = reshape(p,n,n);
            obj.F = Fs;
            obj.dp = diag(p);
        end

		function [g, fj, R, v] = chk_spd(obj, x, t)
            n = length(x);
            m = size(obj.F{1},1);
            A = sparse(m,m);
            A(obj.dp) = A(obj.dp) + t;
            [g, fj, R, v] = chk_mtx(obj, A, x);
        end

 		function [g, fj, R, v] = chk_mtx(obj, A, x)
            n = length(x);
            g = zeros(n,1); fj = -1; v = 0;
            A1 = obj.F{n+1};
            for i=1:n,
                A1 = A1 + obj.F{i}*x(i);
            end
            P = A - A1'*A1;
            [R, p] = chol_ext(P);
            if (p == 0), return; end;
            e = zeros(p,1);
            e(p) = 1;
            v = R\e;
            fj = -v'*P(1:p,1:p)*v;
            Av = A1(:,1:p)*v;
            for i=1:n,
                g(i) = (obj.F{i}(:,1:p)*v)'*Av;
            end
        end

		function [g, fj), t] = assess(obj, x, t)
            n = length(x);
            g = zeros(n,1);
            [g, fj] = obj.chk_spd_t(x, t);
            if fj > 0, return; end;
            t = t - 1;
            fj = 0;
        end
	end
end
