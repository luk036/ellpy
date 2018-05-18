classdef lmi_oracle < handle
    properties
        F
        dp
	end
	methods
		function obj = lmi_oracle(F)
			obj.F = F;
            n = size(F{1},1);
            p = 1:n*n;
            p = reshape(p,n,n);
            obj.dp = diag(p);
        end

		function [g, fj, R, v] = chk_spd_t(obj, x, t)
            n = length(x);
            A = obj.F{n+1};
            A(obj.dp) = A(obj.dp) + t;
            [g, fj, R, v] = chk_mtx(obj, A, x);
        end

		function [g, fj, R, v] = chk_spd(obj, x)
            n = length(x);
            A = obj.F{n+1};
            [g, fj, R, v] = chk_mtx(obj, A, x);
        end

		function [g, fj, R, v] = chk_mtx(obj, A, x)
            n = length(x);
            g = zeros(n,1); fj = -1; v = 0;
            for i=1:n,
                A = A + obj.F{i}*x(i);
            end
            [R, p] = chol_ext(A);
            if (p == 0), return; end;
            e = zeros(p,1);
            e(p) = 1;
            v = R\e;
            fj = -v'*A(1:p,1:p)*v;
            for i=1:n,
                g(i) = -v'*obj.F{i}(1:p,1:p)*v;
            end
        end

        function [g, fj), t] = evaluate(obj, x, t)
            [g, fj] = chk_spd(obj, x);
            if fj<0, t = t-1; end; % to record x
        end
	end
end

