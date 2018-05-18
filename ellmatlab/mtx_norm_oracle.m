classdef mtx_norm_oracle < handle
	properties
        C
	end
	methods
		function obj = mtx_norm_oracle(Fs)
            % Formulate matrix norm minimization as semidefinite programming
            % Reference: 
            %   L. Vandenberghe and S. Boyd. Semidefinite Programming, 
            %   SIAM Review, Vol. 38, No. 1., 1996, p.55
            %  min norm \| A(x) \| can be cast as
            %        min    t
            %        s.t.   [t*I    A(x)
            %                A(x)'  t*I ] >= 0
			n = size(Fs{1},1);
            nc = length(Fs);
            Zero = zeros(n,n);
            
            % F{1} = I; % t
            for i=1:nc,
              F{i} = [Zero     Fs{i}
                      Fs{i}'  Zero];
            end
            obj.C = lmi_oracle(F);
        end

		function [g, fj, R] = chk_spd(obj, x, t)
            % F2 = obj.F;
            % Fn = F2{end};
            % Fn = Fn + t*eye(size(Fn));
            % F2{end} = Fn;
            %C = lmi_oracle(F2);
            [g, fj, R] = obj.C.chk_spd_t(x, t);
        end

		function [g, fj), t] = evaluate(obj, x, t)
            n = length(x);
            g = zeros(n,1);
            [g, fj, R] = obj.C.chk_spd_t(x, t);
            if fj > 0, return; end;
            [dt, v] = eig_min(R'*R, t);
            for i=1:n,
                g(i) = -v'*obj.F{i}*v;
            end
            t = t-dt;
            fj = 0;
        end
	end
end
