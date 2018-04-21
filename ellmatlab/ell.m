classdef ell < handle
    % ell = { x | (x - xc)' * P^-1 * (x - xc) <= 1 }
    properties
        P, xc, c1
    end    
	methods
        function obj = ell(val, x)
            n = length(x);
            if isscalar(val), 
                obj.P = diag(val*ones(n,1));
            elseif isvector(val),
                obj.P = diag(val);
            else % ismatrix(val)
                obj.P = val;
            end
            obj.xc = x;
            obj.c1 = n*n/(n*n-1);
        end
        
		function tau = update_cc(obj, g) % central cut
            n = size(obj.xc,1);
            Pg = obj.P * g;
            tsq = g'*Pg;
            tau = sqrt(tsq);
            rho = 1/(n+1);
            sigma = 2*rho;
            delta = obj.c1;
            obj.xc = obj.xc - (rho/tau)*Pg;
            obj.P = delta*(obj.P - (sigma/tsq)* (Pg * Pg'));
        end
        
        function [status, rho, sigma, delta] = calc_dc(obj, alpha)
            rho = 0; sigma = 0; delta = 0;
            if alpha > 1, status = 1; return; end; % no sol'n
            n = size(obj.xc,1);
            if n*alpha < -1, status = 3; return; end; % no effect
            rho = (1+n*alpha)/(n+1);
            sigma = 2*rho/(1+alpha);
            delta = obj.c1*(1-alpha*alpha);
            status = 0; % okay
        end

        function [status, rho, sigma, delta] = calc_ll(obj, alpha)
            % General parallel cut support
            rho = 0; sigma = 0; delta = 0;
            if length(alpha) == 1 || alpha(2) >= 1, % deep cut
                [status, rho, sigma, delta] = calc_dc(obj, alpha(1));
                return;
            end
            if alpha(1) > alpha(2), status = 1; return; end; % no sol'n
            aprod = alpha(1) * alpha(2);
            n = size(obj.xc,1);
            if -n*aprod >= 1, status = 3; return; end; % no effect
            asq = alpha.*alpha;
            asum = alpha(1) + alpha(2);
            xi = sqrt(4*(1 - asq(1))*(1 - asq(2)) + n*n*(asq(2) - asq(1))^2);
            sigma = (n + (2*(1 + aprod - xi/2)/(asum*asum)))/(n+1);
            rho = asum * sigma/2;
            delta = obj.c1*(1 - (asq(1) + asq(2) - xi/n)/2);
            status = 0; % okay
        end        
        
		function [status, tau] = update_dc(obj, g, beta)
            [status, tau] = obj.update_core(@obj.calc_dc, g, beta);
        end

        function [status, tau] = update(obj, g, beta)
            [status, tau] = obj.update_core(@obj.calc_ll, g, beta);
        end

        function [status, tau] = update_core(obj, calc_ell, g, beta)
            Pg = obj.P*g;  
            tsq = g'*Pg;
            tau = sqrt(tsq);
            alpha = beta/tau;
            [status, rho, sigma, delta] = calc_ell(alpha);
            if status ~= 0, return; end;
            obj.xc = obj.xc - (rho/tau)*Pg;
            obj.P = delta*(obj.P - (sigma/tsq)*(Pg*Pg'));
            status = 0; % okay
		end

	end
end
