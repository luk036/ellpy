classdef FIR_oracle < handle
    properties
        Ap
        As
        Anr
        Lpsq
        Upsq
        i_Anr
        i_As
        i_Ap
		count
    end    
    
	methods
        function obj = FIR_oracle(Ap, As, Anr, Lpsq, Upsq)   
            obj.Ap = Ap;
            obj.As = As;
            obj.Anr = Anr;
            obj.Lpsq = Lpsq;
            obj.Upsq = Upsq;
            
            % for round robin counters
            obj.i_Anr = 0;
            obj.i_As = 0;
            obj.i_Ap = 0;
			obj.count = 0;
        end             

        function [g, f, Spsq] = evaluate(obj, x, Spsq)
			% 1. nonnegative-real constraint			
			m = length(x);
			
			for cnt=1:4,
			  count = count + 1;
			  if count > 4, count = 1; end
			  switch obj.count
			
			case 1,
			
            if x(1) < 0,
                g = zeros(m,1);
                g(1) = -1;
                f = -x(1);
                return;
            end

% 			u = x(m:-1:1)';
% 			u(m) = 0.5*x(1)-0.00001;
% 			d = roots(u);
%             md = abs(d);
%             [mdmin, idx] = min(md);
% 			if mdmin <= 1,
%                 g = -real([0.5, d(idx).^(1:m-1)]');
%                 f = 0.00001+g'*x;
%                 return;
%             end

            case 2,
			
			% 2. passband constraints
            [n,m] = size(obj.Ap);
            k = obj.i_Ap;
            for i=1:n,
                k = k + 1;
                if k > n, k = 1; end    % round robin
                v = obj.Ap(k,:)*x;
                if v > obj.Upsq,
                    %f = v - Upsq ;
                    g = obj.Ap(k,:)'; 
                    f = [v - obj.Upsq, v - obj.Lpsq];
                    obj.i_Ap = k;
                    return;
                end
                if v < obj.Lpsq,
                    %f = Lpsq - v;
                    g = -obj.Ap(k,:)'; 
                    f = [-v + obj.Lpsq, -v + obj.Upsq];
                    obj.i_Ap = k;
                    return;
                end
            end

			case 3, 
			
            % 3. stopband constraint
            [n,~] = size(obj.As);
            k = obj.i_As;
			w = zeros(n,1);
            for i=1:n,
                k = k + 1;
                if k > n, k = 1; end        
                w(k) = obj.As(k,:)*x;
                if w(k) > Spsq,
                    %f = v - Spsq ;
                    g = obj.As(k,:)'; 
                    %f = [w(k) - Spsq, w(k)];
                    f = w(k) - Spsq;
                    obj.i_As = k;
                    return;
                end
                if w(k) < 0,
                    %f = v - Spsq ;
                    g = -obj.As(k,:)'; 
                    f = [-w(k), -w(k) + Spsq];
                    return;
                end
            end

			case 4,
            % 1. nonnegative-real constraint
            [n,~] = size(obj.Anr);
            for k=1:n,
                v = obj.Anr(k,:)*x;
                if (v < 0),
                    f = -v;
                    g = -obj.Anr(k,:)';
                    %obj.i_Anr = k;
                    return;
                end
            end

			  end  % switch
			end % for
			
            % Begin objective function
			[Spsq, imax] = max(w); % update best so far Spsq
			f = [0, w(imax)];
            %f = 0;
			g = obj.As(imax,:)';
        end
    end
end
