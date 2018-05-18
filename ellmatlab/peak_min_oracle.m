classdef peak_min_oracle < handle
    properties
        Apc
        Asc
        Anrc
        Lpsq
        Upsq
        Spsq
        iL
        iD
        F
        N
        i_Anr
        i_As
        i_Ap
        i_diag
    end    
    
	methods
        function obj = peak_min_oracle(N, Apc, Asc, Anrc, Lpsq, Upsq, Spsq, iL)   
            obj.N = N;
            
            obj.Apc = Apc;
            obj.Asc = Asc;
            obj.Anrc = Anrc;
            obj.Lpsq = Lpsq;
            obj.Upsq = Upsq;
            obj.Spsq = Spsq;
            
            % for round robin counters
            obj.i_Anr = 0;
            obj.i_As = 0;
            obj.i_Ap = 0;
            obj.i_diag = 0;
            
            obj.iL = iL;
            obj.iD = zeros(N,1);
            k = 1;
            tm = N;
            for i=1:N,
                obj.iD(i) = k;
                k = k + tm;
                tm = tm - 1;
            end

            % Construct F{i}, i=1,...,M
            % Note that F{0} = 0
            k = 1;
            % Note F0 == 0;
            for i=1:N, 
                for j=i:N,
                    Qi = spalloc(N,N,2);
                    Qi(j,i) = 1;
                    Qi(i,j) = 1;	
                    F{k} = Qi;
                    k = k+1;
                end
            end
            obj.F = F;
        end
            
        function [g, f, t, x, loop] = evaluate(obj, x0, t)
            G = toMat(obj, x0, obj.N);
            m = length(x0);
            x = x0;
            for loop = 0:1,
                % 2. passband constraints
                [n,~] = size(obj.Apc);
                k = obj.i_Ap;
                for i=1:n,
                    k = k + 1;
                    if k > n, k = 1; end    % round robin
                    v = obj.Apc(k,:)*x;
                    if (v > obj.Upsq),
                        %f = v - Upsq ;
                        g = obj.Apc(k,:)'; 
                        f = [v - obj.Upsq, v - obj.Lpsq];
                        obj.i_Ap = k;
                        return;
                    end
                    if (v < obj.Lpsq),
                        %f = Lpsq - v;
                        g = -obj.Apc(k,:)'; 
                        f = [-v + obj.Lpsq, -v + obj.Upsq];
                        obj.i_Ap = k;
                        return;
                    end
                end

                % 3. stopband constraint
                [n,~] = size(obj.Asc);
                k = obj.i_As;
                for i=1:n,
                    k = k + 1;
                    if k > n, k = 1; end        
                    v = obj.Asc(k,:)*x;
                    if (v > obj.Spsq),
                        %f = v - Spsq ;
                        g = obj.Asc(k,:)'; 
                        f = [v - obj.Spsq, v];
                        obj.i_As = k;
                        return;
                    end
                    if (v < 0),
                        %f = v - Spsq ;
                        g = -obj.Asc(k,:)'; 
                        f = [-v, -v + obj.Spsq];
                        obj.i_As = k;
                        return;
                    end
                end

                % 1. nonnegative-real constraint
                [n,~] = size(obj.Anrc);
                k = obj.i_Anr;
                for i=1:n,
                    k = k + 1;
                    if k > n, k = 1; end
                    v = obj.Anrc(k,:)*x;
                    if (v < 0),
                        f = -v;
                        g = -obj.Anrc(k,:)';
                        obj.i_Anr = k;
                        return;
                    end
                end
    
                % 5. max(diag(G)) <= t
                v = diag(G);
                k = obj.i_diag;
                for i=1:obj.N,
                    k = k + 1;
                    if k > obj.N, k = 1; end        
                    if (v(k) > t),
                        f = v(k) - t ;
                        g = zeros(m,1);
                        g(obj.iD(k)) = 1;
                        obj.i_diag = k;
                        return;
                    end
                end

                if loop == 1, break, end;
    
                % 4. G >= 0 constraint
                [R, p] = chol_ext(G);
                if (p ~= 0),
                    e = zeros(p,1);
                    e(p) = 1;
                    v = R\e;
                    f = -v'*G(1:p,1:p)*v;
                    g = zeros(m,1);
                    for i=1:m,
                        g(i) = -v'*obj.F{i}(1:p,1:p)*v;
                    end
                    return;
                end
	
                [v,d,~] = svd(G);
                h = v(:,1)*sqrt(d(1,1));	
                %[v,d] = eigs(G,1);
                %h = v * sqrt(d);
                G1 = h*h';
                %norm(G - G1,'fro'),
                G = G1;
                x = G(obj.iL);
                s1 = obj.Lpsq/min(obj.Apc*x);
                s2 = min([obj.Upsq/max(obj.Apc*x), obj.Spsq/max(obj.Asc*x)]);
                %if s1 <= s2, 
                %    s = s1;
                %else
                    s = (s1 + s2)/2;
                %end
                x = s*x; % rescale to the closest point
                G = toMat(obj, x0, obj.N);
            end

            % Begin objective function
            v = diag(G);
            [t, imax] = max(v); % update best so far t
            disp(t);
            f = 0;
            %g = zeros(length(x), 1);
            g = zeros(m,1);
            g(obj.iD(imax)) = 1;
        end

        function G = toMat(obj, x, N)
            G = zeros(N,N);
            G(obj.iL) = x;
            G = G';
            G(obj.iL) = x;
        end
    end
end
