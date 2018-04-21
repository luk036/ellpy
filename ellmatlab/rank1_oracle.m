classdef rank1_oracle < handle
    properties
        iL
        iD
        F
        N
        i_diag
        c
    end    
	methods
        function obj = rank1_oracle(N, A, iL)
            obj.N = N;
            obj.i_diag = 0;

            Au = triu(A,1)';
            obj.c = A(iL) + Au(iL);
            
            obj.iL = iL;
            obj.iD = zeros(N,1);
            k = 1;
            m = N;
            for i=1:N,
                obj.iD(i) = k;
                k = k + m;
                m = m - 1;
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

        function [g, f, t, x, loop] = assess(obj, x0, t, restart)
            % Begin constraints checking
            G = toMat(obj, x0, obj.N);
            n = length(x0);
            x = x0;
            for loop = restart:1,
                % 1. 1 <= trace(A*G) <= 2
                v = obj.c'*x;
                if v > 2,
                    g = obj.c;
                    f = [v-2, v-1];
                    return;
                end
                if (v < 1),
                    g = -obj.c;
                    f = [-v+1, -v+2];
                    return;
                end
	
                % 5. max(diag(G)) <= t
                v = diag(G);
                m = length(v);
                k = obj.i_diag;
                for i=1:m,
                    k = k + 1;
                    if k > m, k = 1; end
                    if (v(k) > t),
                        f = v(k) - t ;
                        g = zeros(n,1);
                        g(obj.iD(k)) = 1;
                        obj.i_diag = k;
                        return;
                    end
                end

                if loop == 1, break; end
	
                % 4. G >= 0 constraint
                [R, p] = chol_ext(G);
                if (p ~= 0),
                    e = zeros(p,1);
                    e(p) = 1;
                    v = R\e;
                    f = -v'*G(1:p,1:p)*v;
                    g = zeros(n,1);
                    for i=1:n,
                        g(i) = -v'*obj.F{i}(1:p,1:p)*v;
                    end
                    return;
                end

                [v,d,q] = svd(G);
                h = v(:,1)* sqrt(d(1,1));
                %[v,d] = eigs(G,1);
                %h = v * sqrt(d);
                G = h*h';
                x = G(obj.iL);
                s = 1/(obj.c'*x);
                x = s*x; % rescale to the closest point
                G = toMat(obj, x0, obj.N);
            end

            % Begin objective function
            v = diag(G);
            [t, imax] = max(v); % update best so far t
            f = 0;
            %g = zeros(length(x), 1);
            g = zeros(n,1);
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
