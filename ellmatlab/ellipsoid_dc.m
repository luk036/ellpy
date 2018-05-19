function [x_best, t1, iter, feasible, status] ...
    = ellipsoid_dc(evaluate, E, t, max_it, tol)
% -- Ellipsoid method for solving convex optimization problem
%
% input   
%         oracle        perform assessment on x0
%	      E(P,xc)       ellipsoid containing x*
%         t             best-so-far optimal sol'n
%         max_it        maximum number of iterations
%         tol           error tolerance                  
%
% output  
%         x             solution vector
%         iter          number of iterations performed
feasible = 0; % no sol'n
x_best = NaN;

for iter = 1:max_it,
    [cut, t1] = evaluate(E.xc, t);
    if (t ~= t1), % best t obtained
        feasible = 1;
	    t = t1;
        x_best = E.xc;
    end  
    [status, tau] = update(E,g,h);
    if status == 1, return; end  
    if tau < tol, status = 2; return; end; % no more,
end
% END of ellipsoid_dc
