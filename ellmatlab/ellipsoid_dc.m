function [x_best, t1, iter, flag, status] ...
    = ellipsoid_dc(assess, E, t, max_it, tol)
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
flag = 0; % no sol'n
x_best = NaN;

for iter = 1:max_it,
    [cut, t1] = assess(E.xc, t);
    if (t ~= t1), % best t obtained
        flag = 1;
	    t = t1;
        x_best = E.xc;
    end  
    [status, tau] = update(E,g,h);
    if status == 1, return; end  
    if tau < tol, status = 2; return; end; % no more,
end
% END of ellipsoid_dc
