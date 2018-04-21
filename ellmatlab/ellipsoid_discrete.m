function [x_best, t1, iter, flag, status] = ellipsoid_discrete(assess, E, t, max_it, tol)
% -- Ellipsoid method for solving discrete convex problem
%
% input   
%         oracle        perform assessment on x0
%	      E(Ae,x0)      ellipsoid containing x*
%         t             best-so-far f
%         max_it        maximum number of iterations
%         tol           error tolerance                  
%
% output  
%         x             solution vector
%         iter          number of iterations performed
flag = 0; % no sol'n
x_last = E.xc;
x_best = NaN;
global Vol
%count = 100;
status = 1; % new

for iter = 1:max_it,
  if status == 3, % can't cut in the previous iteration
    [g, h, t1, x, loop] = assess(x, t, 1);  % new
	if loop == 0, % no more alternative cut
	  if flag == 0, x_best = x; end % output x anyway	  
	  return
    end
	h = h + g'*(x - E.xc);
  else
    [g, h, t1, x, loop] = assess(E.xc, t, 0);
    if loop == 1, % discrete sol'n
      h = h + g'*(x - E.xc);
	end
  end

  if (t ~= t1), % best t obtained
    flag = 1;
	t = t1;
    x_best = x;
  end  
  [status, tau] = update(E,g,h);
  R = chol(E.P);
  Vol = [Vol, sum(log(diag(R)))];
  %elplot(Ae,x0);
  
  if status == 1, % no sol'n
      return; 
  % elseif status == 3, % retry 20 times
      % count = count - 1;
	    % if count == 0,
          % if flag == 0, x_best = x; end;	  
	        % return; 
	    % end;
	    % continue;
  end
  
  if tau < tol, status = 2; return; end; % no more,
  if norm(x_last - E.xc) < 1e-8, 
      status = 4; 
      return; 
  end;
  x_last = E.xc;  
  count = 20; % restart the count
end
% END of ellipsoid_discrete
