function [a] = lsq_corr_fn(Y, s)
% 1. Assume that the spatial correlation is isotopic: 
%   i.e. Sig_{ij} = rho(|| s_j - s_i ||)
% 2. Assume $Y$ is positive semidefinite
% Use matrix norm estimation.

nn = 5; % 4 terms => cubic polynomial
% x = [t, kappa, p(a)]
x0 = ones(nn+1,1); % initial x0
%x0(1) = 1;
%Ae = diag(100*ones(nn,1)); % initial ellipsoid (sphere)
E = ell(100, x0);
P = lsq_corr_oracle(Y, s, nn);
u = 100;
l = 0;
bx = [];
tol = 1e-4;
while (u - l > tol),
    t = (u + l)/2;
    [x, bf, iter, feasible, status] = ...
	   ellipsoid_dc(@P.evaluate, E, t, 1000, tol);
    if feasible,
        u = t;
        bx = x;
    else
        l = t;
    end
end

kappa = bx(end)
a = bx(1:end-1);

