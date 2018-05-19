function [pp, degree] = lsq_corr_2d_fn(Y, s)
% 1. Assume that the spatial correlation is isotopic: 
%   i.e. Sig_{ij} = rho(|| s_j - s_i ||)
% 2. Assume $Y$ is positive semidefinite
% Use matrix norm estimation.

degree = [4 5];
nn = (degree(1)+1)*(degree(2)+1);

% x = [t, p(a), kappa]
x0 = ones(nn+2,1); % initial x0
x0(end) = 0.00001;
%x0(1) = 1;
%Ae = diag(100*ones(nn,1)); % initial ellipsoid (sphere)
E = ell(100, x0);
P = lsq_corr_2d_oracle(Y, s, nn, degree);
[x, bf, iter, feasible, status] = ellipsoid_dc(@P.evaluate, E, 100, 20000, 1e-4)
%kappa = x(end)
a = x(2:end-1);
pp = reshape(a, degree+1);
