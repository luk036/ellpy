function [a] = mle_corr_fn(Y, s)
% 1. Assume that the spatial correlation is isotopic: 
%   i.e. Sig_{ij} = rho(|| s_j - s_i ||)
% 2. Assume $Y$ is positive semidefinite
% Use Maximum likelihood estimation.

nn = 8; % 4 terms => cubic polynomial
x0 = ones(nn+1,1); % initial x0
%x0(1) = 3;
%Ae = diag(100*ones(nn,1)); % initial ellipsoid (sphere)
E = ell(100, x0);
P = mle_corr_oracle(Y, s, nn);
[x, bf, iter, flag] = ellipsoid_dc(@P.evaluate, E, Inf, 1000, 1e-4)
kappa = x(end)
a = x(1:end-1);
