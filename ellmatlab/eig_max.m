function [ t, v ] = eig_max( A, t )
%EIG_MAX Maximum eigenvalue
%   Detailed explanation goes here
n = size(A,1);
v = zeros(n,1);
while (1)
    At = t*eye(n) - A;
    [R, p] = chol_ext(At);
    if (p == 0), return; end;
    e = zeros(p,1);
    e(p) = 1;
    v = R\e;
    d = v'*v;
    v = v/sqrt(d); 
    t = t + 1/d + 1e-6; % numerical problem???
end

