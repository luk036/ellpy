function [ t, v ] = eig_min( A, t )
%EIG_MIN Minimum eigenvalue
%   Detailed explanation goes here
n = size(A,1);
v = zeros(n,1);
while (1)
    At = A - t*eye(n);
    [R, p] = chol_ext(At);
    if (p == 0), break; end;
    e = zeros(p,1);
    e(p) = 1;
    v = R\e;
    d = v'*v;
    v = v/sqrt(d); 
    t = t - 1/d - 1e-10; % numerical problem???
end
assert(length(v) == n);
