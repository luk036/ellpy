clear all  
global Vol

N = 9;

%*********************************************************************
% For vector-to-matrix and matrix-to-vector
%*********************************************************************
p = 1:(N*N);
P = reshape(p, [N N]);
L = tril(P);
iL = L(find(L));

R = rand(N,N);
A = (1/N)*R'*R + eye(N);
%A = 100*eye(N);

%********************************************************************
% optimization
%********************************************************************
tic
h0 = zeros(N,1); % initial h0
G0 = h0*h0';
x0 = G0(iL);
M = length(x0);

Vol = [];

%Ae = diag(10*N*N*ones(M,1)); % initial ellipsoid (sphere)
E = ell(10*N*N, x0);
P = rank1_oracle(N, A, iL);
[x, t_new, iter, feasible, status] = ellipsoid_discrete(@P.evaluate, E, Inf, M*1000, 1e-8);
toc

plot(Vol);

G = zeros(N,N);
G(iL) = x;
G = G';
G(iL) = x;
[v,d,q] = svd(G);
h = v(:,1)*sqrt(d(1,1));
%[v,d] = eigs(G,1);
%h = v*sqrt(d);
h'*A*h,
