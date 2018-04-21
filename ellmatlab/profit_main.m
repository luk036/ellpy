%global log_pA log_k v a alpha beta v1 v2
p = 20;
A = 40;
alpha = 0.1;
beta = 0.4;
v1 = 10;
v2 = 35;
k = 30.5;

y0 = [0, 0]'; % initial x0
Ae = diag([100, 100]); % initial ellipsoid (sphere)

%figure;
%axis([-12 12 -12 12]);
%axis square;
%elplot(Ae,y0);
%hold on

E = ell(Ae, y0);
P = profit_oracle(p, A, alpha, beta, v1, v2, k);
[yb1, fb, iter, flag, status] = ellipsoid_dc(@P.assess, E, 0, 200, 1e-4);
fb
iter

ui = 1;
e1 = 0.003;
e2 = 0.007;
e3 = 1;

% E = ell(Ae, y0);
% P = profit_rb_oracle(p, A, alpha, beta, v1, v2, k, ui, e1, e2, e3);
% [yb1, fb, iter, flag, status] = ellipsoid_dc(@P.assess, E, 0, 200, 1e-4);
% fb
% iter

E = ell(Ae, y0);
P = profit_q_oracle(p, A, alpha, beta, v1, v2, k);
[yb1, fb, iter, flag, status] = ellipsoid_discrete(@P.assess, E, 0, 200, 1e-4);
fb
x = exp(yb1)
iter

% y0 = [0, 0, 0]'; % initial x0
% Ae = diag([100, 100, 1000]); % initial ellipsoid
% E = ell(Ae, y0);
% P = profit_oracle2(p, A, alpha, beta, v1, v2, k);
% [yb, fb, iter, flag, status] = ellipsoid_dc(@P.assess, E, Inf, 200, 1e-4);
% fb2 = exp(-fb)
% iter