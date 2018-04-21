clear all
exp1d_data;
Y = (Ys*Ys')./N;

% Assume that y (vector) is Gaussian with zero mean, i.e., y ~ N(0,Sig)
% Assume that the spatial correlation is isotopic: 
%   i.e. Sig_{ij} = rho(|| s_j - s_i ||)
% Use B-spline for representing the correlation function rho:
%   i.e. rho(x) = sum_i p_i * B_i(x)
% Request that rho has to be nonincreasing: i.e. p_i >= p_{i+1)
% Request that Sig has to be symmetric positive definite
% Use maximum likelihood estimation.
% log-likelihood function is given by:
%   log det Sig + Trace(Sig^{-1} * Y)
% The first term is concave whereas the second term is convex
% Use convex-concave procedure: successively solve a sequence of convexified 
% problems

% Setup B-spline parameters
kx = 3; % quadratic
knotsx = augknt([0:0.2:0.8, 1:10],kx); % knot sequence
C0 = spcol(knotsx, kx, 0); % collocation matrix for zero
nx = size(C0,2);

F = zeros(n,n,nx);
for i=1:n,
  F(i,i,:) = C0;
end
for i=1:n-1,
  for j=i+1:n,
    d = norm(s(j) - s(i));
    C = spcol(knotsx, kx, d);
    F(i,j,:) = C;
    F(j,i,:) = F(i,j,:); % symmetric
  end
end
% Note: in fact all F's should be sparse



%% % Successively solve the convex-concave problem
%%log_mle_solver; % take the non-parametric sol'n as an initial value
%%Sig = inv(S);  % initial correlation matrix
Sig = eye(n);

maxiter = 100;
cvx_quiet(true);
for k=1:maxiter,
  Sig_k = Sig; % covariance matrix at the kth iteration
  %% t1 = log(det(Sig_k)); 
  T2 = inv(Sig_k);
  clear q;
  clear Sig;
  k,
  cvx_begin sdp
    variable p(nx)  % coefficents of B-spline
    variable S(n,n) symmetric
    variable Sig(n,n) symmetric
    % minimize the convexified negative log-likehood function
    % Note: CVX does not accept sum(matrix_frac(Ys(:,i),Sig)) or
    %   trace(Sig^{-1}*Y)
    minimize(trace(T2*(Sig - Sig_k)) + trace(S*Y))

    subject to
      for i=1:nx-1,
        p(i) >= p(i+1); % nonincreasing
      end
      p(nx) >= 0;    % nonnegative

      % The following statement is equivalent to Sig = sum( F(:,:,i)*p(i) )
      Sig == reshape(reshape(F,n*n,nx)*p, n, n);

      [Sig     eye(n);
       eye(n)  S       ] >= 0;

  cvx_end

  err = norm(Sig - Sig_k),
  if (err < 1e-4),
    disp('converge!');
    break;
  end 
end

sp = spmak(knotsx, p, nx);
fnplt(sp,'r');
