% CVX code by Almir Mutapcic in 2006.  
% Adapted in 2010 for impulse response peak-minimization by convex iteration by Christine Law.
%
% "FIR Filter Design via Spectral Factorization and Convex Optimization"
% by S.-P. Wu, S. Boyd, and L. Vandenberghe
%
% Designs an FIR lowpass filter using spectral factorization method with
% constraint on maximum passband ripple and stopband attenuation:
%
%   minimize   max |H(w)|                      for w in stopband
%       s.t.   1/delta <= |H(w)| <= delta      for w in passband
%
% We change variables via spectral factorization method and get:
%
%   minimize   max R(w)                        for w in stopband
%       s.t.   (1/delta)^2 <= R(w) <= delta^2  for w in passband
%              R(w) >= 0                       for all w
%
% where R(w) is squared magnitude frequency response
% (and Fourier transform of autocorrelation coefficients r).
% Variables are coeffients r and G = hh' where h is impulse response.  
% delta is allowed passband ripple.
% This is a convex problem (can be formulated as an SDP after sampling).
clear all, clc, close all, fclose('all');
rand('twister',sum(100*clock));
randn('state',sum(100*clock));
%*********************************************************************
% filter specs (for a low-pass filter)
%*********************************************************************
% number of FIR coefficients (including zeroth)
N = 32;
wpass = 0.12*pi;   % end of passband
wstop = 0.20*pi;   % start of stopband
delta0_wpass = 0.125; 
delta0_wstop = 0.125; 
delta  = 20*log10(1 + delta0_wpass);  % maximum passband ripple in dB (+/- around 0 dB)
delta2 = 20*log10(delta0_wstop);      % stopband attenuation desired in dB

%*********************************************************************
% optimization parameters
%*********************************************************************
% rule-of-thumb discretization (from Cheney's Approximation Theory)
m = 15*N;
w = linspace(0,pi,m)';  % omega

% A is the matrix used to compute the power spectrum
% A(w,:) = [1 2*cos(w) 2*cos(2*w) ... 2*cos(N*w)]
A = [ones(m,1) 2*cos(kron(w,[1:N-1]))];

% passband 0 <= w <= w_pass
ind_p = find((0 <= w) & (w <= wpass));    % passband
Lp  = 10^(-delta/20);
Up  = 10^(+delta/20);
Ap  = A(ind_p,:);

% stopband (w_stop <= w)
ind_s = find((wstop <= w) & (w <= pi));   % stopband
Sp  = 10^(delta2/20);
As  = A(ind_s,:);

%remove redundant contraints
ind_nr = setdiff(1:m,ind_p);   % fullband less passband
Anr = A(ind_nr,:);

% make I matrices
B = zeros(N,N^2);
for i=0:N-1
   C = zeros(N,N);
   C = spdiags(ones(N,1),i,C);
   %B(i+1,:) = vect(C)';
   B(i+1,:) = C(:)';
end

%initial direction vector
W = eye(N);

%********************************************************************
% optimization
%********************************************************************
convergence = [];
iteration = 1;
while 1
   tic
   fprintf('\niteration %d\n', iteration);
   cvx_quiet(true);
   cvx_solver('sdpt3');
   cvx_precision('best');
   cvx_begin
      variable r(N,1);
      variable G(N,N) symmetric;
      minimize(trace(W'*G));
      %peak impulse response by cut and try
      max(diag(G)) <= 0.1053^2;
      % passband constraints
      Ap*r >= Lp.^2;
      Ap*r <= Up.^2;
      %stopband constraint
      As*r <= Sp.^2;
      % nonnegative-real constraint
      Anr*r >= 0;
      % relate r to h
      %r == B*vect(G);
      r == B*G(:);
      G == semidefinite(N);
   cvx_end
   toc
   
   %compute new direction vector
   [v,d,q] = svd(G);
   f = diag(d); fprintf('first few eigenvalues of G:\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n', f(1:7));
   W = v(:,2:N)*v(:,2:N)';
   rankG = sum(diag(d) > max(diag(d))*1e-5);
   fprintf('rank(G)=%d,  trace(W*G)=%f\n', rankG, trace(G*W)); 
   
   figure(1)
   % FIR impulse response
   h = v(:,1)*sqrt(d(1,1));
   plot([0:N-1],h','ob:')
   xlabel('t'), ylabel('h(t)')

   % monitor convergence to 0
   convergence = [convergence trace(G*W)];
   if iteration > 1
      figure(4)
      plot(convergence)
      set(gcf,'position',[70 200 256 256])
   end
   pause(1)
   
   % check if problem was successfully solved
   disp(['Problem is ' cvx_status])
   if (rankG == 1)
      break
   end
   if ~strfind(cvx_status,'Solved')
       fprintf(2,'Excuse me.\n')
   end

   iteration = iteration + 1;
end

% compute the min attenuation in the stopband (convert to original vars)
Ustop = delta2;
fprintf(1,'Min attenuation in the stopband is %3.2f dB.\n',Ustop);

%*********************************************************************
% plotting routines
%*********************************************************************
% frequency response of the designed filter, where j = sqrt(-1)
H = [exp(-j*kron(w,[0:N-1]))]*h;

figure(2);
subplot(121)
% magnitude
plot(w,20*log10(abs(H)), ...
   [0 wpass],[delta delta],'r--', ...
   [0 wpass],[-delta -delta],'r--', ...
   [wstop pi],[Ustop Ustop],'r--')
xlabel('w')
ylabel('mag H(w) in dB')
axis([0 pi -50 5])
title(sprintf('N=%d, w_p(pi)=%3.2f, w_s(pi)=%3.2f, delta=%3.2f', N, wpass/pi, wstop/pi, delta));

%compare impulse response designed by conventional method
subplot(122)
h_sp = spectral_fact(r);  %from CVX distribution, Examples subdirectory
plot([0:N-1],h_sp','+r--');
hold on;
plot([0:N-1],h(end:-1:1)','ob:');
legend('conventional','optimal');
xlabel('t'), ylabel('h(t)'); grid
title(sprintf('h_{max} conventional=%3.4f, h_{max} optimal=%3.4f',max(abs(h_sp)),max(abs(h))));
set(gcf,'Outerposition',[300 300 256*4 256*2])
 
figure(1)
% FIR impulse response
plot([0:N-1],h','ob:'); 
xlabel('t'), ylabel('h(t)')