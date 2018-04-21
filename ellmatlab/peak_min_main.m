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
clear all
clc, close all, fclose('all');
global Vol

%rand('twister',sum(100*clock));
%randn('state',sum(100*clock));
%*********************************************************************
% filter specs (for a low-pass filter)
%*********************************************************************
% number of FIR coefficients (including zeroth)
N = 14;
wpass = 0.12*pi;   % end of passband
wstop = 0.350*pi;   % start of stopband
delta0_wpass = 0.18; 
delta0_wstop = 0.05; 
delta  = 20*log10(1 + delta0_wpass);  % maximum passband ripple in dB (+/- around 0 dB)
delta2 = 20*log10(delta0_wstop);      % stopband attenuation desired in dB

%*********************************************************************
% optimization parameters
%*********************************************************************
% rule-of-thumb discretization (from Cheney's Approximation Theory)
m = 15*N;
w = linspace(0,pi,m)';  % omega

%*********************************************************************
% For vector to matrix and matrix to vector
%*********************************************************************
p = 1:(N*N);
P = reshape(p, [N N]);
L = tril(P);
U = triu(P)';
iL = L(find(L));
iU = U(find(U));
%iD = diag(P);

% make I matrices
B = zeros(N,N^2);
for i=0:N-1
   C = zeros(N,N);
   C = spdiags(ones(N,1),i,C);
   %B(i+1,:) = vect(C)';
   B(i+1,:) = C(:)';
end

% A is the matrix used to compute the power spectrum
% A(w,:) = [1 2*cos(w) 2*cos(2*w) ... 2*cos(N*w)]
A = [ones(m,1) 2*cos(kron(w,[1:N-1]))];

AB = A*B;
Ac = zeros(m, length(iL));
for i=1:m,
    v = AB(i,:);
    K = reshape(v, [N N]);
    Ac(i,:) = K(iU)';
end

% passband 0 <= w <= w_pass
ind_p = find((0 <= w) & (w <= wpass));    % passband
Apc  = Ac(ind_p,:);

% stopband (w_stop <= w)
ind_s = find((wstop <= w) & (w <= pi));   % stopband
Asc  = Ac(ind_s,:);

%remove redundant contraints
ind_nr = setdiff(1:m,ind_p);   % fullband less passband
ind_nr = setdiff(ind_nr, ind_s); % luk: for making parallel cut
Anrc = Ac(ind_nr,:);

%**********************************************************************************
Lp  = 10^(-delta/20);
Up  = 10^(+delta/20);
Sp  = 10^(delta2/20);

Lpsq = Lp^2;
Upsq = Up^2;
Spsq = Sp^2;


%********************************************************************
% optimization
%********************************************************************
tic
G0 = 1e-10*eye(N);
x0 = G0(iL);

Vol = [];
E = ell(1, x0);
P = peak_min_oracle(N, Apc, Asc, Anrc, Lpsq, Upsq, Spsq, iL);
[x, t_new, iter, flag, status] = ...
		ellipsoid_discrete(@P.assess, E, Inf, N*10000, 1e-6);
toc

figure(3);
plot(Vol);

%if flag == 0, return; end;

G = zeros(N,N);
G(iL) = x;
G = G';
G(iL) = x;

[v,d,q] = svd(G);
h = v(:,1)*sqrt(d(1,1));
if h(1) < 0, h = -h; end
r = B*G(:);




%***********************************************************************************

% compute the min attenuation in the stopband (convert to original vars)
Ustop = delta2;
fprintf(1,'Min attenuation in the stopband is %3.2f dB.\n',Ustop);

%*********************************************************************
% plotting routines
%*********************************************************************
% frequency response of the designed filter, where j = sqrt(-1)
j = sqrt(-1);
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