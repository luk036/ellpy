# -*- coding: utf-8 -*-
from __future__ import print_function
 
from pprint import pprint
import numpy as np
from scipy.interpolate import UnivariateSpline


#function [sp, tau2, Sig]
def lsq_spline_fn(Y, s):
  '''
  # Assume that the spatial correlation is isotopic: 
  #   i.e. Sig_{ij} = rho(|| s_j - s_i ||)
  # Use Quadratic B-spline for representing the correlation function rho:
  #   i.e. rho(h) = sum_i p_i * B_i(h)
  # Optional Constraints:
  #   'm' : monotonic	i.e. p_i >= p_{i+1)
  #   'f' : fft		i.e. fft(rho(h)) elementwise >= 0
  #   'n' : non-negative	i.e. p_i >= 0 
  # Use least square estimation.
  '''
  n = len(s)
  d = 1  # dimension
  s_end = s[-1]
  s_beg = s(0)
  dmax = s_end - s_beg

  # Setup B-spline parameters
  kx = 3 # quadratic
  deltax = dmax/(n^(1/d))/sqrt(d)/4     # decide the resolution. better idea???
# If anthing goes wrong, try to reduce deltax and see

site = [deltax:deltax:dmax] # decide the site location
knotsx = augknt([-site,0,site], kx) # knot sequence
C0 = spcol(knotsx, kx, 0, 'sparse')   # collocation matrix for zero
xx = aveknt(knotsx,kx)'     # get the site locations
nn = size(xx,1)

disp('constructing F...')
# Convert to sparse matrices, make cvx solver more efficient???
Fs = cell(nn,1)
for k=1:nn
  Fs{k} = sparse([], [], [], n, n, 10*n)
end
for i=1:n,
  for j=1:n,
    dist = norm(s(j,:) - s(i,:))
    C = spcol(knotsx, kx, dist) # get the collocation matrix of b-spline
    for k=1:nn,
      if (C(k) ~= 0),
	Fs{k}(i,j) = C(k)
      end
    end
  end
  ##fprintf('#d,',i)
end


## F = sparse(n*n,nn)
## for i=1:n,
##   F(i+(i-1)*n,:) = C0
## end
## for i=1:n-1,
##   for j=i+1:n,
##     dist = norm(s(j,:) - s(i,:))
##     C = spcol(knotsx, kx, dist, 'sparse') # get the collocation matrix of b-spline
##     F(i+(j-1)*n,:) = C
##     C = spcol(knotsx, kx, -dist, 'sparse')
##     F(j+(i-1)*n,:) = C # symmetric
##   end
##   fprintf('#d,',i)
## end
## 
## 
## # Convert to sparse matrices, make cvx solver more efficient???
## Fs = cell(nn,1)
## for i=1:nn
##   Fs{i} = reshape(F(:,i),n,n)
## end
## clear F

disp('Setup for Fast Fourier Transform...')
d2 = 0:0.2:dmax  ## 0.1 ???
C2 = spcol(knotsx, kx, d2)
F2 = real(fftshift(C2))
nf = size(d2,2)


I = speye(n)
cvx_quiet(false)
cvx_begin
  variable p2(nn/2) # coefficents of B-spline
  variable tau2

  p = [p2(nn/2:-1:1) p2]
  Sig = 0
  for i=1:nn
    Sig = Sig + Fs{i}*p(i)
  end

  minimize( norm(Sig + tau2*I - Y,'fro') )

  subject to
    if (~isempty(strfind(opt, 'm'))), # monotonic decreasing
      for i=1:nn/2-1,
        p2(i) >= p2(i+1)
      end
    end

    if (~isempty(strfind(opt, 'n'))), # nonnegative
      p2 >= 0
    end

    if (~isempty(strfind(opt, 'f'))), # Fourier transform
      F2*p >= 0
    end

    Sig == semidefinite(n)
    tau2 >= 0
cvx_end

sp = spmak(knotsx, p, nn)

