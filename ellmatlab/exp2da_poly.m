clear all
close all
exp2da_data;   % generate data

Y = cov(Ys',1);
[pp, degree] = lsq_corr_2d_fn(Y,s);
%a = mle_corr_fn(Y,s);


%pp = reshape(p,degree+1);

dmax = (s_end - s_beg);
hx = 0:0.2:dmax(1);
hy = -dmax(2):0.2:dmax(2);
nhx = size(hx,2);
nhy = size(hy,2);
pval = zeros(nhx,nhy);

for i=1:nhx,
  for j=1 :nhy,
    pval(i,j) = (hx(i).^(0:degree(1))) * pp * (hy(j).^(0:degree(2)))';
  end
end

%for i=1:(nhx+1)/2,
%  for j=1:nhy,
%    pval(i,j) =  pval(nhx+1-i,nhy+1-j);
%  end 
%end
figure

[HX HY] = ndgrid(hx,hy);
%%surf(HX,HY,pval);

figure
subplot(1,2,1);
%%fnplt(sp);
surf(HX,HY,pval);
subplot(1,2,2);
surf(HX,HY,pval);
view(2);
title('Anisotropic polynomial method estimation');
