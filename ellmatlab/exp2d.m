clear all
close all
exp2d_data;   % generate data

Y = cov(Ys',1);
a = lsq_corr_fn(Y,s);
%a = mle_corr_fn(Y,s);

figure
dist = 14;
h = 0:0.1:dist;
corr = var*var*exp(-0.5*(h.*h)/(sdkern*sdkern)/2);
plot(h,corr,'r');

hold on;
nn = length(a);
corr = polyval(a(nn:-1:1), h);
plot(h,corr,'b');

xlim([0,dist]);
xlabel('distance');
ylabel('variance');
legend('original', 'computed');
title('Isotropic spatial correlation estimation');

