close all
exp1d_data;  
Y = cov(Ys',1);
a = lsq_corr_fn(Y,s);
%a = mle_corr_fn(Y,s);

figure
h = 0:0.1:dist;
corr = var*var*exp(-0.5*(h.*h)/(sdkern*sdkern)/2);
plot(h,corr,'r');

hold on;
nn = length(a);
ezplot(@(x)polyval(a(nn:-1:1),x),[0 dist -1 5]);
%corr = zeros(1,length(h));
%for i=nn:-1:1,
%  corr = a(i) + corr .* h .* h;
%end
%plot(h,corr,'b');

xlim([0,dist]);
xlabel('distance');
ylabel('variance');
legend('original', 'computed');
title('Isotropic spatial correlation estimation');

