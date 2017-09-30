close all;
exp1d_data;   % generate data
%%load exp1d_data;
figure
plot(s,Ys(:,1:4),'-s');

figure
hold on
%% sp = log_mle_cc_corr_fn(Y,s,'mf');
%% fnplt(sp);
sp1 = lsq_corr_fn(Y,s,'mf');
fnplt(sp1,'g--');
%%sp2 = log_mle_cc_corr_fn(Y,s,[]);
%%fnplt(sp2,'r--');
[knotsx, kx] = fnbrk(sp1, 'knots', 'order');
xx = aveknt(knotsx, kx)';
%%corr = var*var*exp(-0.5*sqrt(xx.*xx)/(sdkern*sdkern)/2);
corr1 = var*var*exp(-0.5*(xx.*xx)/(sdkern*sdkern)/2);
corr2 = var*var*exp(-0.5*sqrt(xx.*xx)/sdkern);
corr = max([corr1'; corr2'])';
plot(xx,corr,'ro--');
xlim([0,xx(size(xx,1))]);
xlabel('distance');
ylabel('variance');
legend('Least squares','Max. Likelihood');
title('Isotropic spatial correlation estimation');

hold off
disp('Relative error:');
%%disp(norm(fnval(sp,xx) - corr)/norm(corr));
