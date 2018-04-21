% a fake dataset to make the bumps with
n = 20;   % number of points
s_begin = 1;
s_end = 10;
sdkern = 0.5;  % width of kernel
var = 2.0;     % standard derivation
N = 500;       % number of samples

dist = s_end - s_begin;
s = linspace(s_begin, s_end, n)';

Sig = ones(n,n);
for i=1:n-1,
  for j=i+1:n,
    d = s(j,:)' - s(i,:)';
    Sig(i,j) = exp(-0.5*(d'*d)/(sdkern*sdkern)/2);
    Sig(j,i) = Sig(i,j);
  end
end

A = sqrtm(Sig);
Ys = zeros(n,N);
ym = rand(n,1);
for k=1:N
  x = var * randn(n,1);
  y = A*x + ym + 0.5*randn(n,1);
  Ys(:,k) = y;
end

