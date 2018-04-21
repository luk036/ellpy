% Synthesize 2D isotropic Gaussian random field data
nx = 10;	 % number of points in x and y direction
ny = 10;
s_beg = [1 1];
s_end = [11 11];

sdkern = 0.8;  % width of kernel (isotropic)
var = 2.0;     % standard derivation
N = 500;        % number of samples

% Create sites s in a uniform grid
sx = linspace(s_beg(1), s_end(1), nx)';
sy = linspace(s_beg(1), s_end(1), ny)';
%sx = s_beg(1):(s_end(1)-s_beg(1))/(nx-1):s_end(1);
%sy = s_beg(2):(s_end(2)-s_beg(2))/(ny-1):s_end(2);
[xx,yy] = ndgrid(sx, sy); 
s = [xx(:), yy(:)];
n = size(s,1);	      % equal to nx * ny

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

figure
subplot(2,2,1);
contourf(xx,yy,reshape(Ys(:,1),nx,ny));
subplot(2,2,2);
contourf(xx,yy,reshape(Ys(:,2),nx,ny));
subplot(2,2,3);
contourf(xx,yy,reshape(Ys(:,3),nx,ny));
subplot(2,2,4);
contourf(xx,yy,reshape(Ys(:,4),nx,ny));

