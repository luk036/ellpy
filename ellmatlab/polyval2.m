function [ val ] = polyval2( pp, x, y )
%POLYVAL2 Summary of this function goes here
%   Detailed explanation goes here
    [nx, ny] = size(pp);
    ax = zeros(1,nx);
    if x < 0, x = -x; y = -y; end;
    for i=1:nx,
       ax(i) = polyval(pp(i,ny:-1:1),y);
    end
    val = polyval(ax(nx:-1:1),x);
end

