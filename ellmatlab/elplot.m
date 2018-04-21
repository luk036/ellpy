function elplot(Ae,c)
A = inv(Ae);
elFunc = @(A11,A22,A12,A21,c1,c2,x,y) (c1-x).*(A11*(c1-x)+A21*(c2-y))+(c2-y).*(A12*(c1-x)+A22*(c2-y))-1;
ezplot(@(x,y) elFunc(A(1,1),A(2,2),A(1,2),A(2,1),c(1),c(2),x,y), [-12,12])
