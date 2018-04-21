function [R, p] = chol_ext(A)
% If $A$ is positive definite, then $p$ is zero.
% If it is not, then $p$ is a positive integer,
% such that $v = R^{-1} e_p$ is a certificate vector 
% to make $v'*A(1:p,1:p)*v < 0$ 
[R, p] = chol(A);
if p == 0, return; end 
if p == 1, R(1,1) = sqrt(-A(1,1)); return; end

R(1,p) = A(1,p) / R(1,1);
if p > 2,
  for k=2:p-1,
    R(k,p) = (A(k,p) - R(1:k-1,p)'*R(1:k-1,k)) / R(k,k);
  end
end
R(p,p) = sqrt(-A(p,p) + R(1:p-1,p)'*R(1:p-1,p));
