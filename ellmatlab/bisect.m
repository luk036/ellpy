function x = bisect(m, l, u, tol)
% line search
assert(m(u) ~= 0);
assert(m(l) == 0);
while (1),
    v = u - l;
    x = l + v/2;
    if m(x) ~= 0,
        u = x;
    else
        l = x;
    end
    if norm(v) < tol, break; end
end
% END of ellipsoid_dc
