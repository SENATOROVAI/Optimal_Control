function P = hinf_riccati(A, B, Bf, Q, gamma)
% Решение H-inf Риккати через гамильтониан (порт solve_P из work6.py).
% Возвращает [] если допустимого P нет.
R = B*B' - (Bf*Bf')/gamma^2;
H = [A, -R; -Q, -A'];
[V, D] = eig(H); d = diag(D);
[~, order] = sort(real(d));
stable = order(1:size(A,1));
if max(real(d(stable))) > -1e-6, P = []; return; end
U = V(:, stable); U1 = U(1:size(A,1), :); U2 = U(size(A,1)+1:end, :);
P = real(U2 / U1); P = (P + P')/2;
if min(eig(P)) < -1e-6, P = []; return; end
Acl = A - B*B'*P;                         % проверка устойчивости (как в work6.py)
if max(real(eig(Acl))) > -1e-9, P = []; return; end
end
