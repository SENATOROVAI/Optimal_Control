function [gamma_min, K] = hinf_design(p)
% Бисекция по gamma (порт work6.py): gammas убывают от 20 к 0.5,
% первый gamma с пустым Риккати => gamma_lo (break), последний допустимый => gamma_hi.
gammas = linspace(20, 0.5, 400);
gamma_lo = []; gamma_hi = [];
for g = gammas
    if isempty(hinf_riccati(p.A,p.B,p.Bf,p.Q,g)), gamma_lo = g; break; end
    gamma_hi = g;
end
if isempty(gamma_lo)
    gamma_min = gamma_hi;
else
    while gamma_hi - gamma_lo > 1e-4
        mid = (gamma_lo + gamma_hi)/2;
        if isempty(hinf_riccati(p.A,p.B,p.Bf,p.Q,mid)), gamma_lo = mid; else, gamma_hi = mid; end
    end
    gamma_min = gamma_hi;
end
gamma_design = gamma_min * 1.01;
P = hinf_riccati(p.A, p.B, p.Bf, p.Q, gamma_design);
K = -(p.B' * P);   % знак как в эталоне: u = K x
end
