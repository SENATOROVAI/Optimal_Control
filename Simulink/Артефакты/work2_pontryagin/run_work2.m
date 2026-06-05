function run_work2()
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'lib'), here);
p = work2_params();
fig = fullfile(here,'figures'); res = fullfile(here,'results');
if ~exist(fig,'dir'), mkdir(fig); end
if ~exist(res,'dir'), mkdir(res); end

[l0, sol] = solve_bvp_l0(p, p.r);
z0 = [p.xa; l0];
assignin('base', 'z0_var', z0);

m = 'work2';
load_system(fullfile(here, 'work2.slx'));
set_param([m '/Aug'], 'X0', 'z0_var');
so = sim(m);
t = so.z_log.Time; Z = so.z_log.Data;
x1s = Z(:,1); x2s = Z(:,2); l2s = Z(:,4);
us = -l2s / (2*p.r);
xT = [x1s(end), x2s(end)];   % из прогона Simulink-модели (краевое условие)

% Критерий считаем по плотному решению краевой задачи (как в Python-эталоне):
% прямое интегрирование сопряжённой системы неустойчиво и накапливает ошибку.
tg = linspace(0, p.T, 400);
Yg = deval(sol, tg); x1 = Yg(1,:)'; x2 = Yg(2,:)'; l2 = Yg(4,:)';
u = -l2 / (2*p.r);
Lc = x1.^2 + 4*x2.^2 + 5*u.^2;
Jt = cumtrapz(tg(:), Lc); J_opt = Jt(end);

f=figure('visible','off'); plot(t,x1s,t,x2s); legend('x_1','x_2'); xlabel('t'); title('Pontryagin: optimal trajectory'); saveas(f,fullfile(fig,'state.png')); close(f);
f=figure('visible','off'); plot(t,us); xlabel('t'); ylabel('u'); title('Pontryagin: optimal control'); saveas(f,fullfile(fig,'control.png')); close(f);
f=figure('visible','off'); plot(tg,Jt); xlabel('t'); ylabel('J(t)'); title('Pontryagin: accumulated cost'); saveas(f,fullfile(fig,'cost.png')); close(f);

% Чувствительность J(r): номинальный критерий (вес r=5) для регуляторов,
% синтезированных при разных r; минимум должен достигаться при r=5.
rs = linspace(3, 7, 9); Jr = zeros(size(rs));
for i = 1:numel(rs)
    rr = rs(i);
    [~, solr] = solve_bvp_l0(struct('xa',p.xa,'xb',p.xb,'T',p.T), rr);
    Yr = deval(solr, tg); x1r = Yr(1,:)'; x2r = Yr(2,:)'; l2r = Yr(4,:)';
    ur = -l2r / (2*rr);
    Jr(i) = trapz(tg(:), x1r.^2 + 4*x2r.^2 + 5*ur.^2);
end
f=figure('visible','off'); plot(rs,Jr,'o-'); xline(5,':'); xlabel('control weight r'); ylabel('J'); title('Pontryagin: cost vs r'); saveas(f,fullfile(fig,'cost_vs_r.png')); close(f);

out = struct('xT', xT, 'J_opt', J_opt, 'l0', l0(:)', 'r_grid', rs, 'J_grid', Jr);
jsonwrite(fullfile(res, 'work2.json'), out);
close_system(m, 0);
fprintf('run_work2: xT=[%.4f %.4f] J_opt=%.4f\n', xT(1), xT(2), J_opt);
end

function [l0, sol] = solve_bvp_l0(p, r)
if isfield(p,'M'), M = p.M; else, M = []; end
if isempty(M)
    M = [0 1 0 0; -6 -5 0 -1/(2*r); -2 0 0 6; 0 -8 -1 5];
end
odefun = @(t, z) M*z;
bcfun  = @(za, zb) [za(1) - p.xa(1); za(2) - p.xa(2); zb(1) - p.xb(1); zb(2) - p.xb(2)];
solinit = bvpinit(linspace(0, p.T, 50), [0;0;0;0]);
sol = bvp4c(odefun, bcfun, solinit);
l0 = sol.y(3:4, 1);
end
