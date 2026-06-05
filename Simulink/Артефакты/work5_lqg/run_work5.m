function run_work5()
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'lib'), here);
p = work5_params();
fig = fullfile(here, 'figures'); res = fullfile(here, 'results');
if ~exist(fig,'dir'), mkdir(fig); end
if ~exist(res,'dir'), mkdir(res); end

m = 'work5';

% Калмановский фильтр (наблюдатель) и LQR-регулятор
Pf = care(p.A', p.C', p.W, p.V);
L  = (Pf * p.C') / p.V;
K  = lqr(p.A, p.b, p.Qc, p.rc);

assignin('base', 'A_obs',  p.A - L*p.C);
assignin('base', 'B_obs',  [p.b, L]);
assignin('base', 'minusK', -K);
assignin('base', 'cholW',  chol(p.W, 'lower'));

ecl = sort(real(eig(p.A - p.b*K)));   % 2 значения, как в эталоне

load_system(fullfile(here, 'work5.slx'));

% Базовый прогон
so = sim(m);
t  = so.x_log.Time;  X = so.x_log.Data;  Xh = so.xhat_log.Data;

f=figure('visible','off'); plot(t,X(:,1),'b', t,Xh(:,1),'r--');
legend('x_1','xhat_1'); xlabel('t'); title('LQG: state and estimate');
saveas(f, fullfile(fig,'lqg.png')); close(f);

f=figure('visible','off'); plot(t,X(:,1)-Xh(:,1), t,X(:,2)-Xh(:,2));
legend('e_1','e_2'); xlabel('t'); title('LQG: estimation errors');
saveas(f, fullfile(fig,'errors.png')); close(f);

% Критерий оптимальности наблюдателя J = E[e_H' e_H]. Динамика ошибки
% e' = (A-LC)e + Gw - Lv не зависит от управления u, поэтому J одинаков
% при u=sin(t) и при LQG-обратной связи. В установившемся режиме E[e e'] = Pf,
% значит J_inf = trace(Pf). График J(t)=tr(Sigma(t)) — решение ковариационного
% уравнения Риккати, сходящееся к J_inf.
J_obs = trace(Pf);
odeS = @(tt, sv) reshape( p.A*reshape(sv,2,2) + reshape(sv,2,2)*p.A' + p.W ...
    - reshape(sv,2,2)*p.C'*(1/p.V)*p.C*reshape(sv,2,2), 4, 1);
[tS, Sv] = ode45(odeS, linspace(0, p.T, 200), reshape(eye(2),4,1));
Jt = zeros(size(tS));
for i = 1:numel(tS), Jt(i) = trace(reshape(Sv(i,:),2,2)); end
f=figure('visible','off'); plot(tS, Jt, 'b', [0 p.T], [J_obs J_obs], 'r--');
legend('J(t)=tr\Sigma(t)','J_\infty=tr P'); xlabel('t'); ylabel('J');
title('Kalman: estimation-error cost J'); saveas(f, fullfile(fig,'cost_J.png')); close(f);

% Чувствительность к усилению наблюдателя L (0.5L и 2L)
f=figure('visible','off'); hold on; leg={};
for sc = [0.5 1 2]
    Ls = sc * L;
    assignin('base', 'A_obs', p.A - Ls*p.C);
    assignin('base', 'B_obs', [p.b, Ls]);
    s = sim(m);
    plot(s.x_log.Time, s.x_log.Data(:,1) - s.xhat_log.Data(:,1));
    leg{end+1} = sprintf('L*%.1f', sc);
end
legend(leg); xlabel('t'); ylabel('e_1'); title('LQG: effect of gain L');
saveas(f, fullfile(fig,'vary_L.png')); close(f);
% восстановить базу
assignin('base', 'A_obs', p.A - L*p.C);
assignin('base', 'B_obs', [p.b, L]);

% Чувствительность к ковариации шума процесса W
W2 = [8 3.5; 3.5 10];
Pf2 = care(p.A', p.C', W2, p.V);
L2  = (Pf2 * p.C') / p.V;
assignin('base', 'A_obs', p.A - L2*p.C);
assignin('base', 'B_obs', [p.b, L2]);
assignin('base', 'cholW', chol(W2, 'lower'));
sW = sim(m);
f=figure('visible','off'); plot(sW.x_log.Time, sW.x_log.Data(:,1) - sW.xhat_log.Data(:,1));
xlabel('t'); ylabel('e_1'); title('LQG: larger process noise W');
saveas(f, fullfile(fig,'vary_W.png')); close(f);
% восстановить базу
assignin('base', 'A_obs', p.A - L*p.C);
assignin('base', 'B_obs', [p.b, L]);
assignin('base', 'cholW', chol(p.W, 'lower'));

% Чувствительность к ковариации шума измерения V
V2 = 20;
Pf3 = care(p.A', p.C', p.W, V2);
L3  = (Pf3 * p.C') / V2;
assignin('base', 'A_obs', p.A - L3*p.C);
assignin('base', 'B_obs', [p.b, L3]);
set_param([m '/vNoise'], 'Cov', mat2str(V2));
sV = sim(m);
f=figure('visible','off'); plot(sV.x_log.Time, sV.x_log.Data(:,1) - sV.xhat_log.Data(:,1));
xlabel('t'); ylabel('e_1'); title('LQG: larger measurement noise V');
saveas(f, fullfile(fig,'vary_V.png')); close(f);
% восстановить базу
assignin('base', 'A_obs', p.A - L*p.C);
assignin('base', 'B_obs', [p.b, L]);
set_param([m '/vNoise'], 'Cov', mat2str(p.V));


out = struct('L', L(:)', 'K', K(:)', 'Acl_lqg_eig_real', ecl(:)', 'J_obs', J_obs);
jsonwrite(fullfile(res, 'work5.json'), out);
close_system(m, 0);
fprintf('run_work5: L=[%.4f %.4f] K=[%.4f %.4f] J_obs=%.4f\n', L(1), L(2), K(1), K(2), J_obs);
end
