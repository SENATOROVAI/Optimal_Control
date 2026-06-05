function run_work4()
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'lib'), here);
p = work4_params();
fig = fullfile(here,'figures'); res = fullfile(here,'results');
if ~exist(fig,'dir'), mkdir(fig); end
if ~exist(res,'dir'), mkdir(res); end

odeP = @(t, pv) reshape(-(p.A'*reshape(pv,2,2) + reshape(pv,2,2)*p.A ...
    - reshape(pv,2,2)*p.b*(1/p.r)*p.b'*reshape(pv,2,2) + p.Q), 4, 1);
tgrid = linspace(p.T, 0, 501);
[tb, Pb] = ode45(odeP, tgrid, zeros(4,1));
[ts, ix] = sort(tb); Pf = Pb(ix, :);
Kt = zeros(numel(ts), 2);
for i = 1:numel(ts)
    Pi = reshape(Pf(i,:), 2, 2);
    Kt(i, :) = (p.b' * Pi) / p.r;
end
P0 = reshape(Pf(1,:), 2, 2);
K0 = (p.b' * P0) / p.r;
J_opt = p.x0' * P0 * p.x0;

% From Workspace должен выдавать K(t) как строку 1x2 на каждом шаге:
% используем 3D-данные (1 x 2 x N), иначе сигнал трактуется как столбец 2x1
% и matrix-Product [1x2]*[2x1] не собирается.
Kts = timeseries(reshape(Kt', 1, 2, numel(ts)), ts);
assignin('base', 'Kts', Kts);

m = 'work4';
load_system(fullfile(here, 'work4.slx'));
so = sim(m);
t = so.x_log.Time; X = so.x_log.Data; U = so.u_log.Data;

f=figure('visible','off'); plot(t,X(:,1),t,X(:,2)); legend('x_1','x_2'); xlabel('t'); title('Bellman: states (finite horizon)'); saveas(f,fullfile(fig,'states.png')); close(f);
f=figure('visible','off'); plot(t,U); xlabel('t'); ylabel('u'); title('Bellman: control'); saveas(f,fullfile(fig,'control.png')); close(f);
f=figure('visible','off'); plot(ts,Kt(:,1),ts,Kt(:,2)); legend('K_1(t)','K_2(t)'); xlabel('t'); title('Bellman: gain K(t)'); saveas(f,fullfile(fig,'gain_Kt.png')); close(f);

% Накопленный критерий J(t) оптимального регулятора
Lc = sum((X*p.Q).*X,2) + p.r*U.^2; Jt = cumtrapz(t, Lc); J_sim = Jt(end);
f=figure('visible','off'); plot(t,Jt); xlabel('t'); ylabel('J(t)'); title('Bellman: accumulated cost J(t)'); saveas(f,fullfile(fig,'cost.png')); close(f);

% Стоимость при отклонении параметров регулятора.
% (а) J(t) для опт. и K(t)·0.8 / ·1.2; (б) J(T) как функция масштаба K(t):
% минимум достигается в оптимуме — отклонение увеличивает критерий.
f=figure('visible','off','Position',[100 100 1000 420]);
subplot(1,2,1); hold on; plot(t,Jt,'LineWidth',1.4); leg={'opt.'};
for sc = [0.8 1.2]
    assignin('base','Kts', timeseries(reshape((sc*Kt)',1,2,numel(ts)), ts));
    sp = sim(m); tp=sp.x_log.Time; Xp=sp.x_log.Data; Up=sp.u_log.Data;
    Jtp = cumtrapz(tp, sum((Xp*p.Q).*Xp,2) + p.r*Up.^2);
    plot(tp, Jtp, '--'); leg{end+1} = sprintf('K\\times%.1f', sc);
end
legend(leg,'Location','northwest'); xlabel('t'); ylabel('J(t)'); title('Accumulated cost J(t)');
scsweep = 0.5:0.05:1.5; Jend = zeros(size(scsweep));
for i = 1:numel(scsweep)
    assignin('base','Kts', timeseries(reshape((scsweep(i)*Kt)',1,2,numel(ts)), ts));
    sp = sim(m); tp=sp.x_log.Time; Xp=sp.x_log.Data; Up=sp.u_log.Data;
    Ji = cumtrapz(tp, sum((Xp*p.Q).*Xp,2) + p.r*Up.^2); Jend(i) = Ji(end);
end
[~, im] = min(Jend);
subplot(1,2,2); plot(scsweep,Jend,'-o','MarkerSize',3); hold on;
plot(scsweep(im),Jend(im),'rp','MarkerSize',12,'MarkerFaceColor','r');
xlabel('K(t) scale'); ylabel('J(T)'); grid on; title('Cost under controller deviation');
sgtitle('Bellman: cost under controller parameter deviation');
saveas(f,fullfile(fig,'perturbed.png')); close(f);
assignin('base','Kts', Kts); % восстановить оптимальный K(t)

out = struct('K0', K0(:)', 'J_opt', J_opt, 'J_sim', J_sim, 'P0', P0);
jsonwrite(fullfile(res, 'work4.json'), out);
close_system(m, 0);
fprintf('run_work4: K0=[%.4f %.4f] J_opt=%.4f\n', K0(1), K0(2), J_opt);
end
