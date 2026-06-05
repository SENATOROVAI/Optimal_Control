function run_work3()
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'lib'), here);
p = work3_params();
fig = fullfile(here, 'figures'); res = fullfile(here, 'results');
if ~exist(fig,'dir'), mkdir(fig); end
if ~exist(res,'dir'), mkdir(res); end

m = 'work3';
load_system(fullfile(here, 'work3.slx'));

K = lqr(p.A, p.b, p.Q, p.r);
P = care(p.A, p.b, p.Q, p.r);
ecl = sort(real(eig(p.A - p.b*K)));
Jinf = p.x0' * P * p.x0;

so = sim(m);
t = so.x_log.Time; X = so.x_log.Data; U = so.u_log.Data;
Lcost = sum((X*p.Q).*X, 2) + p.r * U.^2;
Jt = cumtrapz(t, Lcost); Jsim = Jt(end);

f=figure('visible','off'); plot(t,X(:,1),t,X(:,2)); legend('x_1','x_2'); xlabel('t'); title('LQR: optimal states'); saveas(f,fullfile(fig,'states.png')); close(f);
f=figure('visible','off'); plot(t,U); xlabel('t'); ylabel('u'); title('LQR: optimal control'); saveas(f,fullfile(fig,'control.png')); close(f);
f=figure('visible','off'); plot(t,Jt); xlabel('t'); ylabel('J(t)'); title('LQR: accumulated cost'); saveas(f,fullfile(fig,'cost.png')); close(f);

% Чувствительность: r in {1,5,20} — для каждого строим x1, x2, u, J
rs = [1 5 20];
f=figure('visible','off','Position',[100 100 900 650]);
ax = [subplot(2,2,1) subplot(2,2,2) subplot(2,2,3) subplot(2,2,4)];
arrayfun(@(a) hold(a,'on'), ax); leg={};
for rr = rs
    Kr = lqr(p.A, p.b, p.Q, rr);
    set_param([m '/Kgain'], 'Gain', mat2str(-Kr));
    s = sim(m); ti=s.x_log.Time; Xi=s.x_log.Data; Ui=s.u_log.Data;
    Ji = cumtrapz(ti, sum((Xi*p.Q).*Xi,2) + rr*Ui.^2);
    plot(ax(1),ti,Xi(:,1)); plot(ax(2),ti,Xi(:,2)); plot(ax(3),ti,Ui); plot(ax(4),ti,Ji);
    leg{end+1} = sprintf('r=%g', rr);
end
title(ax(1),'x_1'); title(ax(2),'x_2'); title(ax(3),'u'); title(ax(4),'J(t)');
arrayfun(@(a) xlabel(a,'t'), ax); legend(ax(1), leg);
sgtitle('LQR: effect of weight r (x_1, x_2, u, J)');
saveas(f,fullfile(fig,'vary_r.png')); close(f);

% Чувствительность: Q = k*Q, k in {0.5,1,2} — для каждого строим x1, x2, u, J
ks = [0.5 1 2];
f=figure('visible','off','Position',[100 100 900 650]);
ax = [subplot(2,2,1) subplot(2,2,2) subplot(2,2,3) subplot(2,2,4)];
arrayfun(@(a) hold(a,'on'), ax); leg={};
for kk = ks
    Qk = kk*p.Q; Kk = lqr(p.A, p.b, Qk, p.r);
    set_param([m '/Kgain'], 'Gain', mat2str(-Kk));
    s = sim(m); ti=s.x_log.Time; Xi=s.x_log.Data; Ui=s.u_log.Data;
    Ji = cumtrapz(ti, sum((Xi*Qk).*Xi,2) + p.r*Ui.^2);
    plot(ax(1),ti,Xi(:,1)); plot(ax(2),ti,Xi(:,2)); plot(ax(3),ti,Ui); plot(ax(4),ti,Ji);
    leg{end+1} = sprintf('k=%g', kk);
end
title(ax(1),'x_1'); title(ax(2),'x_2'); title(ax(3),'u'); title(ax(4),'J(t)');
arrayfun(@(a) xlabel(a,'t'), ax); legend(ax(1), leg);
sgtitle('LQR: effect of Q=kQ (x_1, x_2, u, J)');
saveas(f,fullfile(fig,'vary_Q.png')); close(f);

% Возмущённый K (+5%): x1, x2, u, J в сравнении с оптимальным
Kp = 1.05 * K;
set_param([m '/Kgain'], 'Gain', mat2str(-Kp)); sp = sim(m);
tp=sp.x_log.Time; Xp=sp.x_log.Data; Up=sp.u_log.Data;
Jtp = cumtrapz(tp, sum((Xp*p.Q).*Xp,2) + p.r*Up.^2);
Acl_p = p.A - p.b*Kp; ecl_p = sort(real(eig(Acl_p)));
Pp = lyap(Acl_p', p.Q + Kp'*p.r*Kp); Jp = p.x0'*Pp*p.x0;
f=figure('visible','off','Position',[100 100 900 650]);
subplot(2,2,1); plot(t,X(:,1),tp,Xp(:,1),'--'); title('x_1'); xlabel('t'); legend('opt.','K+5%');
subplot(2,2,2); plot(t,X(:,2),tp,Xp(:,2),'--'); title('x_2'); xlabel('t');
subplot(2,2,3); plot(t,U,tp,Up,'--'); title('u'); xlabel('t');
subplot(2,2,4); plot(t,Jt,tp,Jtp,'--'); title('J(t)'); xlabel('t');
sgtitle('LQR: perturbed K (+5%) vs optimal');
saveas(f,fullfile(fig,'perturbed_K.png')); close(f);
set_param([m '/Kgain'], 'Gain', mat2str(-K)); % восстановить


out = struct('K', K(:)', 'closed_loop_eig', ecl(:)', 'J_inf', Jinf, 'J_sim', Jsim, ...
    'K_pert', Kp(:)', 'closed_loop_eig_pert', ecl_p(:)', 'J_ss_pert', Jp);
jsonwrite(fullfile(res, 'work3.json'), out);
close_system(m, 0);
fprintf('run_work3: K=[%.4f %.4f] Jinf=%.4f Jsim=%.4f\n', K(1), K(2), Jinf, Jsim);
end
