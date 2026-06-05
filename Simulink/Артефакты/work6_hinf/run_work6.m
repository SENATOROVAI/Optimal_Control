function run_work6()
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'lib'), here);
p = work6_params();
fig = fullfile(here,'figures'); res = fullfile(here,'results');
if ~exist(fig,'dir'), mkdir(fig); end
if ~exist(res,'dir'), mkdir(res); end

m = 'work6';
load_system(fullfile(here, 'work6.slx'));
[gamma_min, K] = hinf_design(p);
ecl = sort(real(eig(p.A + p.B*K)));     % Acl = A + B K (K=-B'P)

Acl = p.A + p.B*K;
sysC1 = ss(Acl, p.Bf, [1 0], 0);
sysC2 = ss(Acl, p.Bf, [0 1], 0);
sysFull = ss(Acl, p.Bf, eye(2), [0;0]);
hinf_C1 = norm(sysC1, inf); hinf_C2 = norm(sysC2, inf); hinf_full = norm(sysFull, inf);

so = sim(m);
t = so.x_log.Time; X = so.x_log.Data; U = so.u_log.Data;
f=figure('visible','off'); plot(t,X(:,1),t,X(:,2)); legend('x_1','x_2'); xlabel('t'); title('H_\infty: states under disturbance'); saveas(f,fullfile(fig,'states.png')); close(f);
f=figure('visible','off'); plot(t,U); xlabel('t'); ylabel('u'); title('H_\infty: control'); saveas(f,fullfile(fig,'control.png')); close(f);
% АЧХ всех трёх передаточных функций: C1, C2 и полная (наибольшее сингулярное число)
f=figure('visible','off'); w=logspace(-2,3,400);
m1=squeeze(abs(freqresp(sysC1,w))); m2=squeeze(abs(freqresp(sysC2,w)));
Hf=freqresp(sysFull,w); mf=zeros(size(w));
for i=1:numel(w), mf(i)=max(svd(Hf(:,:,i))); end
semilogx(w,20*log10(m1), w,20*log10(m2), w,20*log10(mf),'--'); grid on;
xlabel('\omega'); ylabel('dB'); legend('C_1','C_2','full (\sigma_{max})','Location','best');
title('H_\infty: magnitude response of channels C_1, C_2 and full transfer'); saveas(f,fullfile(fig,'freq.png')); close(f);

out = struct('gamma_min', gamma_min, 'gamma_design', gamma_min*1.01, 'K', K(:)', ...
    'closed_loop_eig_real', ecl(:)', 'hinf_C1', hinf_C1, 'hinf_C2', hinf_C2, 'hinf_full', hinf_full);
jsonwrite(fullfile(res, 'work6.json'), out);
close_system(m, 0);
fprintf('run_work6: gamma_min=%.4f hinf_full=%.4f K=[%.2f %.2f]\n', gamma_min, hinf_full, K(1), K(2));
end
