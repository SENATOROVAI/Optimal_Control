function run_all()
% Runs all works (2-6): each script simulates its model.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'lib'));
works = {'work2_pontryagin','2'; 'work3_lqr','3'; 'work4_bellman','4'; ...
         'work5_lqg','5'; 'work6_hinf','6'};
for i = 1:size(works,1)
    d = fullfile(here, works{i,1});
    addpath(d);
    feval(['run_work' works{i,2}]);
end
disp('RUN_ALL DONE');
end
