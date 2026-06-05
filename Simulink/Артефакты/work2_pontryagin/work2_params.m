function p = work2_params()
p.r  = 5;
p.T  = 4;
p.xa = [0; 0];   % x(0)
p.xb = [1; 0];   % x(T)
p.M  = [0 1 0 0;
       -6 -5 0 -1/(2*p.r);
       -2 0 0 6;
        0 -8 -1 5];
end
