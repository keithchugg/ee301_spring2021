N=1024; %%% Enter even values of N
n=0:1:N-1
m=n-N/2;

h=((-1).^m)./m;  %% inverse DTFT of jOmega (truncated)
h(N/2+1)=0;      %% delayed it so that it is a causal h[n]

figure
subplot(2,1,1)
stem(h)
title('h[n] -- Impulse Response')

subplot(2,1,2)
H=fft(h);
stem(abs(H))
title('H[k] DFT')

%%  See that the magnitude of H[k] is that of j(2pi)k/N
%% make N larger and the approximation becomes better
%% processing complexity and delay increases with N...