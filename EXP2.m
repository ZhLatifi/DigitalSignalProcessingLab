clc
close all
clear
%% 1-2-a)

% function conv = myconv(h, x)
% 
%     lenh = length(h);
%     lenx = length(x);
%     lenRes = lenh + lenx - 1;
%    
%     fliph = fliplr(h);
%     zeroPadx = [zeros(1, lenh - 1), x, zeros(1, lenh - 1)];
%     conv = zeros(1, lenRes);
%     
%     for i=1:1:lenRes
%         conv(i) = fliph*zeroPadx(i:i + lenh - 1)';
%     end
%     
% end

t = 0:0.05:2;
xa = ones(1, 21);
resa = myconv(xa, xa);

% Testing my function
figure(1);
plot(t, resa);
title("Convolution of rect signal with it self");
ylabel("Amplitude");
xlabel("Time");
%% 1-2-b)

t = 0:0.01:1.99;
xb = square(2*pi*t, 50);
hb = ones(1,10)/10;
resb = myconv(xb, hb);

figure(2);
subplot(3,1,1);
plot(xb)
title("Input Signal b");
ylabel("Amplitude");
xlabel("Time");

subplot(3,1,2);
plot(hb);
title("Filter b");
ylabel("Amplitude");
xlabel("Time");

subplot(3,1,3);
plot(resb);
title("Convolution b");
ylabel("Amplitude");
xlabel("Time");
%% 1-2-c)

t = 0:0.01:1.99;
xc = square(2*pi*t, 50);
hc =  zeros(1,15);

for i=1:1:15
    hc(i)=0.25*(0.75^(i-1));
end

resc = myconv(xc, hc);

figure(3);
subplot(3,1,1);
plot(xc)
title("Input Signal c");
ylabel("Amplitude");
xlabel("Time");

subplot(3,1,2);
plot(hc);
title("Filter c");
ylabel("Amplitude");
xlabel("Time");

subplot(3,1,3);
plot(resc);
title("Convolution c");
ylabel("Amplitude");
xlabel("Time");
%% 1-2-d)

t = 0:0.01:1.99;
xd = square(2*pi*t, 50);
resd = filter([1, -5, 10, -10, 5, -1], 5, xd);

figure(4);
subplot(2,1,1);
plot(xd)
title("Input Signal d");
ylabel("Amplitude");
xlabel("Time");

subplot(2,1,2);
plot(resd);
title("Convolution d");
ylabel("Amplitude");
xlabel("Time");
%% 2-2-a)

w1 = 0.05*pi;
w2 = 0.20*pi;
w3 = 0.35*pi;
wa = 0.15*pi;
wb = 0.25*pi;
t = 0:1:200;

s = sin(w2*t);
v = sin(w1*t) + sin(w3*t);
x = s + v;

figure(5)
plot(t, x);
hold on;
plot(t, s);
legend("x","s");
title("x & s");
%% 2-2-b)

M = 100;
w = zeros(1, M);
h = zeros(1, M);

for i=1:1:M
    w(i) = 0.54 - 0.46*sin(2*pi*i/M);
end

for i=1:1:M
    h(i) = w(i)*((wb/pi)*sinc((wb/pi)*(i-M/2)) - (wa/pi)*sinc((wa/pi)*(i-M/2)));
end

y = filter(h, 1, x);

figure(6);
plot(t, s);
hold on;
plot(t, y);
legend("s","filtered x");
title("Filtered");
%% 2-2-c)

coef = load("coef.mat").Num;
y2 = filter(coef , 1, x);

figure(7);
plot(t, s);
hold on;
plot(t, y);
legend("s","filtered x");
title("Filtered - FDAtool");
%% 2-3-a)

[audio, fs] = audioread("Audio01.wav");
audio = audio';
%% 2-3-b)

audio_filter = load("coef2.mat").filter;
%% 2-3-c)

carrier = zeros(1, size(audio, 2));

for i=1:1:size(audio, 2)
    carrier(i) = 2*cos(pi*i/2);
end

filtered1 = filter(audio_filter, 1, audio);   
carrMult1 = filtered1.*carrier;
filtered2 = filter(audio_filter, 1, carrMult1);
%% 2-3-d)

filtered3 = filter(audio_filter, 1, filtered2);
carrMult2 = filtered3.*carrier;
filtered4 = filter(audio_filter, 1, carrMult2);
% sound(audio', fs);
sound(filtered4, fs);
figure(8);
plot(audio),hold on, plot(filtered4);
%% 2-4-a)

t = 0:0.001:1.999;
f0 = 400;
f1 = 200;
t1 = 2;

x1 = chirp(t, f0 , t1, f1, 'linear');
x2 = sin(2*pi*100*t);
x3 = zeros(1, length(x2));
x3(250) = 50;

s = x1 + x2 + x3;

figure(9);
n = -1000:1:999;
plot(n, fftshift(abs(fft(s))));
title("FFT of Signal");
%% 2-4-b)

figure(10);
[spc1, w1, T1] = spectrogram(s, hamming(256));
spectrogram(s, hamming(256));
title("256 - Spectrogram");
figure(11);
[spc, w, T] = spectrogram(s, hamming(512));
spectrogram(s, hamming(512));
title("512 - Spectrogram");
figure(12);
mesh(w, T, abs(spc)');
view([0 90])
figure(13);
mesh(w1, T1, abs(spc1)');
view([0.0 90.0])
%% 2-5)

[clean, noisy] = wnoise('doppler', 10, 7);

figure(14);
subplot(2,1,1)
plot(noisy);
title("Noisy Doppler as a testFunction");
subplot(2,1,2)
plot(clean);
title("Clean Doppler as a testFunction");
[cA,cD] = dwt(noisy,'db1');

figure(15);
subplot(3,1,1);
plot(cA);
title("Approximation Coefficients");

subplot(3,1,2);
plot(cD);
title("Detail Coefficients");

xrec = idwt(cA, zeros(size(cA)),'sym4');
subplot(3,1,3);
plot(xrec);
hold on;
plot(noisy);
title("IDWT");
legend("IDWT", "Noisy");
%%
function conv = myconv(h, x)

    lenh = length(h);
    lenx = length(x);
    lenRes = lenh + lenx - 1;
   
    fliph = fliplr(h);
    zeroPadx = [zeros(1, lenh - 1), x, zeros(1, lenh - 1)];
    conv = zeros(1, lenRes);
    
    for i=1:1:lenRes
        conv(i) = fliph*zeroPadx(i:i + lenh - 1)';
    end
    
end
