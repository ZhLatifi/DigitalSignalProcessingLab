%% 3-1-a)

w1 = 0:0.01:pi;
R = [0.8, 0.9, 0.99];
f0 = 500;
fs = 10000;
w0 = (2*pi*f0)/fs;
for i = 1:1:3
    G = (1-R(i)).*(1-2*R(i)*cos(2*w0) + R(i)^2)^0.5;
    H = (G^2) ./ ((1-2*R(i)*cos(w1-w0) + R(i)^2).*(1-2*R(i)*cos(w1+w0) + R(i)^2));
    subplot(3, 1, i);
    plot(w1, H);
    title(sprintf("|H(jw)|^2, R = %.2f", R(i)));
end
%% 3-1-b)

n = 0:1:300;
for i = 1:1:3
    hn = (G/sin(w0)).*(R(i).^n).*sin(w0*n+w0);
    subplot(3, 1, i);
    plot(n, hn/G);
    title(sprintf("h[n]/G, R = %.2f", R(i)));
end
%% 3-1-c)

n = 0:1:300;
v = randn(1, 301);
s = cos(w0*n);
x = s + v;

for i = 1:1:3
    a1 = -2*R(i)*cos(w0);
    a2 = R(i).^2;
    w1 = 0;
    w2 = 0;
    y = zeros(1, 301);
    for j = 1:1:301
        y(j) = -a1*w1 - a2*w2 +G*x(j);
        w2 = w1;
        w1 = y(j);
    end
    subplot(3, 1, i);
    plot(n, y);
    hold on;
    plot(n, s);
    title(sprintf("y[n] for x[n] as input, R = %.2f", R(i)));
    legend("recovered", "desired");
end
%% 3-1-d)

figure();
NRR = zeros(1, 3);
for i = 1:1:3
    a1 = -2*R(i)*cos(w0);
    a2 = R(i).^2;
    w1 = 0;
    w2 = 0;
    yv = zeros(1, 301);
    for j = 1:1:301
        yv(j) = -a1*w1 - a2*w2 +G*v(j);
        w2 = w1;
        w1 = yv(j);
    end
    subplot(3, 1, i);
    plot(n, yv);
    hold on;
    plot(n, v);
    title(sprintf("y_v[n] for v[n] as input, R = %.2f", R(i)));
    NRR(i) = (std(yv)/std(v))^2;
end
%% 3-1-e)

for i = 1:1:3
    fprintf('NRR(%.2f) = %.5f\n', R(i), NRR(i));
end
%% 3-2-a)

b1 = [0.969531, -1.923772, 0.969531];
a1 = [1, -1.923772, 0.939063];
freqz(b1, a1);
title("H_1(z)");
b2 = [0.996088, -1.976468, 0.996088];
a2 = [1, -1.976468, 0.992177];
freqz(b2, a2);
title("H_2(z)");
%% 3-2-b)

fs = 400;
Ts = 1/fs;
H1 = tf(b1, a1, Ts);
stepinfo(H1, 'SettlingTimeThreshold', 0.01)
H2 = tf(b2, a2, Ts);
stepinfo(H2, 'SettlingTimeThreshold', 0.01)
%% 3-2-c&d)

f1 = 4;
f2 = 8;
f3 = 12;
t1 = 0:Ts:2-Ts;
t2 = 2:Ts:4-Ts;
t3 = 4:Ts:6-Ts;
t = [t1, t2, t3];
x1 = cos(2*pi*f1*t1);
x2 = cos(2*pi*f2*t2);
x3 = cos(2*pi*f3*t3);
x = [x1, x2, x3];
figure();
plot(t, x);
title("Input Signal");
xlabel('t (sec)');
ylabel('x(t)');
figure();
plot(t, filter(b1, a1, x));
title("Notch Filtered Output - H1");
xlabel('t (sec)');
ylabel('x(t)');
%% 3-2-e)

figure();
plot(t, filter(b2, a2, x));
title("Notch Filtered Output - H2");
xlabel('t (sec)');
ylabel('x(t)');
%% 3-2-f)

[mag1, phase1, w1] = bode(H1, {0, 20*2*pi});
mag1 = squeeze(mag1);
figure();
plot(w1/(2*pi), mag1);
title('Notch Filter Response');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
[mag2, phase2, w2] = bode(H2, {0, 20*2*pi});
mag2 = squeeze(mag2);
hold on;

plot(w2/(2*pi), mag2);
grid on;
grid minor;
hold on;

ind = find(abs(mag1) < sqrt(1/2), 1, 'first');
fL1 = w1(ind)/(2*pi);
ind = find(abs(mag1) < sqrt(1/2), 1, 'last');
fH1 = w1(ind)/(2*pi);
plot([fL1 fH1], sqrt(1/2).*ones(1,2), 'ro');
hold on;

ind = find(abs(mag2) < sqrt(1/2), 1, 'first');
fL2 = w2(ind)/(2*pi);
ind = find(abs(mag2) < sqrt(1/2), 1, 'last');
fH2 = w2(ind)/(2*pi);
plot([fL2 fH2], sqrt(1/2).*ones(1,2), 'rx');
legend("H_1", "H_2", "fc_1", "fc_2");
%% 3-2-g)

bw1 = fH1 - fL1
bw2 = fH2 - fL2
%% 3-2-h)

b1 = [0.030469, 0, -0.030469];
a1 = [1, -1.923772, 0.939063];
freqz(b1, a1);
title("H_1(z)");
b2 = [0.003912, 0, -0.003912];
a2 = [1, -1.976468, 0.992177];
freqz(b2, a2);
title("H_2(z)");
fs = 400;
Ts = 1/fs;
H1 = tf(b1, a1, Ts);
stepinfo(H1, 'SettlingTimeThreshold', 0.01)
H2 = tf(b2, a2, Ts);
stepinfo(H2, 'SettlingTimeThreshold', 0.01)
figure();
plot(t, filter(b1, a1, x));
title("Peaking Filtered Output - H1");
xlabel('t (sec)');
ylabel('x(t)');
figure();
plot(t, filter(b2, a2, x));
title("Peaking Filtered Output - H2");
xlabel('t (sec)');
ylabel('x(t)');
%%
[mag1, phase1, w1] = bode(H1, {0, 2*pi*20});
mag1 = squeeze(mag1);
figure();
plot(w1/(2*pi), mag1);
title('Peaking Filter Response');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
[mag2, phase2, w2] = bode(H2, {0, 2*pi*20});
mag2 = squeeze(mag2);
hold on;

plot(w2/(2*pi), mag2);
grid on;
grid minor;
hold on;

ind = find(abs(mag1) >= sqrt(1/2), 1, 'first');
fL1 = w1(ind-1)/(2*pi);
ind = find(abs(mag1) >= sqrt(1/2), 1, 'last');
fH1 = w1(ind+1)/(2*pi);
plot([fL1 fH1], sqrt(1/2).*ones(1,2), 'ro');
hold on;

ind = find(abs(mag2) >= sqrt(1/2), 1, 'first');
fL2 = w2(ind-1)/(2*pi);
ind = find(abs(mag2) >= sqrt(1/2), 1, 'last');
fH2 = w2(ind)/(2*pi);
plot([fL2 fH2], sqrt(1/2).*ones(1,2), 'rx');
legend("H_1", "H_2", "fc_1", "fc_2");
%%
bw1 = fH1 - fL1
bw2 = fH2 - fL2
%% 3-3-b)

f = 0.15/2;
A1 = 2;
A2 = 4;
A3 = 0.5;
t1 = 0:1:200;
t2 = 201:1:400;
t3 = 401:1:600;
t = [t1, t2, t3];
x1 = A1*cos(2*pi*f*t1);
x2 = A2*cos(2*pi*f*t2);
x3 = A3*cos(2*pi*f*t3);
x = [x1, x2, x3];

lambda = 0.9;
rho = 0.2;
c = zeros(1, 601);
c(1) = 0.5;
for n = 2:600
    c(n) = lambda*c(n-1) + (1-lambda)*abs(x(n));
end

G = zeros(1, 601);
for n = 1:601
    if c(n) >= c(1)
       G(n) = (c(n)/c(1)).^(rho-1);
    else
       G(n) = 1;
    end
    y(n) = G(n)*x(n);
end

figure();
subplot(1, 2, 1);
plot(t, x);
ylim([-5 5])
title("Input Signal");
xlabel('time samples n');
ylabel('x(n)');
grid on;

subplot(1,2,2);
plot(t, y);
ylim([-5 5])
title("Comprosser Output");
xlabel('time samples n');
ylabel('y(n)');
grid on;
figure();
subplot(1,2,1);
plot(t, c);
ylim([0 3])
title("Control Signal");
xlabel('time samples n');
ylabel('c(n)');
grid on;

subplot(1,2,2);
plot(t, G);
ylim([0 1.25])
title("Compressor Gain");
xlabel('time samples n');
ylabel('G(n)');
grid on;
g = movmean(G, 7);
y = g.*x;

figure();
subplot(1,2,1);
plot(t, y);
ylim([-5 5])
title("Compressor Output");
xlabel('time samples n');
ylabel('y(n)');
grid on;

subplot(1,2,2);
plot(t, g);
ylim([0 1.25])
title("Compressor Gain");
xlabel('time samples n');
ylabel('g(n)');
grid on;
L = 7;
g = zeros(1, 601);
lambda = 0.9;
rho = 0.1;
c = zeros(1, 601);
c(1) = 1.5;
for n = 2:600
    c(n) = lambda*c(n-1) + (1-lambda)*abs(x(n));
end

G = zeros(1, 601);
for n = 1:601
    if c(n) < c(1)
       G(n) = 1;
    else
       G(n) = (c(n)/c(1)).^(rho-1);
    end    
end
g = movmean(G, L);
y = g.*x;


figure();
subplot(1,2,1);
plot(t, y);
ylim([-5 5])
title("Compressor Output");
xlabel('time samples n');
ylabel('y(n)');
grid on;

subplot(1,2,2);
plot(t, g);
ylim([0 1.25])
title("Compressor Gain");
xlabel('time samples n');
ylabel('g(n)');
grid on;
lambda = 0.9;
rho = 2;
c = zeros(1, 601);
c(1) = 0.5;
for n = 2:600
    c(n) = lambda*c(n-1) + (1-lambda)*abs(x(n));
end

G = zeros(1, 601);
for n = 1:601
    if c(n) >= c(1)
       G(n) = 1;
    else
       G(n) = (c(n)/c(1)).^(rho-1);
    end
end

g = movmean(G, L);
y = g.*x;

figure();
subplot(1,2,1);
plot(t, y);
ylim([-5 5])
title("Expander Output");
xlabel('time samples n');
ylabel('y(n)');
grid on;

subplot(1,2,2);
plot(t, g);
ylim([0 1.25])
title("Expander Gain");
xlabel('time samples n');
ylabel('G(n)');
grid on;
lambda = 0.9;
rho = 10;
c = zeros(1, 601);
c(1) = 0.5;
for n = 2:600
    c(n) = lambda*c(n-1) + (1-lambda)*abs(x(n));
end

G = zeros(1, 601);
for n = 1:601
    if c(n) >= c(1)
       G(n) = 1;
    else
       G(n) = (c(n)/c(1)).^(rho-1);
    end
end

g = movmean(G, L);
y = g.*x;

figure();
subplot(1,2,1);
plot(t, y);
ylim([-5 5])
title("Expander Output");
xlabel('time samples n');
ylabel('y(n)');
grid on;

subplot(1,2,2);
plot(t, g);
ylim([0 1.25])
title("Expander Gain");
xlabel('time samples n');
ylabel('G(n)');
grid on;
%% 3-3-c)

lambda = 0.9;
rho = 0.25;
c = zeros(1, 601);
c(1) = 0.5;
for n = 2:600
    c(n) = lambda*c(n-1) + (1-lambda)*abs(x(n));
end

G = zeros(1, 601);
for n = 1:601
    if c(n) >= c(1)
       G(n) = (c(n)/c(1)).^(rho-1);
    else
       G(n) = 1;
    end
    y(n) = G(n)*x(n);
end

figure();
subplot(1, 2, 1);
plot(t, x);
ylim([-5 5])
title("Input Signal");
xlabel('time samples n');
ylabel('x(n)');
grid on;

subplot(1,2,2);
plot(t, y);
ylim([-5 5])
title("Comprosser Output");
xlabel('time samples n');
ylabel('y(n)');
grid on;
figure();
subplot(1,2,1);
plot(t, c);
ylim([0 3])
title("Control Signal");
xlabel('time samples n');
ylabel('c(n)');
grid on;

subplot(1,2,2);
plot(t, G);
ylim([0 1.25])
title("Compressor Gain");
xlabel('time samples n');
ylabel('G(n)');
grid on;
%% 3-3-d)

lambda = 0.9;
rho = 0.5;
c = zeros(1, 601);
c(1) = 1.3;
for n = 2:600
    c(n) = lambda*c(n-1) + (1-lambda)*abs(x(n));
end

G = zeros(1, 601);
for n = 1:601
    if c(n) >= c(1)
       G(n) = (c(n)/c(1)).^(rho-1);
    else
       G(n) = 1;
    end
    y(n) = G(n)*x(n);
end

figure();
subplot(1, 2, 1);
plot(t, x);
ylim([-5 5])
title("Input Signal");
xlabel('time samples n');
ylabel('x(n)');
grid on;

subplot(1,2,2);
plot(t, y);
ylim([-5 5])
title("Comprosser Output");
xlabel('time samples n');
ylabel('y(n)');
grid on;
figure();
subplot(1,2,1);
plot(t, c);
ylim([0 3])
title("Control Signal");
xlabel('time samples n');
ylabel('c(n)');
grid on;

subplot(1,2,2);
plot(t, G);
ylim([0 1.25])
title("Compressor Gain");
xlabel('time samples n');
ylabel('G(n)');
grid on;
