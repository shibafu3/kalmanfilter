rho = 1.23; g = 9.81; eta = 6e3;
M = 3e4; a = 3e4;

T = 0.5;
EndTime = 30;
time = 0:T:EndTime;
N = EndTime/T+1;
n = 3;
R = 4e3;
Q = 0;
B = [0; 0; 0];

f = @(x) [x(1)+T*x(2);
    x(2)+T*(0.5*rho*exp(-x(1)/eta)*x(2)^2*x(3)-g);
    x(3)];
h = @(x) sqrt(M^2+(x(1)-a).^2);

rng('default')
w = randn(N, 1)*sqrtm(R);
x = zeros(N, n); y = zeros(n, 1);
x(1,:) = [90000;-6000;0.003]';
y(1) = h(x(1,:))+w(1);

for k=2:N
    x(k,:) = f(x(k-1,:));
    y(k) = h(x(k,:))+w(k);
end

xhat_ukf=zeros(N, 3);
xhat_ukf(1,:) = x(1,:);
P_ukf = [9e3 0 0;0 4e5 0;0 0 0.4];

for k = 2:N
    [xhat_ukf(k,:), P_ukf] = ukf(f, h, B, Q, R, y(k), xhat_ukf(k-1,:), P_ukf);
end

figure(1), clf
for p=1:3
    subplot(3,1,p)
    plot(time,x(:,p));
    xlabel('Time [s]'), ylabel(sprintf('xd%', p))
end
figure(2), clf
for p=1:3
    subplot(3,1,p)
    plot(time,x(:,p), 'k', ...
    time, xhat_ukf(:,p),'b-');
    xlabel('Time [s]'), ylabel(sprintf('xd%', p))
    legend('true', 'ukf')
end

function [xhat_new, P_new, G] = ukf(f, h, B, Q, R, y, xhat, P)
    xhat = xhat(:); y = y(:);
    [xhatm, Pm] = ut(f, xhat, P);
    Pm          = Pm + B*Q*B';
    [yhatm, Pyy, Pxy] = ut(h, xhatm, Pm);
    G = Pxy/(Pyy +R);
    xhat_new = xhatm + G*(y - yhatm);
    P_new = Pm - G*Pxy';
end

function [ym, Pyy, Pxy] = ut(f, xm, Pxx)
    xm = xm(:);
    mapcols = @(f, x) cell2mat(cellfun(f, mat2cell(x, size(x, 1), ones(1, size(x,2))), 'UniformOutput', false));
    n = length(xm);
    kappa = 3-n;
    w0 = kappa/(n+kappa);
    wi = 1/(2*(n+kappa));
    W = diag([w0; wi*ones(2*n, 1)]);
    L = chol(Pxx);
    X = [xm'; ones(n,1)*xm'+sqrt(n+kappa)*L; ones(n,1)*xm'-sqrt(n+kappa)*L];
    Y = mapcols(f, X')';
    ym = sum(W*Y)';
    Yd = bsxfun(@minus, Y, ym');
    Xd = bsxfun(@minus, X, xm');
    Pyy = Yd'*W*Yd;
    Pxy = Xd'*W*Yd;
end