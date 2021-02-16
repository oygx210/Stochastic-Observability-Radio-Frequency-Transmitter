% PAPER: Stochastic Observability and Uncertainty Characterizationin
% Simulatneous Reciever and Transmitter Localization
% DATE: November 17th, 2020
% AUTHOR: Alex Nguyen
% DESCRIPTION: Replicate Figure 3 with the EKF-based radio SLAM model and
% parameters given in table III 

clc; clear; close all;
                            
%------------------------Simulation Parameters----------------------------%
% Reciever
x_rx0 = [0 0 10 10 100 10]';                % Initial Rx State
h0_rx = 9.4e-20; hneg2_rx = 3.8e-21;       % Rx Power-Law Coefficients 

% Radio Frequency (RF) Transmitter
rs(1:2, 1) = [-110, 240]';                  % RF Tx 1-5 Initial Conditions
rs(1:2, 2) = [-150, 340]'; 
rs(1:2, 3) = [-215, -60]';
rs(1:2, 4) = [-75, 105]';
rs(1:2, 5) = [-5, 80]';
x_s0 = zeros(4, length(rs));   % Preallocation
for ii = 1:5
    % RF Tx 1-5 Initial States
    x_s0(:, ii) = [rs(:, ii); 10; 1]';
end
h0_s = 8.0e-20; hneg2_s = 4.0e-23;         % RF Tx Power-Law Coefficients                             

% Speed of Light [m/s]
c = 299792458;                              

% Simulation Time
T = 10e-3;                                  % Sampling Period [s]
w = 0.1;                                    % Constant Turn Rate [rad/s] 
t = [0:T:80]';                              % Experiment Time Duration [s]
SimL = length(t);                           % Simulation Time Length

%-------------------------Power Spectral Density--------------------------%
S_wts = h0_s/2; S_wtsdot = 2*pi^2*hneg2_s;    % RF Transmitter
S_wtr = h0_rx/2; S_wtrdot = 2*pi^2*hneg2_rx;  % Reciever
Sw = 0.01;                                    % Process Noise [m^2*rad^2/s^2]

%-------------------------RF Transmitter Dynamics-------------------------%
Fclk = [1 T; ...             % RF Clock "Dynamics Jacobian"  
        0 1];
     
Fs = blkdiag(eye(2), Fclk);  % "Jacobian" for RF Dynamics

Qclk_s = [S_wts*T + S_wtsdot*T^3/3, S_wtsdot*T^2/2; ...
          S_wtsdot*T^2/2          , S_wtsdot*T];  % Process Noise Covariance
          
Qs = blkdiag(zeros(2), c^2*Qclk_s);               % White Noise Covariance

%----------------------------Reciever Dynamics----------------------------%
Fpv = [1, 0,  sin(w*T)/w     , -(1 - cos(w*T))/w; ...
       0, 1, (1 - cos(w*T))/w,   sin(w*T)/w; ...
       0, 0, cos(w*T)        ,  -sin(w*T); ...
       0, 0, sin(w*T)        ,   cos(w*T)];    % Constant Turn Rate Model

Fr = blkdiag(Fpv, Fclk);                       % "Jacobian" for Rx Dynamics

Qpv = Sw*[2*(w*T - sin(w*T))/w^3, 0,                       (1 - cos(w*T))/w^2,   (w*T - sin(w*T))/w^2; ...
          0,                      2*(w*T - sin(w*T))/w^3, -(w*T - sin(w*T))/w^2, (1 - cos(w*T))/w^2; ...
         (1 - cos(w*T))/w^2,    -(w*T - sin(w*T))/w^2,      T,                    0; ...
         (w*T - sin(w*T))/w^2,   (1 - cos(w*T))/w^2,        0,                    T]; % Position and Velocity Process Noise

Qclk_r = [S_wtr*T + S_wtrdot*T^3/3, S_wtrdot*T^2/2; ...
          S_wtrdot*T^2/2          , S_wtrdot*T];  % Process Noise Covariance
          
Qr = blkdiag(Qpv, c^2*Qclk_r);  % White Noise Covariance

%--------------------------EKF State Estimation---------------------------%
% Number of States
nx = 26;                                       % Full System States (Rx & RF Tx 1-5)
nz = 5;                                        % RF Tx 1 - 5 Measurement States

% Augmented System
Phi_s = blkdiag(Fs, Fs, Fs, Fs, Fs);           % RF 1 - 5 Matrices
Fk = blkdiag(Fr, Phi_s);                       % "Jacobian" State Matrix  
f = @(x) Fk*x;                                 % "Evolution" State Function

% Noise Covariance Matrices and Standard Deviations (e.g. wk & vk)
R = 20*eye(nz);                                % Measurment Noise Covariance   
r = sqrt(diag(R));                             % Measurement Noise St. Dev. 
Q = blkdiag(Qr, Qs, Qs, Qs, Qs, Qs);           % Process Noise Covariance
q = sqrt(diag(Q));                             % Process Noise St. Dev.

% Estimation Error Matrices
P_rx0 = blkdiag(0, 0, 0, 0, 0, 0);                      % Initial Rx Covariance
P_s0 = 1e2*blkdiag(1, 1, 30, 3);                        % RF Tx 1-5 Initial Covariance
P_est0 = blkdiag(P_rx0, P_s0, P_s0, P_s0, P_s0, P_s0);  % Full System Covariance

% EKF State Initialization
x_0 = [x_rx0;  x_s0(:)];                      % True System States  
xz = x_0 + sqrt(diag(P_est0)).*randn(nx, 1);  % Estimate System States      

% Preallocation
z = zeros(nz, SimL);                              
x_est = zeros(nx, SimL); 
P_est = x_est;
x_true = x_est;

for k = 1:SimL
    % Euclidean Norm (Distance)
    D1 = @(x) norm(x(1:2) -  x(7:8));
    D2 = @(x) norm(x(1:2) -  x(11:12));
    D3 = @(x) norm(x(1:2) -  x(15:16));
    D4 = @(x) norm(x(1:2) -  x(19:20));
    D5 = @(x) norm(x(1:2) -  x(23:24));
    
    % Measurment Propogation for RF Tx 1-5
    h1 = @(x) D1(x) + (x(5) - x(9));
    h2 = @(x) D2(x) + (x(5) - x(13));
    h3 = @(x) D3(x) + (x(5) - x(17));
    h4 = @(x) D4(x) + (x(5) - x(21));
    h5 = @(x) D5(x) + (x(5) - x(25));
        
    % Jacobian Component for Rx
    Jrx1 = @(x) [(x(1) - x(7))/D1(x),  (x(2) - x(8))/D1(x),  0, 0, 1, 0];
    Jrx2 = @(x) [(x(1) - x(11))/D2(x), (x(2) - x(12))/D2(x), 0, 0, 1, 0];
    Jrx3 = @(x) [(x(1) - x(15))/D3(x), (x(2) - x(16))/D3(x), 0, 0, 1, 0];
    Jrx4 = @(x) [(x(1) - x(19))/D4(x), (x(2) - x(20))/D4(x), 0, 0, 1, 0];
    Jrx5 = @(x) [(x(1) - x(23))/D5(x), (x(2) - x(24))/D5(x), 0, 0, 1, 0];
    
    % Jacobian Components for RF Tx 1-5
    Jrf1 = @(x) [(x(7) - x(1))/D1(x), (x(8) - x(2))/D1(x), -1, 0];
    Jrf2 = @(x) [(x(11) - x(1))/D2(x), (x(12) - x(2))/D2(x), -1, 0];
    Jrf3 = @(x) [(x(15) - x(1))/D3(x), (x(16) - x(2))/D3(x), -1, 0];
    Jrf4 = @(x) [(x(19) - x(1))/D4(x), (x(20) - x(2))/D4(x), -1, 0];
    Jrf5 = @(x) [(x(23) - x(1))/D5(x), (x(24) - x(2))/D5(x), -1, 0];
    
    % Jacobian
    Hk = @(x) [Jrx1(x), Jrf1(x), zeros(1, 16); ...
               Jrx2(x), zeros(1, 4), Jrf2(x), zeros(1, 12); ...
               Jrx3(x), zeros(1, 8), Jrf3(x), zeros(1, 8); ...
               Jrx4(x), zeros(1, 12), Jrf4(x), zeros(1, 4); ...
               Jrx5(x), zeros(1, 16), Jrf5(x)];
        
    % True Pseudorange Measurment RF Tx 1 -5
    z_true = [h1(x_0); h2(x_0); h3(x_0); h4(x_0); h5(x_0)];
    z(:, k) = z_true + r.*randn(nz, 1);  

    % True State Values (Rx and SOP 2)
    x_true(:, k) = x_0;                   
    
    if k == 1       
        % Prediction
        x_estn = xz;
        P_estn = P_est0;
        
        % Update
        H = Hk(x_estn);
        z_est = [h1(x_estn); h2(x_estn); h3(x_estn); h4(x_estn); h5(x_estn)];
        yk_res = z(:, k) - z_est;
        Sk = H*P_estn*H' + R;
        Kk = P_estn*H'*inv(Sk);
        
%         P_xy = P_estn*H';
%         P_yy = H*P_estn*H' + R;
%         Kk = P_xy/P_yy;
         
        % Correction
        xz = x_estn + Kk*yk_res;
        P_est0 = (eye(nx) - Kk*H)*P_estn;

%         xz = x_estn + Kk*yk_res;
%         A = eye(nx) - Kk*H;
%         P_est0 = A*P_estn*A' + Kk*R*Kk';
        
        % Save Estimate Values
        x_est(:, k) = xz;
        P_est(:, k) = diag((P_est0));
        
    else
        % Prediction
        x_estn = f(xz);
        P_estn = Fk*P_est0*Fk' + Q;
        
        % Update
        H = Hk(x_estn);
        z_est = [h1(x_estn); h2(x_estn); h3(x_estn); h4(x_estn); h5(x_estn)];
        yk_res = z(:, k) - z_est;
        Sk = H*P_estn*H' + R;
        Kk = P_estn*H'*inv(Sk);
        
%         P_xy = P_estn*H';
%         P_yy = H*P_estn*H' + R;
%         Kk = P_xy/P_yy;
         
        % Correction
        xz = x_estn + Kk*yk_res;
        P_est0 = (eye(nx) - Kk*H)*P_estn;
        
%         xz = x_estn + Kk*yk_res;
%         A = eye(nx) - Kk*H;
%         P_est0 = A*P_estn*A' + Kk*R*Kk';
         
        % Save Values
        x_est(:, k) = xz;
        P_est(:, k) = diag((P_est0));
        
    end
    
    % Next Step
    x_0 = f(x_0) + q.*randn(nx, 1);
    
end

% Estimation Error 
x_tilde = x_true - x_est;    % Full System's Error Trajectories
x_P = sqrt(P_est);           % Full System's Error Variance Bounds

%-------------------------Plot Estimation Results-------------------------%
figure;  % Position Error Trajectory Rx
subplot(5,2,1)
x = plot(t, x_tilde(1, :), 'linewidth', 2); hold on;
plot(t, 2*x_P(1, :), 'r--',  'linewidth',1.5); 
plot(t, -2*x_P(1, :), 'r--', 'linewidth',1.5); hold off;
ylabel('$\tilde{x}_r$ [m]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);
legend('$\tilde{x}_1(t_k)$','$\pm 2 \sigma_1(t_k)$','interpreter','latex')

subplot(5,2,2)
x = plot(t, x_tilde(2, :), 'linewidth', 2); hold on;
plot(t, 2*x_P(2, :), 'r--', 'linewidth',1.5); 
plot(t, -2*x_P(2, :), 'r--', 'linewidth',1.5); hold off;
ylabel('$\tilde{y}_r$ [m]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);
legend('$\tilde{x}_2(t_k)$','$\pm 2 \sigma_2(t_k)$','interpreter','latex')

subplot(5,2,3)  % Velocity Error Trajectory Rx
x = plot(t, x_tilde(3, :), 'linewidth', 2); hold on;
plot(t, 2*x_P(3, :), 'r--', 'linewidth',1.5); 
plot(t, -2*x_P(3, :), 'r--', 'linewidth',1.5); hold off;
ylabel('$\tilde{\dot{x}}_r$ [m/s]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);
legend('$\tilde{x}_3(t_k)$','$\pm 2 \sigma_3(t_k)$','interpreter','latex')

subplot(5,2,4)
x = plot(t, x_tilde(4, :), 'linewidth', 2); hold on;
plot(t, 2*x_P(4, :), 'r--' , 'linewidth',1.5); 
plot(t, -2*x_P(4, :), 'r--' , 'linewidth',1.5); hold off;
ylabel('$\tilde{\dot{y}}_r$ [m/s]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);
legend('$\tilde{x}_4(t_k)$','$\pm 2 \sigma_4(t_k)$','interpreter','latex')

subplot(5,2,5)  % Clock Bias Error Trajectory Rx
x = plot(t, x_tilde(5, :), 'linewidth', 2); hold on;
plot(t, 2*x_P(5, :), 'r--', 'linewidth',1.5); 
plot(t, -2*x_P(5, :), 'r--', 'linewidth',1.5); hold off;
ylabel('$c \delta_{t_t}$ [m]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);
legend('$\tilde{x}_5(t_k)$','$\pm 2 \sigma_5(t_k)$','interpreter','latex')

subplot(5,2,6)
x = plot(t, x_tilde(6, :), 'linewidth', 2); hold on;
plot(t, 2*x_P(6, :), 'r--' , 'linewidth',1.5); 
plot(t, -2*x_P(6, :), 'r--' , 'linewidth',1.5); hold off;
ylabel('$c \delta_{\dot{t}_r}$ [m]', 'interpreter', 'latex')
ax = x.Parent; set(ax, 'XTick', []);
legend('$\tilde{x}_6(t_k)$','$\pm 2 \sigma_6(t_k)$','interpreter','latex')

subplot(5,2,7)  % Position Error Trajectory RF Tx 1 
x = plot(t, x_tilde(7, :), 'linewidth', 2); hold on;
plot(t, 2*x_P(7, :), 'r--', 'linewidth',1.5); 
plot(t, -2*x_P(7, :), 'r--', 'linewidth',1.5); hold off;
ylabel('$\tilde{x}_{s1}$ [m]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);
legend('$\tilde{x}_7(t_k)$','$\pm 2 \sigma_7(t_k)$','interpreter','latex')

subplot(5,2,8)
x = plot(t, x_tilde(8, :), 'linewidth', 2); hold on;
plot(t, 2*x_P(8, :), 'r--' , 'linewidth',1.5); hold on;
plot(t, -2*x_P(8, :), 'r--' , 'linewidth',1.5); hold off;
ylabel('$\tilde{y}_{s1}$ [m]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);
legend('$\tilde{x}_8(t_k)$','$\pm 2 \sigma_8(t_k)$','interpreter','latex')

subplot(5,2,9)  % Clock Bias Error Trajectory RF Tx 1
plot(t, x_tilde(9, :), 'linewidth', 2); hold on;
plot(t, 2*x_P(9, :), 'r--' , 'linewidth',1.5); hold on;
plot(t, -2*x_P(9, :), 'r--' , 'linewidth',1.5); hold off;
xlabel('Time [s]'); ylabel('$\tilde{c \delta t}_{s1}$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_9(t_k)$','$\pm 2 \sigma_9(t_k)$','interpreter','latex')

subplot(5,2,10)
plot(t, x_tilde(10, :), 'linewidth', 2); hold on;
plot(t, 2*x_P(10, :), 'r--' , 'linewidth',1.5); hold on;
plot(t, -2*x_P(10, :), 'r--' , 'linewidth',1.5); hold off;
xlabel('Time [s]'); ylabel('$\tilde{c \delta \dot{t}}_{s1}$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_10(t_k)$','$\pm 2 \sigma_10(t_k)$','interpreter','latex')
sgtitle('System $\Sigma$: Estimation Error Trajectories with $\pm 2\sigma$ Bounds','interpreter','latex')

for ii = 1:10
    sph = subplot(5,2,ii); % Resize Subplots
    dx0 = -0.05;
    dy0 = -0.025;
    dwithx = 0.03;
    dwithy = 0.03;
    set(sph,'position',get(sph,'position') + [dx0, dy0, dwithx, dwithy])
end

% Simulated Environment Used in EKF
figure;
plot(x_true(1, :), x_true(2, :), 'k', 'linewidth', 2); hold on;
for ii = 7:4:23
    plot(x_true(ii, :), x_true(ii + 1, :), 'rs', 'linewidth', 5);
end
hold off; 
title('System $\Sigma$: Simulated Environment for UAV', 'interpreter', 'latex', 'fontsize', 14)
xlabel('x [m]'); ylabel('y [m]');
legend('Reciever Trajectory', 'RF Tx 1-5 Locations', 'location', 'best')
grid on;
