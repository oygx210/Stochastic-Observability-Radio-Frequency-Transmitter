% PAPER: Stochastic Observability and Uncertainty Characterizationin
% Simulatneous Reciever and Transmitter Localization
% DATE: November 21st, 2020
% AUTHOR: Alex Nguyen
% DESCRIPTION: Replicate Figure 1 with the Reduced-Order KF-based radio
% SLAM model and parameters given in table II

clc; clear; close all;
                            
%------------------------Simulation Parameters----------------------------%
% Reciever Clock Error
xr_clk0 = [100 10]';             % Initial Rx State
h0_rx = 9.4e-20; hneg2_rx = 0;  % Rx Power-Law Coefficients 

% Radio Frequency (RF) Transmitter Clock Error
xs_clk0 = zeros(2, 5);           % Preallocation
for ii = 1:5
    % RF Tx 1-5 Initial States
    xs_clk0(:, ii) = [10, 1]';
end
h0_s = 8.0e-20; hneg2_s = 0;    % RF Tx Power-Law Coefficients   

% Total System Clock Error
x_clk0 = [xr_clk0; xs_clk0(:)];

% Speed of Light [m/s]
c = 299792458;  

% Simulation Time
T = 10e-3;                       % Sampling Period [s]
t = (0:T:80)';                   % Experiment Time Duration [s]
SimL = length(t);                % Simulation Length

%-------------------------Power Spectral Density--------------------------%
S_wts = h0_s/2; S_wtsdot = 2*pi^2*hneg2_s;    % RF Transmitter
S_wtr = h0_rx/2; S_wtrdot = 2*pi^2*hneg2_rx;  % Reciever

%-------------------------Simplified LTI System---------------------------%  
% Covariance Matrices
Qclk_r = [S_wtr*T + S_wtrdot*T^3/3, S_wtrdot*T^2/2; ... % Rx Process Noise 
          S_wtrdot*T^2/2          , S_wtrdot*T];  

Qclk_s = [S_wts*T + S_wtsdot*T^3/3, S_wtsdot*T^2/2; ... % RF Tx Process Noise
          S_wtsdot*T^2/2          , S_wtsdot*T];  
                
Qclk = c^2*blkdiag(Qclk_r, Qclk_s, Qclk_s, Qclk_s, Qclk_s, Qclk_s); % White Noise Process     

% Observation Model
hclk = [1, 0]';   % Clock Bias "Jacobian" 

Hclk = [hclk' -hclk' zeros(1, 8); ...  % Observation Measurement "Jacobian"
        hclk' zeros(1, 2) -hclk' zeros(1, 6); ...
        hclk' zeros(1, 4) -hclk' zeros(1, 4); ...
        hclk' zeros(1, 6) -hclk' zeros(1, 2); ...
        hclk' zeros(1, 8) -hclk'];

% Clock Dynamics
Fclk = [1 T; ...  % "Clock Dynamics"  
        0 1];
       
Phi_clk = blkdiag(Fclk, Fclk, Fclk, Fclk, Fclk, Fclk);  % State Transition Model 

f = @(x) Phi_clk*x;  % State Evolution 

h = @(x) Hclk*x;     % Observation Evolution

%----------------------Reduced-Order Kalman Filter------------------------%
% Design Matrix
g = [0, 1]';  

G = [eye(2) zeros(2) zeros(2) zeros(2) zeros(2) zeros(2); ...  
      g'     -g'    zeros(1, 8); ...
      g'  zeros(1, 2) -g' zeros(1, 6); ...  
      g'  zeros(1, 4) -g' zeros(1, 4); ...
      g'  zeros(1, 6) -g' zeros(1, 2); ...
      g'  zeros(1, 8) -g']; 

% Reduced-Order Matrices 
L1L2 = inv([Hclk; G]);  % Augmented 
L1 = L1L2(:, 1:5);
L2 = L1L2(:, 6:12);

% Standard Basis Vectors
e = L2'; 
e1_ro = e(:, 1); e2_ro = e(:, 2); e3  = e2_ro - e(:, 4); 
e1_clk = [1 zeros(1, 11)]';

% Posterior Estimation Error Covariance Matrices
Psi = G*Phi_clk*L2;
Xi = Hclk*Phi_clk*L2;
Rro = Hclk*Qclk*Hclk';

%----------------------Initializing Estimation----------------------------%
% Number of States
nx = length(x_clk0);             % Full System States (Rx & RF Tx 1-5)
nz = length(xs_clk0);            % RF Tx 1 - 5 Measurement States
nro = size(G, 1);                % Reduced-Order System States

% Noise Covariance Matrices and Standard Deviations (e.g. wk & vk)
Q = Qclk;                        % Process Noise Covariance
q = sqrt(diag(Q));               % Process Noise St. Dev.
R = 0*eye(nz);                   % Measurment Noise Covariance   
r = sqrt(diag(R));               % Measurement Noise St. Dev. 

% Estimation Error Covariance Matrices
P_clk0 = 1e2*blkdiag(0, 0, 3, 0.3, 3, 0.3, 3, 0.3, 3, 0.3, 3, 0.3);  % Clock System 
P_xro0 = G*P_clk0*G';  % Reduced-Order System

% Reduced-Order State Vector Initialization
xro0 = G*x_clk0;                                       % True System States
xro_est0 = G*x_clk0 + sqrt(diag(P_xro0)).*randn(nro, 1); % Estimate System States 

% Preallocation
Pro_est = zeros(nro, SimL);  % Reduced-Order Covarance
Pclk_est = zeros(nx, SimL);  % Rx & RF Tx 1-5 Clock Bias Covariance
gamma = zeros(SimL, 1);      % Divergence Rate of Estimation Error Variance
xclk_true = zeros(nx, SimL); % True Clock States
xclk_est = xclk_true;        % Estimate Clock States
xro_true = zeros(nro, SimL); % True Reduced-Order States
xro_est = xro_true;         % Estimate Reduced-Order States

%----------------------Reduced-Order KF Estimation------------------------%
for k = 1:SimL  
    % True States  
    z_clk = h(x_clk0) + r.*randn(nz, 1); % Pseudorange Measurment
    x_clk = L1*z_clk + L2*xro0;          % Clock Dynamics
    xclk_true(:, k) = x_clk0;
    xro_true(:, k) = xro0;
    
    % Estimated State Values
    xclk_est0 = L1*z_clk + L2*xro_est0; % Clock Dynamics
    xro_est(:, k) = xro_est0;
    
    % Predict Reduced-Order States
    x_estn = xro_est0;
    yk_res = Xi*xro0 + Hclk*sqrt(Qclk)*randn(nx, 1);
    
    % Estimation Error Covariance, P_xr0(k|k)
    P_estclk = P_clk0;       % Full System
    P_estn = G*P_estclk*G';  % Reduced-Order System
    
    % Posterior Estimation Error
    Lambda = (Psi*P_estn*Xi' + G*Qclk*Hclk')/(Xi*P_estn*Xi' + Rro);
    A = (Psi - Lambda*Xi)*P_estn*(Psi - Lambda*Xi)'; % Term 1
    B = G*Qclk*G' - G*Qclk*Hclk'*Lambda';            % Term 2
    C = -Lambda*Hclk*Qclk'*G' + Lambda*Rro*Lambda';  % Term 3
    
    P_xro = A + B + C;
    P_clk = L2*P_xro*L2';
    
    % Correct Reduced-Order State
    xro_est0 = Psi*x_estn + G*Phi_clk*L1*z_clk + Lambda*(yk_res - Xi*x_estn);
    
    % Save Covariance Estimate Values
    xclk_est(:, k) = xclk_est0;
    Pro_est(:, k) = diag(P_xro);
    Pclk_est(:, k) = diag(P_clk);
    gamma(k, 1) = e1_clk'*(P_clk - P_estclk)*e1_clk;
    
    % Correction
    P_clk0 = L2*P_xro*L2';
    P_estn = P_xro;
    
    % Next Step
    yk1 = Xi*xro0 + Hclk*Phi_clk*L1*z_clk + Hclk*sqrt(Qclk)*randn(nx, 1);
    xro0 = Psi*xro0 + G*Phi_clk*L1*z_clk + G*(q.*randn(nx, 1));
    x_clk0 = L1*yk1 + L2*xro0;
end

% Estimation Error 
xro_tilde = xro_true - xro_est;     % Reduced-Order Error Variance Bounds
xro_P = sqrt(Pro_est);              % Reduced-Order Error Variance Bounds
xclk_tilde = xclk_true - xclk_est;  % Clock Bias System's Error Trajectories
xclk_P = sqrt(Pclk_est);            % Clock Bias Error Variance Bounds
qr = ones(SimL, 1)*c^2*S_wtr*T;     % Divergence Rate Bound

%-------------------------Plot Estimation Results-------------------------%
figure;  
subplot(3,2,1)  % Reduced-Order 
x = plot(t, xro_tilde(1, :), 'linewidth', 2); hold on; 
plot(t, 2*xro_P(1, :), 'r--',  'linewidth',1.5); 
plot(t, -2*xro_P(1, :), 'r--', 'linewidth',1.5); hold off;
ylabel('$\tilde{x}_{ro1}$ [m]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);

subplot(3,2,3)
x = plot(t, xro_tilde(3, :), 'linewidth', 2); hold on; 
plot(t, 2*xro_P(3, :), 'r--', 'linewidth',1.5); 
plot(t, -2*xro_P(3, :), 'r--', 'linewidth',1.5); hold off;
ylabel('$\tilde{x}_{ro3}$ [m/s]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);
ylim([-0.75 0.75])

subplot(3,2,2)  % Rx Clock Bias
x = plot(t, xclk_tilde(1, :), 'linewidth', 2); hold on; 
plot(t, 2*xclk_P(1, :), 'r--', 'linewidth',1.5); 
plot(t, -2*xclk_P(1, :), 'r--', 'linewidth',1.5); hold off;
ylabel('$c \tilde{\delta t}_{r}$ [m]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);

subplot(3,2,4) % RF Tx Clock Bias
x = plot(t, xclk_tilde(3, :), 'linewidth', 2); hold on; 
plot(t, 2*xclk_P(3, :), 'r--', 'linewidth',1.5); 
plot(t, -2*xclk_P(3, :), 'r--', 'linewidth',1.5); hold off;
ylabel('$c \tilde{\delta t}_{s1}$ [m]', 'interpreter', 'latex');
ax = x.Parent; set(ax, 'XTick', []);

subplot(3,2,6)
plot(t, xclk_tilde(4, :), 'linewidth', 2); hold on; 
plot(t, 2*xclk_P(4, :), 'r--', 'linewidth',1.5); hold on;
plot(t, -2*xclk_P(4, :), 'r--', 'linewidth',1.5); hold off;
xlabel('Time [s]');
ylabel('$c \tilde{\delta \dot{t}}_{s1}$ [m/s]', 'interpreter', 'latex');
legend('Estimation Error', '$\pm 2 \sigma$', 'location', 'southeast', 'interpreter','latex') 
ylim([-0.75 0.75])

subplot(3,2,5)  % Divergence Rate
plot(t, gamma, 'linewidth', 2); hold on;
plot(t, qr, 'k--', 'linewidth', 1.5); hold off;
xlabel('Time [s]'); ylabel('$\gamma$(k)', 'interpreter', 'latex');
legend('Divergence Rate', '$q_r$ Limit', 'location', 'southeast', 'interpreter','latex') 
ylim([4.221e-5 4.2245e-5])
sgtitle('System $\Sigma_{III}$: Estimation Error Trajectories with $\pm 2\sigma$ Bounds','interpreter','latex')

for ii = 1:6
    sph = subplot(3,2,ii); % Resize Subplots
    dx0 = -0.05;
    dy0 = -0.025;
    dwithx = 0.03;
    dwithy = 0.03;
    set(sph,'position',get(sph,'position') + [dx0, dy0, dwithx, dwithy])
end

