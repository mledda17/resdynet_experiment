%% test_n4sid_linear_system.m
clear;
clc;
close all;

rng(42);

%% ============================================================
% True discrete-time LTI system
% =============================================================

Ts = 1.0;

A_true = [
    0.7  1.0  0.0;
    0.0  0.5  1.0;
    0.0  0.0  0.3
];

B_true = [
    1.0;
    1.0;
    1.0
];

C_true = [
    1.0  1.0  1.0
];

D_true = 0.0;

nx = size(A_true,1);

sys_true = ss(A_true,B_true,C_true,D_true,Ts);

fprintf('========================================\n');
fprintf('TRUE SYSTEM\n');
fprintf('========================================\n');

disp('A_true =');
disp(A_true);

disp('B_true =');
disp(B_true);

disp('C_true =');
disp(C_true);

disp('D_true =');
disp(D_true);

eig_true = eig(A_true);

disp('True eigenvalues:');
disp(eig_true);

fprintf('Spectral radius: %.6f\n',max(abs(eig_true)));
fprintf('Stable: %d\n',all(abs(eig_true)<1));

%% ============================================================
% Controllability and observability
% =============================================================

Co = ctrb(A_true,B_true);
Ob = obsv(A_true,C_true);

fprintf('\nControllability rank: %d / %d\n',rank(Co),nx);
fprintf('Observability rank : %d / %d\n',rank(Ob),nx);

%% ============================================================
% Dataset configuration
% =============================================================

N_train = 10000;
N_test  = 3000;

u_max = 1.0;

%% ============================================================
% Rich training excitation
% =============================================================

u_train = generate_rich_input(N_train,u_max,1);

%% ============================================================
% Independent test excitation
% =============================================================

u_test = generate_rich_input(N_test,u_max,100);

%% ============================================================
% Simulate TRUE system
% =============================================================

x0_train = zeros(nx,1);
x0_test  = zeros(nx,1);

[y_train,~,x_train] = lsim( ...
    sys_true, ...
    u_train, ...
    (0:N_train-1)'*Ts, ...
    x0_train);

[y_test,~,x_test] = lsim( ...
    sys_true, ...
    u_test, ...
    (0:N_test-1)'*Ts, ...
    x0_test);

%% ============================================================
% Optional measurement noise
%
% Start with noise_std = 0 to test the ideal case.
% Then increase it later.
% =============================================================

noise_std = 0.0;

y_train_noisy = y_train + noise_std*randn(size(y_train));
y_test_noisy  = y_test  + noise_std*randn(size(y_test));

%% ============================================================
% Create identification datasets
% =============================================================

data_train = iddata( ...
    y_train_noisy, ...
    u_train, ...
    Ts);

data_test = iddata( ...
    y_test_noisy, ...
    u_test, ...
    Ts);

%% ============================================================
% N4SID identification
% =============================================================

fprintf('\n========================================\n');
fprintf('RUNNING N4SID\n');
fprintf('========================================\n');

opt = n4sidOptions;

% Display progress if desired
opt.Display = 'on';

% Estimate a state-space model with the TRUE order
sys_n4sid = n4sid( ...
    data_train, ...
    nx, ...
    opt);

%% ============================================================
% Extract identified matrices
% =============================================================

A_hat = sys_n4sid.A;
B_hat = sys_n4sid.B;
C_hat = sys_n4sid.C;
D_hat = sys_n4sid.D;

fprintf('\n========================================\n');
fprintf('IDENTIFIED SYSTEM\n');
fprintf('========================================\n');

disp('A_hat =');
disp(A_hat);

disp('B_hat =');
disp(B_hat);

disp('C_hat =');
disp(C_hat);

disp('D_hat =');
disp(D_hat);

%% ============================================================
% Eigenvalue comparison
% =============================================================

eig_hat = eig(A_hat);

fprintf('\n========================================\n');
fprintf('EIGENVALUE COMPARISON\n');
fprintf('========================================\n');

disp('True eigenvalues:');
disp(sort(eig_true));

disp('N4SID eigenvalues:');
disp(sort(eig_hat));

fprintf('True spectral radius : %.8f\n',max(abs(eig_true)));
fprintf('N4SID spectral radius: %.8f\n',max(abs(eig_hat)));

fprintf('True model stable : %d\n',all(abs(eig_true)<1));
fprintf('N4SID model stable: %d\n',all(abs(eig_hat)<1));

%% ============================================================
% Match eigenvalues and calculate errors
% =============================================================

eig_true_sorted = sort(eig_true);
eig_hat_sorted  = sort(eig_hat);

eig_abs_error = abs(eig_true_sorted - eig_hat_sorted);

fprintf('\nEigenvalue absolute errors:\n');
disp(eig_abs_error);

fprintf('Maximum eigenvalue error: %.6e\n',max(eig_abs_error));

%% ============================================================
% Open-loop simulation on test set
% =============================================================

[y_hat_test,~,~] = lsim( ...
    sys_n4sid, ...
    u_test, ...
    (0:N_test-1)'*Ts);

err = y_test - y_hat_test;

RMSE = sqrt(mean(err.^2));

NRMSE = RMSE / std(y_test);

fprintf('\n========================================\n');
fprintf('TEST OPEN-LOOP PERFORMANCE\n');
fprintf('========================================\n');

fprintf('RMSE       : %.8e\n',RMSE);
fprintf('NRMSE      : %.8e\n',NRMSE);
fprintf('NRMSE [%%]  : %.6f %%\n',100*NRMSE);

%% ============================================================
% Transfer-function / frequency-response comparison
% =============================================================

G_true = tf(sys_true);
G_hat  = tf(sys_n4sid);

fprintf('\nTRUE transfer function:\n');
G_true

fprintf('IDENTIFIED transfer function:\n');
G_hat

%% ============================================================
% Attempt similarity transformation
%
% The state-space realization returned by N4SID does NOT need to
% have the same coordinates as the true system.
%
% Ideally:
%
%   A_hat = T A_true T^{-1}
%   B_hat = T B_true
%   C_hat = C_true T^{-1}
%
% We estimate T from observability matrices.
% =============================================================

O_true = obsv(A_true,C_true);
O_hat  = obsv(A_hat,C_hat);

% From:
%
%   O_hat = O_true T^{-1}
%
% therefore:
%
%   T = inv(O_hat) * O_true
%
% when square/full-rank.
%
% Here ny=1 and nx=3, so O is 3x3.

if rank(O_true)==nx && rank(O_hat)==nx

    T = O_hat \ O_true;

    A_sim = T*A_true/T;
    B_sim = T*B_true;
    C_sim = C_true/T;

    fprintf('\n========================================\n');
    fprintf('SIMILARITY-TRANSFORM CHECK\n');
    fprintf('========================================\n');

    disp('Estimated T =');
    disp(T);

    fprintf('||A_hat - T*A_true/T||_F = %.6e\n', ...
        norm(A_hat-A_sim,'fro'));

    fprintf('||B_hat - T*B_true||_F = %.6e\n', ...
        norm(B_hat-B_sim,'fro'));

    fprintf('||C_hat - C_true/T||_F = %.6e\n', ...
        norm(C_hat-C_sim,'fro'));

    fprintf('|D_hat - D_true| = %.6e\n', ...
        norm(D_hat-D_true,'fro'));

else

    warning('Observability matrices are not full rank.');

end

%% ============================================================
% Plots
% =============================================================

t_test = (0:N_test-1)'*Ts;

figure;
plot(t_test,y_test,'LineWidth',1.1);
hold on;
plot(t_test,y_hat_test,'--','LineWidth',1.1);
grid on;

xlabel('Time');
ylabel('y');
legend('True system','N4SID','Location','best');
title('Open-loop test simulation');

figure;
plot(t_test,err,'LineWidth',1.0);
grid on;

xlabel('Time');
ylabel('y-\hat{y}');
title('Open-loop identification error');

%% Eigenvalues in complex plane

theta = linspace(0,2*pi,500);

figure;
plot(cos(theta),sin(theta),'k--');
hold on;

plot(real(eig_true),imag(eig_true), ...
    'o','MarkerSize',9,'LineWidth',1.5);

plot(real(eig_hat),imag(eig_hat), ...
    'x','MarkerSize',10,'LineWidth',1.5);

axis equal;
grid on;

xlabel('Real');
ylabel('Imaginary');

legend( ...
    'Unit circle', ...
    'True eigenvalues', ...
    'N4SID eigenvalues', ...
    'Location','best');

title('Eigenvalues of A');

%% ============================================================
% Local function
% =============================================================

function u = generate_rich_input(N,u_max,seed)

    rng(seed);

    u = zeros(N,1);

    % Split trajectory into three excitation regimes
    edges = round(linspace(1,N+1,4));

    %% 1. Random piecewise-constant steps

    idx = edges(1):(edges(2)-1);

    k = 1;

    while k <= length(idx)

        hold_length = randi([3,30]);

        value = ...
            -u_max + 2*u_max*rand();

        k_end = min(k+hold_length-1,length(idx));

        u(idx(k:k_end)) = value;

        k = k_end+1;

    end

    %% 2. Multisine

    idx = edges(2):(edges(3)-1);

    t = (0:length(idx)-1)';

    frequencies = [
        0.005
        0.011
        0.023
        0.047
        0.091
        0.150
    ];

    phases = 2*pi*rand(length(frequencies),1);

    sig = zeros(length(idx),1);

    for j = 1:length(frequencies)

        sig = sig + ...
            sin(2*pi*frequencies(j)*t + phases(j));

    end

    sig = sig/max(abs(sig));

    u(idx) = 0.9*u_max*sig;

    %% 3. White/random excitation

    idx = edges(3):(edges(4)-1);

    u(idx) = u_max*(2*rand(length(idx),1)-1);

end

dataset.train.u = u_train;
dataset.train.y = y_train_noisy;
dataset.train.x = x_train;

dataset.test.u = u_test;
dataset.test.y = y_test_noisy;
dataset.test.x = x_test;

dataset.Ts = Ts;

dataset.A_true = A_true;
dataset.B_true = B_true;
dataset.C_true = C_true;
dataset.D_true = D_true;

save('linear_order3_dataset.mat', 'dataset', '-v7.3');