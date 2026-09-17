%% generate_rlc_dataset.m
%
% Dataset for ResDyNet latent-state experiment (E5)
%
% Continuous-time RLC circuit:
%
%   x_dot = A x + B u
%       y = C x
%
%   x = [i_L; v_C]
%   y = v_C
%
% IMPORTANT:
%   Only (u,y) are used for system identification.
%   The physical state x is stored only for the post-training
%   latent-state analysis.
%
% Generated data:
%
%   train               -> ResDyNet training
%   val                 -> hyperparameter/model selection
%   test                -> final prediction evaluation
%   affine_calibration  -> estimation of P*, c*
%   latent_test{j}      -> out-of-sample validation of P*, c*
%

clear;
clc;
close all;

rng(42);

%% ================================================================
% RLC PARAMETERS
% ================================================================

R = 1.0;      % Resistance [Ohm]
L = 0.5;      % Inductance [H]
C = 0.2;      % Capacitance [F]

% State:
%
% x1 = i_L
% x2 = v_C

Ac = [-R/L, -1/L;
       1/C,     0];

Bc = [1/L;
       0];

Cc = [0, 1];

Dc = 0;

sys_c = ss(Ac, Bc, Cc, Dc);

fprintf('========================================\n');
fprintf('Continuous-time RLC system\n');
fprintf('========================================\n');

disp('A_c =');
disp(Ac);

disp('B_c =');
disp(Bc);

disp('C_c =');
disp(Cc);

fprintf('Continuous-time poles:\n');
disp(eig(Ac));


%% ================================================================
% DISCRETIZATION
% ================================================================

Ts = 0.02;       % [s]

sys_d = c2d(sys_c, Ts, 'zoh');

Ad = sys_d.A;
Bd = sys_d.B;
Cd = sys_d.C;
Dd = sys_d.D;

fprintf('\n========================================\n');
fprintf('Discrete-time RLC system\n');
fprintf('========================================\n');

fprintf('Ts = %.4f s\n', Ts);

disp('A_d =');
disp(Ad);

disp('B_d =');
disp(Bd);

disp('C_d =');
disp(Cd);

fprintf('Discrete-time poles:\n');
disp(eig(Ad));


%% ================================================================
% STANDARD IDENTIFICATION DATASETS
% ================================================================

Ntrain = 50000;
Nval   = 10000;
Ntest  = 20000;

u_min = -2;
u_max =  2;

% Number of samples for which each random level is held
hold_min = 5;
hold_max = 50;


%% Training dataset

u_train = generate_random_steps( ...
    Ntrain, ...
    u_min, u_max, ...
    hold_min, hold_max);

train = simulate_rlc( ...
    Ad, Bd, Cd, Dd, ...
    u_train, Ts, ...
    [0; 0]);


%% Validation dataset

u_val = generate_random_steps( ...
    Nval, ...
    u_min, u_max, ...
    hold_min, hold_max);

val = simulate_rlc( ...
    Ad, Bd, Cd, Dd, ...
    u_val, Ts, ...
    [0.3; -0.4]);


%% Test dataset

u_test = generate_random_steps( ...
    Ntest, ...
    u_min, u_max, ...
    hold_min, hold_max);

test = simulate_rlc( ...
    Ad, Bd, Cd, Dd, ...
    u_test, Ts, ...
    [-0.4; 0.3]);


%% ================================================================
% AFFINE-CALIBRATION TRAJECTORY
%
% This trajectory is NOT used for ResDyNet training.
%
% After ResDyNet has been trained, it will be used to estimate
%
%       x_hat ~= P*x + c
%
% P and c must then be frozen.
% ================================================================

Naffine = 10000;

u_affine = generate_random_steps( ...
    Naffine, ...
    -1.8, 1.8, ...
    8, 40);

affine_calibration = simulate_rlc( ...
    Ad, Bd, Cd, Dd, ...
    u_affine, Ts, ...
    [0.6; -0.3]);


%% ================================================================
% LATENT-STATE TEST TRAJECTORIES
%
% These trajectories must NEVER be used:
%
%   - for ResDyNet training
%   - for validation/model selection
%   - to estimate P and c
%
% They are exclusively used to evaluate whether the SAME affine
% transformation generalizes to unseen trajectories.
% ================================================================

latent_test = {};

Nlatent = 5000;

%% ------------------------------------------------
% Test 1: random steps, same approximate range
% -------------------------------------------------

u = generate_random_steps( ...
    Nlatent, ...
    -2, 2, ...
    5, 50);

latent_test{end+1} = simulate_rlc( ...
    Ad, Bd, Cd, Dd, ...
    u, Ts, ...
    [-0.7; 0.6]);

latent_test{end}.name = ...
    'random_steps_nominal';


%% ------------------------------------------------
% Test 2: faster random switching
% -------------------------------------------------

u = generate_random_steps( ...
    Nlatent, ...
    -2, 2, ...
    2, 10);

latent_test{end+1} = simulate_rlc( ...
    Ad, Bd, Cd, Dd, ...
    u, Ts, ...
    [0.8; -0.5]);

latent_test{end}.name = ...
    'random_steps_fast';


%% ------------------------------------------------
% Test 3: slower random switching
% -------------------------------------------------

u = generate_random_steps( ...
    Nlatent, ...
    -2, 2, ...
    40, 100);

latent_test{end+1} = simulate_rlc( ...
    Ad, Bd, Cd, Dd, ...
    u, Ts, ...
    [-0.5; -0.5]);

latent_test{end}.name = ...
    'random_steps_slow';


%% ------------------------------------------------
% Test 4: multisine input
% -------------------------------------------------

t = (0:Nlatent-1)' * Ts;

u = ...
      0.70*sin(2*pi*0.10*t) ...
    + 0.50*sin(2*pi*0.35*t + 0.4) ...
    + 0.35*sin(2*pi*0.80*t + 1.1) ...
    + 0.20*sin(2*pi*1.50*t + 0.7);

latent_test{end+1} = simulate_rlc( ...
    Ad, Bd, Cd, Dd, ...
    u, Ts, ...
    [0.4; 0.7]);

latent_test{end}.name = ...
    'multisine';


%% ------------------------------------------------
% Test 5: chirp input
% -------------------------------------------------

t = (0:Nlatent-1)' * Ts;

% Frequency increases from 0.05 Hz to 2 Hz
u = 1.5 * chirp( ...
    t, ...
    0.05, ...
    t(end), ...
    2.0);

latent_test{end+1} = simulate_rlc( ...
    Ad, Bd, Cd, Dd, ...
    u, Ts, ...
    [-0.8; 0.2]);

latent_test{end}.name = ...
    'chirp';


%% ------------------------------------------------
% Test 6: constant positive input
% -------------------------------------------------

u = 1.2 * ones(Nlatent,1);

latent_test{end+1} = simulate_rlc( ...
    Ad, Bd, Cd, Dd, ...
    u, Ts, ...
    [-0.5; 0.8]);

latent_test{end}.name = ...
    'constant_positive';


%% ------------------------------------------------
% Test 7: constant negative input
% -------------------------------------------------

u = -1.2 * ones(Nlatent,1);

latent_test{end+1} = simulate_rlc( ...
    Ad, Bd, Cd, Dd, ...
    u, Ts, ...
    [0.7; -0.6]);

latent_test{end}.name = ...
    'constant_negative';


%% ------------------------------------------------
% Test 8: zero-input free response
% -------------------------------------------------

u = zeros(Nlatent,1);

latent_test{end+1} = simulate_rlc( ...
    Ad, Bd, Cd, Dd, ...
    u, Ts, ...
    [1.0; -1.0]);

latent_test{end}.name = ...
    'free_response';


%% ------------------------------------------------
% Test 9: amplitude extrapolation
%
% Deliberately outside training input range.
% -------------------------------------------------

u = generate_random_steps( ...
    Nlatent, ...
    -3, 3, ...
    5, 50);

latent_test{end+1} = simulate_rlc( ...
    Ad, Bd, Cd, Dd, ...
    u, Ts, ...
    [0.2; -0.8]);

latent_test{end}.name = ...
    'random_steps_large_amplitude';


%% ------------------------------------------------
% Test 10: combined excitation
% -------------------------------------------------

t = (0:Nlatent-1)' * Ts;

u_steps = generate_random_steps( ...
    Nlatent, ...
    -1, 1, ...
    20, 60);

u = ...
    u_steps ...
    + 0.5*sin(2*pi*0.4*t) ...
    + 0.25*sin(2*pi*1.2*t);

latent_test{end+1} = simulate_rlc( ...
    Ad, Bd, Cd, Dd, ...
    u, Ts, ...
    [-0.6; 0.4]);

latent_test{end}.name = ...
    'combined';


%% ================================================================
% OPTIONAL MEASUREMENT NOISE
% ================================================================

add_measurement_noise = false;

SNR_dB = 30;

if add_measurement_noise

    train.y_clean = train.y;
    val.y_clean   = val.y;
    test.y_clean  = test.y;

    train.y = add_measurement_noise_snr( ...
        train.y, SNR_dB);

    val.y = add_measurement_noise_snr( ...
        val.y, SNR_dB);

    test.y = add_measurement_noise_snr( ...
        test.y, SNR_dB);

end


%% ================================================================
% PARAMETERS
% ================================================================

parameters.R  = R;
parameters.L  = L;
parameters.C  = C;
parameters.Ts = Ts;

parameters.Ac = Ac;
parameters.Bc = Bc;
parameters.Cc = Cc;
parameters.Dc = Dc;

parameters.Ad = Ad;
parameters.Bd = Bd;
parameters.Cd = Cd;
parameters.Dd = Dd;

parameters.Ntrain = Ntrain;
parameters.Nval   = Nval;
parameters.Ntest  = Ntest;

parameters.state_names = { ...
    'i_L', ...
    'v_C'};

parameters.output_name = 'v_C';


%% ================================================================
% SAVE
% ================================================================

save( ...
    'rlc_resdynet_E5_dataset.mat', ...
    'train', ...
    'val', ...
    'test', ...
    'affine_calibration', ...
    'latent_test', ...
    'parameters', ...
    '-v7.3');

fprintf('\n========================================\n');
fprintf('Dataset generated successfully\n');
fprintf('========================================\n');

fprintf('Training samples:       %d\n', Ntrain);
fprintf('Validation samples:     %d\n', Nval);
fprintf('Test samples:           %d\n', Ntest);
fprintf('Affine calibration:     %d\n', Naffine);
fprintf('Latent test trajectories: %d\n', ...
    length(latent_test));

fprintf('\nSaved as:\n');
fprintf('rlc_resdynet_E5_dataset.mat\n');


%% ================================================================
% PLOTS
% ================================================================

Nplot = min(1500, Ntest);

figure;

subplot(4,1,1);

plot( ...
    test.t(1:Nplot), ...
    test.u(1:Nplot), ...
    'LineWidth', 1);

grid on;

ylabel('$u$', ...
    'Interpreter','latex');

title('RLC test dataset');


subplot(4,1,2);

plot( ...
    test.t(1:Nplot), ...
    test.x(1:Nplot,1), ...
    'LineWidth',1);

grid on;

ylabel('$i_L$', ...
    'Interpreter','latex');


subplot(4,1,3);

plot( ...
    test.t(1:Nplot), ...
    test.x(1:Nplot,2), ...
    'LineWidth',1);

grid on;

ylabel('$v_C$', ...
    'Interpreter','latex');


subplot(4,1,4);

plot( ...
    test.t(1:Nplot), ...
    test.y(1:Nplot), ...
    'LineWidth',1);

grid on;

xlabel('$t$ [s]', ...
    'Interpreter','latex');

ylabel('$y$', ...
    'Interpreter','latex');


%% ================================================================
% Plot all latent-test inputs
% ================================================================

figure;

tiledlayout(5,2);

for j = 1:length(latent_test)

    nexttile;

    plot( ...
        latent_test{j}.t, ...
        latent_test{j}.u, ...
        'LineWidth', 0.8);

    grid on;

    title( ...
        strrep(latent_test{j}.name, '_', '\_'));

    xlabel('t [s]');
    ylabel('u');

end


%% ================================================================
% LOCAL FUNCTIONS
% ================================================================

function u = generate_random_steps( ...
    N, ...
    u_min, u_max, ...
    hold_min, hold_max)

    u = zeros(N,1);

    k = 1;

    while k <= N

        hold_length = ...
            randi([hold_min, hold_max]);

        uk = ...
            u_min ...
            + (u_max-u_min)*rand;

        k_end = ...
            min(k + hold_length - 1, N);

        u(k:k_end) = uk;

        k = k_end + 1;

    end

end


function data = simulate_rlc( ...
    A, B, C, D, ...
    u, Ts, x0)

    N = length(u);

    nx = size(A,1);
    ny = size(C,1);

    x = zeros(N,nx);
    y = zeros(N,ny);

    x(1,:) = x0(:).';

    for k = 1:N-1

        y(k,:) = ...
            (C*x(k,:).' + D*u(k)).';

        x(k+1,:) = ...
            (A*x(k,:).' + B*u(k)).';

    end

    y(N,:) = ...
        (C*x(N,:).' + D*u(N)).';

    data.t = ...
        (0:N-1)' * Ts;

    data.u = u;
    data.y = y;
    data.x = x;

end


function y_noisy = ...
    add_measurement_noise_snr(y, SNR_dB)

    signal_power = ...
        mean(y.^2,1);

    noise_power = ...
        signal_power ...
        ./ (10.^(SNR_dB/10));

    noise = ...
        randn(size(y)) ...
        .* sqrt(noise_power);

    y_noisy = ...
        y + noise;

end