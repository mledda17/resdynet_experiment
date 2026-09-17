%% generate_duffing_dataset.m
%
% Dataset generator for the Duffing oscillator used in the ResDyNet paper.
%
% Model:
%
%   x1(k+1) = x1(k) + Ts*x2(k)
%
%   x2(k+1) = x2(k) + Ts * (
%                 -delta*x2(k)
%                 -alpha*x1(k)
%                 -beta*x1(k)^3
%                 +gamma*u(k))
%
%   y(k) = x1(k) + 0.05*x1(k)^2 + v(k)
%
% The excitation is deliberately richer than a single PRBS:
%   1) random piecewise-constant steps
%   2) multisine
%   3) chirp
%   4) smooth random excitation
%
% All inputs are bounded in [-u_max, u_max].
%
% The true state x is stored only for diagnostics / latent-state analysis.
% ResDyNet identification should use ONLY u and y.
%
% -------------------------------------------------------------

clear;
clc;

%% ============================================================
% Configuration
% =============================================================

cfg.Ts    = 0.01;

cfg.alpha = 1.0;
cfg.beta  = 1.0;
cfg.gamma = 1.0;
cfg.delta = 0.2;

cfg.snr_db = 40;

cfg.u_max = 2.0;

cfg.N_train = 90000;
cfg.N_val   = 10000;
cfg.N_test  = 20000;

cfg.seed_train = 10;
cfg.seed_val   = 20;
cfg.seed_test  = 30;

% Independent initial conditions.
% These are intentionally small so that the excitation, rather than
% a large initial transient, determines the explored operating region.
cfg.x0_train = [0.0;  0.0];
cfg.x0_val   = [0.3; -0.2];
cfg.x0_test  = [-0.3; 0.25];

%% ============================================================
% Generate excitation signals
% =============================================================

fprintf('Generating excitation signals...\n');

u_train = generate_rich_excitation( ...
    cfg.N_train, cfg.Ts, cfg.u_max, cfg.seed_train);

u_val = generate_rich_excitation( ...
    cfg.N_val, cfg.Ts, cfg.u_max, cfg.seed_val);

u_test = generate_rich_excitation( ...
    cfg.N_test, cfg.Ts, cfg.u_max, cfg.seed_test);

%% ============================================================
% Simulate CLEAN trajectories
% =============================================================

fprintf('Simulating clean Duffing trajectories...\n');

train_clean = simulate_duffing( ...
    u_train, cfg.x0_train, cfg);

val_clean = simulate_duffing( ...
    u_val, cfg.x0_val, cfg);

test_clean = simulate_duffing( ...
    u_test, cfg.x0_test, cfg);

%% ============================================================
% Measurement noise
%
% Compute noise variance ONLY from the training output.
% The same noise standard deviation is then used for train/val/test.
% =============================================================

signal_std = std(train_clean.y_clean);

noise_std = signal_std / 10^(cfg.snr_db / 20);

fprintf('Training output std : %.6f\n', signal_std);
fprintf('Noise std           : %.6f\n', noise_std);
fprintf('Requested SNR       : %.2f dB\n', cfg.snr_db);

%% ============================================================
% Add independent measurement noise
% =============================================================

rng(1001);
train_noise = noise_std * randn(cfg.N_train, 1);

rng(1002);
val_noise = noise_std * randn(cfg.N_val, 1);

rng(1003);
test_noise = noise_std * randn(cfg.N_test, 1);

train = train_clean;
val   = val_clean;
test  = test_clean;

train.y = train.y_clean + train_noise;
val.y   = val.y_clean   + val_noise;
test.y  = test.y_clean  + test_noise;

train.noise = train_noise;
val.noise   = val_noise;
test.noise  = test_noise;

%% ============================================================
% Useful training-set statistics
% =============================================================

normalization.u_mean = mean(train.u);
normalization.u_std  = std(train.u);

normalization.y_mean = mean(train.y);
normalization.y_std  = std(train.y);

%% ============================================================
% Diagnostics
% =============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('Duffing dataset summary\n');
fprintf('============================================\n');

print_summary('TRAIN', train);
print_summary('VAL',   val);
print_summary('TEST',  test);

fprintf('\nNonlinearity indicator on training data:\n');

x1 = train.x(:,1);

ratio = x1.^2;

fprintf('median(x1^2) = %.4f\n', median(ratio));
fprintf('90%%   (x1^2) = %.4f\n', prctile(ratio,90));
fprintf('95%%   (x1^2) = %.4f\n', prctile(ratio,95));
fprintf('max   (x1^2) = %.4f\n', max(ratio));

% x1^2 is exactly the magnitude ratio
%
%       |beta*x1^3| / |alpha*x1|
%
% for alpha = beta = 1.

%% ============================================================
% Save
% =============================================================

filename = 'duffing_resdynet_dataset.mat';

save( ...
    filename, ...
    'train', ...
    'val', ...
    'test', ...
    'cfg', ...
    'normalization', ...
    '-v7.3');

fprintf('\nDataset saved to:\n');
fprintf('    %s\n', filename);

%% ============================================================
% Quick visualization
% =============================================================

Nplot = min(5000, cfg.N_train);

t = (0:Nplot-1)' * cfg.Ts;

figure;

subplot(3,1,1);
plot(t, train.u(1:Nplot), 'LineWidth', 0.9);
ylabel('u_k');
grid on;
title('Duffing training dataset');

subplot(3,1,2);
plot(t, train.y(1:Nplot), 'LineWidth', 0.9);
ylabel('y_k');
grid on;

subplot(3,1,3);
plot(t, train.x(1:Nplot,1), 'LineWidth', 0.9);
hold on;
plot(t, train.x(1:Nplot,2), 'LineWidth', 0.9);
xlabel('Time [s]');
ylabel('State');
legend('x_1','x_2');
grid on;


%% ============================================================
% Local functions
% =============================================================

function data = simulate_duffing(u, x0, cfg)

    N = length(u);

    x = zeros(N,2);

    x(1,:) = x0(:)';

    for k = 1:N-1

        x1 = x(k,1);
        x2 = x(k,2);

        x1_next = ...
            x1 ...
            + cfg.Ts*x2;

        x2_next = ...
            x2 ...
            + cfg.Ts*( ...
                -cfg.delta*x2 ...
                -cfg.alpha*x1 ...
                -cfg.beta*x1^3 ...
                +cfg.gamma*u(k));

        x(k+1,1) = x1_next;
        x(k+1,2) = x2_next;

    end

    y_clean = ...
        x(:,1) ...
        + 0.05*x(:,1).^2;

    data.u = u(:);
    data.x = x;
    data.y_clean = y_clean(:);

end


function u = generate_rich_excitation(N, Ts, u_max, seed)

    rng(seed);

    % Divide the trajectory into four excitation regimes.
    edges = round(linspace(1,N+1,5));

    u = zeros(N,1);

    %% --------------------------------------------------------
    % 1. Random piecewise-constant excitation
    % ---------------------------------------------------------

    idx = edges(1):(edges(2)-1);

    u(idx) = random_steps( ...
        length(idx), ...
        4, ...
        40, ...
        0.85*u_max);

    %% --------------------------------------------------------
    % 2. Random-phase multisine
    % ---------------------------------------------------------

    idx = edges(2):(edges(3)-1);

    t = (0:length(idx)-1)' * Ts;

    % Frequencies chosen to excite different time scales.
    freqs = [ ...
        0.05, ...
        0.12, ...
        0.25, ...
        0.50, ...
        0.90, ...
        1.50, ...
        2.50 ];

    phases = 2*pi*rand(size(freqs));

    sig = zeros(size(t));

    for j = 1:length(freqs)

        sig = sig + ...
            sin(2*pi*freqs(j)*t + phases(j));

    end

    sig = sig / max(abs(sig));

    u(idx) = 0.90*u_max*sig;

    %% --------------------------------------------------------
    % 3. Linear chirp
    %
    % Implemented analytically so no Signal Processing Toolbox
    % is required.
    % ---------------------------------------------------------

    idx = edges(3):(edges(4)-1);

    t = (0:length(idx)-1)' * Ts;

    T = max(t(end), Ts);

    f0 = 0.03;
    f1 = 3.0;

    chirp_rate = (f1-f0)/T;

    phase = ...
        2*pi*( ...
            f0*t ...
            + 0.5*chirp_rate*t.^2);

    u(idx) = ...
        0.85*u_max*sin(phase);

    %% --------------------------------------------------------
    % 4. Smooth stochastic excitation
    %
    % AR(1) filtering gives a broadband random signal with more
    % temporal correlation than white noise.
    % ---------------------------------------------------------

    idx = edges(4):(edges(5)-1);

    M = length(idx);

    white = randn(M,1);

    smooth = zeros(M,1);

    rho = 0.95;

    for k = 2:M

        smooth(k) = ...
            rho*smooth(k-1) ...
            + sqrt(1-rho^2)*white(k);

    end

    smooth = smooth - mean(smooth);

    max_abs = max(abs(smooth));

    if max_abs > 0
        smooth = smooth/max_abs;
    end

    u(idx) = 0.90*u_max*smooth;

    %% --------------------------------------------------------
    % Safety clipping
    % ---------------------------------------------------------

    u = max(min(u,u_max),-u_max);

end


function u = random_steps(N, min_hold, max_hold, amplitude)

    u = zeros(N,1);

    k = 1;

    while k <= N

        hold_length = ...
            randi([min_hold,max_hold]);

        value = ...
            -amplitude ...
            + 2*amplitude*rand();

        k_end = ...
            min(k + hold_length - 1,N);

        u(k:k_end) = value;

        k = k_end + 1;

    end

end


function print_summary(name,data)

    fprintf('\n%s\n', name);
    fprintf('--------------------------------------------\n');

    fprintf('N             : %d\n', length(data.u));

    fprintf('u range       : [%+.4f, %+.4f]\n', ...
        min(data.u), max(data.u));

    fprintf('y range       : [%+.4f, %+.4f]\n', ...
        min(data.y), max(data.y));

    fprintf('x1 range      : [%+.4f, %+.4f]\n', ...
        min(data.x(:,1)), max(data.x(:,1)));

    fprintf('x2 range      : [%+.4f, %+.4f]\n', ...
        min(data.x(:,2)), max(data.x(:,2)));

end