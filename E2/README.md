# E2 - Residual Evolver Correction vs MLP Correction

Preliminary paired-seed ablation comparing the nonlinear correction inside the
evolver while keeping the linear branch `S_f h` in both models.

Compared models:

```text
ResDyNet-ResidualEvolver:
f(h) = S_f h + W_o(Phi^B(W_i h) - W_i h)

ResDyNet-MLPEvolver:
f(h) = S_f h + psi_MLP(h)
```

Both models use the same encoder, decoder, linear-informed initialization, data
split, optimizer, learning rate, mini-batches, and number of updates.

## Command

```bash
python3 run_cascaded_tanks_e2_evolver_ablation.py \
  --seeds 0 1 2 3 4 5 6 \
  --epochs 60 \
  --output-dir outputs/E2
```

## Setup

- dataset: `nonlinear_benchmarks.Cascaded_Tanks`
- train/validation/test samples: `819 / 205 / 1024`
- rollout windows train/validation/test: `720 / 106 / 925`
- `n_a = n_b = 50`
- prediction horizon `H = 50`
- N4SID linear order `n_L = 6`
- ResDyNet latent dimension `n_x = 8`
- residual evolver blocks `B = 4`
- residual stream dimension `n_h = 19`
- optimizer: Adam
- learning rate: `1e-3`
- gradient clipping: `1.0`
- dtype: `float64`

The N4SID realization is diagonally similarity-scaled before initialization,
preserving the linear predictor while improving parameter scaling.

## Nonlinear Branch Parameter Counts

| model | nonlinear parameters |
| --- | ---: |
| ResDyNet-ResidualEvolver | 10383 |
| ResDyNet-MLPEvolver | 10424 |

Relative gap: `0.39%`.

The MLP correction uses hidden dimensions `(93, 93)`.

## Mean Results Across 7 Seeds

| metric | ResDyNet-ResidualEvolver | ResDyNet-MLPEvolver |
| --- | ---: | ---: |
| best validation NRMSE | 0.24175 ± 0.03328 | 0.25817 ± 0.03086 |
| final validation NRMSE | 0.29123 ± 0.03448 | 0.30665 ± 0.02517 |
| final test NRMSE | 0.19380 ± 0.03120 | 0.20628 ± 0.01006 |

## Time-To-Threshold

| validation NRMSE threshold | ResDyNet-ResidualEvolver | ResDyNet-MLPEvolver |
| --- | --- | --- |
| 0.40 | 7/7 hits, median epoch 4, median time 1.14s | 7/7 hits, median epoch 4, median time 0.70s |
| 0.30 | 7/7 hits, median epoch 11, median time 3.11s | 6/7 hits, median epoch 22.5, median time 3.92s |
| 0.25 | 3/7 hits, median epoch 29, median time 8.12s | 3/7 hits, median epoch 53, median time 8.93s |
| 0.20 | 1/7 hits, median epoch 56, median time 15.45s | 0/7 hits |

## Files

- `parameter_counts.csv`: nonlinear branch parameter counts.
- `trajectories.csv`: validation NRMSE per epoch and wall-clock time.
- `summary.csv`: best/final validation NRMSE and final test NRMSE per seed.
- `time_to_threshold.csv`: threshold hit data for each seed and method.
- `val_nrmse_vs_epoch_mean_std.png`: validation NRMSE mean ± std by epoch.
- `val_nrmse_vs_time_mean_std.png`: validation NRMSE mean ± std by wall-clock time.
