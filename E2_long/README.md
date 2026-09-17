# E2 Long - Residual Evolver vs MLP Correction

Longer version of E2 using 3 paired seeds and 300 epochs.

Compared models:

```text
ResDyNet-ResidualEvolver:
f(h) = S_f h + W_o(Phi^B(W_i h) - W_i h)

ResDyNet-MLPEvolver:
f(h) = S_f h + psi_MLP(h)
```

The linear branch `S_f h` is kept in both models. Both models use the same
linear-informed initialization, data split, optimizer, learning rate,
mini-batches, and number of updates.

## Command

```bash
python3 run_cascaded_tanks_e2_evolver_ablation.py \
  --seeds 0 1 2 \
  --epochs 300 \
  --output-dir outputs/E2_long_3seeds_300epochs
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

## Nonlinear Branch Parameter Counts

| model | nonlinear parameters |
| --- | ---: |
| ResDyNet-ResidualEvolver | 10383 |
| ResDyNet-MLPEvolver | 10424 |

Relative gap: `0.39%`.

## Mean Results Across 3 Seeds

| metric | ResDyNet-ResidualEvolver | ResDyNet-MLPEvolver |
| --- | ---: | ---: |
| best validation NRMSE | 0.22417 ± 0.03731 | 0.24651 ± 0.02189 |
| final validation NRMSE | 0.29707 ± 0.02317 | 0.28512 ± 0.01385 |
| final test NRMSE | 0.19864 ± 0.03968 | 0.18738 ± 0.01456 |

The residual evolver reaches a better best-validation point on all three seeds.
The final-epoch metric is not uniformly better, indicating that best-checkpoint
selection or early stopping matters in this longer run.

## Time-To-Threshold

| validation NRMSE threshold | ResDyNet-ResidualEvolver | ResDyNet-MLPEvolver |
| --- | --- | --- |
| 0.40 | 3/3 hits, median epoch 4, median time 1.15s | 3/3 hits, median epoch 4, median time 0.68s |
| 0.30 | 3/3 hits, median epoch 8, median time 2.21s | 3/3 hits, median epoch 28, median time 4.77s |
| 0.25 | 2/3 hits, median epoch 96, median time 26.38s | 1/3 hits, median epoch 53, median time 9.06s |
| 0.20 | 1/3 hits, median epoch 56, median time 15.44s | 0/3 hits |

## Files

- `parameter_counts.csv`: nonlinear branch parameter counts.
- `trajectories.csv`: validation NRMSE per epoch and wall-clock time.
- `summary.csv`: best/final validation NRMSE and final test NRMSE per seed.
- `time_to_threshold.csv`: threshold hit data for each seed and method.
- `val_nrmse_vs_epoch_mean_std.png`: validation NRMSE mean ± std by epoch.
- `val_nrmse_vs_time_mean_std.png`: validation NRMSE mean ± std by wall-clock time.
