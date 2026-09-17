# E1 - Cascaded Tanks Linear Initialization Ablation

Preliminary paired-seed ablation comparing:

- `ResDyNet-Random`
- `ResDyNet-LinearInit`

Dataset: `nonlinear_benchmarks.Cascaded_Tanks`

## Command

```bash
python3 run_cascaded_tanks_ablation.py \
  --seeds 0 1 2 3 4 5 6 \
  --epochs 60 \
  --dtype float64 \
  --learning-rate 1e-3 \
  --clip-grad-norm 1.0 \
  --output-dir outputs/cascaded_tanks_ablation_7seeds_60epochs
```

## Setup

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

The N4SID realization is diagonally similarity-scaled before initialization.
This preserves the linear input-output predictor while improving numerical
scaling of `S_e`, `S_f`, and `S_g`.

## Median Results Across 7 Seeds

| metric | ResDyNet-Random | ResDyNet-LinearInit |
| --- | ---: | ---: |
| best validation NRMSE | 0.33262 | 0.21256 |
| final validation NRMSE | 0.37443 | 0.25310 |
| final test NRMSE | 0.18804 | 0.17348 |

## Time-To-Threshold

| validation NRMSE threshold | ResDyNet-Random | ResDyNet-LinearInit |
| --- | --- | --- |
| 0.40 | 7/7 hits, median epoch 10, median time 2.80s | 7/7 hits, median epoch 5, median time 1.33s |
| 0.30 | 3/7 hits, median epoch 35, median time 9.62s | 7/7 hits, median epoch 9, median time 2.40s |
| 0.25 | 1/7 hits, median epoch 35, median time 9.62s | 7/7 hits, median epoch 16, median time 4.29s |
| 0.20 | 0/7 hits | 2/7 hits, median epoch 32, median time 8.81s |

Thresholds 0.05, 0.03, and 0.02 were not reached by either method in this
preliminary 60-epoch run.

## Files

- `trajectories.csv`: validation NRMSE per epoch and wall-clock time.
- `summary.csv`: best/final validation NRMSE and final test NRMSE per seed.
- `time_to_threshold.csv`: threshold hit data for each seed and method.
- `val_nrmse_vs_epoch.png`: aggregated validation NRMSE trajectory by epoch.
- `val_nrmse_vs_time.png`: aggregated validation NRMSE trajectory by wall-clock time.
