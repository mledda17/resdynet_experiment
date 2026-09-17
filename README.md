# resdynet_experiment

PyTorch implementation of ResDyNet from the attached Sections II-IV excerpt.

The implementation in `main.py` provides:

- projected-residual encoder and decoder;
- ResDyNet evolver
  `S_f [x, u] + W_o(Phi^B(W_i [x, u]) - W_i [x, u])` with `B` independent
  residual stream blocks;
- exact linear-informed initialization from supplied `(A_L, B_L, C_L, D_L)`,
  including the finite-memory `R_L` construction and inactive extra latent
  coordinates when `n_L < n_x`;
- input-output BLA initialization via N4SID state-space identification;
- zero initialization of every nonlinear branch output layer;
- weighted multi-step rollout loss with one encoder call per rollout;
- random-initialization and plain-MLP evolver baselines.
- adapters for the official `nonlinear_benchmarks` train/test splits.

Install the dependency with:

```bash
pip install -r requirements.txt
```

Minimal usage:

```python
import torch
from main import (
    evaluate_model,
    fit_n4sid_state_space,
    make_resdynet,
    make_official_benchmark_rollout_datasets,
    official_benchmark_split,
    train_on_dataset,
)

train_val_splits, _ = official_benchmark_split("WienerHammerBenchMark")
split = train_val_splits[0]
datasets, test_init_lengths = make_official_benchmark_rollout_datasets(
    "WienerHammerBenchMark",
    na=n,
    nb=n,
    horizon=H,
)

u_mean, u_std = split.u_train.mean(0, keepdim=True), split.u_train.std(0, keepdim=True)
y_mean, y_std = split.y_train.mean(0, keepdim=True), split.y_train.std(0, keepdim=True)
u_train = (split.u_train - u_mean) / u_std
y_train = (split.y_train - y_mean) / y_std

linear = fit_n4sid_state_space(
    u_train,
    y_train,
    order=n_L,
    zero_direct_feedthrough=True,
)
model = make_resdynet(
    ny=ny,
    nu=nu,
    nx=nx,
    na=n,
    nb=n,
    encoder_hidden_dims=(64, 64),
    decoder_hidden_dims=(64, 64),
    stream_dim=2 * (nx + nu) + 1,
    num_blocks=4,
    evolver_hidden_dims=(64,),
    linear=linear,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
history = train_on_dataset(
    model,
    datasets.train,
    epochs=200,
    batch_size=128,
    optimizer=optimizer,
)
val_loss = evaluate_model(model, datasets.val, batch_size=256)
test_loss = evaluate_model(model, datasets.test, batch_size=256)
```

Cascaded Tanks initialization ablation:

```bash
python3 run_cascaded_tanks_ablation.py \
  --seeds 0 1 2 3 4 5 6 \
  --epochs 60 \
  --dtype float64 \
  --learning-rate 1e-3 \
  --clip-grad-norm 1.0 \
  --output-dir outputs/cascaded_tanks_ablation_7seeds_60epochs
```

Cascaded Tanks evolver ablation:

```bash
python3 run_cascaded_tanks_e2_evolver_ablation.py \
  --seeds 0 1 2 3 4 5 6 \
  --epochs 60 \
  --output-dir outputs/E2
```

Cascaded Tanks long evolver ablation:

```bash
python3 run_cascaded_tanks_e2_evolver_ablation.py \
  --seeds 0 1 2 \
  --epochs 300 \
  --output-dir outputs/E2_long_3seeds_300epochs
```

RLC latent-state affine experiment E5:

```bash
python3 run_rlc_latent_experiment.py \
  --mode train-analyze \
  --seeds 0 1 2 3 4 \
  --output-dir E5
```

Regenerate analysis and figures from a trained checkpoint:

```bash
python3 run_rlc_latent_experiment.py \
  --mode analyze \
  --checkpoint E5/checkpoints/seed_0_best.pt \
  --output-dir E5_reanalysis
```
