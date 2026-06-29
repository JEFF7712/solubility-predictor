# Soluble: A GNN Solubility Predictor
This project uses a Graph Neural Network (GNN) to treat molecules as graph structures and predict aqueous solubility (logS). Trained on AqSolDB, filtered to the ~8,662 molecules built from the supported elements.

<img width="2880" height="1800" alt="ss-2026-01-08-02-40-00" src="https://github.com/user-attachments/assets/6995f260-1354-40f4-9c67-26fab54a3a9d" />
Try it out yourself: https://soluble.rupan.dev

## Model Architecture
Uses a **GINEConv** backbone with edge attribute support:
- **Input Layer**: 16-dimensional Node Features + 4-dimensional Edge Features.
- **Hidden Layers**: 3× GINEConv layers (hidden_dim 256) with LayerNorm after each. Width, depth and dropout were tuned with a 50-trial Optuna search on the GPU.
- **Residual Connections**: Skip connection around every GINEConv block to improve gradient flow.
- **Global Pooling**: Concatenates sum, mean, and max pooling to capture size-dependent and functional group features.
- **Prediction Head**: MLP (3 × hidden_dim → hidden_dim → 1) for final solubility prediction.

### Standardization, Uncertainty & Reliability
- **Standardization**: Continuous node features and the logS target are standardized using statistics fit on the training split only, then persisted with the model so the exact transform is reproduced at inference.
- **Deep Ensemble**: Production predictions come from 5 independently-seeded models. The reported logS is the ensemble mean, and the spread across members (**± std**) is a calibrated uncertainty: out-of-domain molecules show a visibly wider spread.
- **Applicability Domain**: Inputs containing elements outside the supported set are flagged (`in_domain: false` + a warning) rather than silently predicted, since the featurizer encodes unknown atoms as an all-zero element one-hot.
- **Graded Solubility**: A four-band qualitative scale (Highly soluble → Poorly soluble) replaces the original binary cutoff.

---

## Model Evolution & Performance

### Iteration 1: Baseline
- Single GINE layer, minimal features.
- **Test MSE**: 1.9883
- **Issue**: Shallow architecture captures only 1-hop neighborhoods.

### Iteration 2: Depth & Regularization
- 3 GINE layers, Dropout, Weight Decay, LR Scheduler.
- **Test MSE**: 0.8209 (~59% improvement)
- **Key changes**: Deeper model, batch normalization, dynamic LR scheduling.

### Iteration 3: Hybrid Pooling & Rich Features
- Multi-pooling (sum/mean/max), residual connections, expanded atom features.
- **Test MSE**: 0.5071 (~38% improvement)
- **Key changes**: Better feature engineering, skip connections reduce vanishing gradients.

### Iteration 4: Normalization & Optimizer Optimization
- Batch normalization → LayerNorm for better stability.
- Adam Optimizer → AdamW with weight decay.
- Refined training with validation set (70/15/15 split).
- **Test MSE**: 0.3197 (~37% improvement)
- **Key changes**: Better normalization technique, improved optimizer, and proper train/val/test split for unbiased evaluation.

### Iteration 5: Standardization & Deep Ensemble
- Feature/target standardization fit on the training split; early stopping on a validation slice.
- Single model → 5-model deep ensemble reporting mean ± std (calibrated uncertainty).
- **Key changes**: Improved random-split accuracy, roughly halved the cross-fold variance on the scaffold split, and added uncertainty quantification plus applicability-domain flagging.
- Metrics (ESOL): random-split CV RMSE 0.5623, scaffold-split CV RMSE 0.8116.

### Iteration 6: AqSolDB & Tuned Architecture
- Dataset: ESOL (~1.1k) → AqSolDB (~8.7k in-domain molecules, ~7.7x more chemical coverage).
- Architecture re-tuned for the larger dataset via a 50-trial GPU Optuna search (hidden_dim 160 → 256, small dropout, larger batch).
- **Key changes**: A much more general model. The headline RMSE is higher than the ESOL iterations not because the model regressed but because AqSolDB is a substantially harder, noisier benchmark (logS -13 to +2, aggregated from 9 sources). The numbers are competitive with published AqSolDB GNN results (~0.9-1.1 random-split RMSE).

## Model Performance Metrics

Metrics below are for the production 5-model deep ensemble (Iteration 6) on
AqSolDB. **They are not comparable to iterations 1-5**, which used the smaller,
easier ESOL dataset. Each fold reserves 15% of its training data as a validation
slice for early stopping.

**5-Fold Cross-Validation** (Random Split):
- **Average RMSE**: 0.9059 ± 0.0052
- **Average R²**: 0.8472 (explains 84.72% of variance)
- **Folds**: RMSE = 0.9101 / 0.9114 / 0.8970 / 0.9074 / 0.9038

**5-Fold Scaffold Split Cross-Validation** (More Rigorous - Prevents Data Leakage):
- **Average RMSE**: 0.9353 ± 0.0981
- **Average R²**: 0.8027 (explains 80.27% of variance)
- **Fold 1**: Train 6482 | Test 2180 → RMSE = 0.9357, R² = 0.8291
- **Fold 2**: Train 7007 | Test 1655 → RMSE = 0.7582, R² = 0.7374
- **Fold 3**: Train 7053 | Test 1609 → RMSE = 1.0577, R² = 0.8103
- **Fold 4**: Train 7053 | Test 1609 → RMSE = 0.9756, R² = 0.7923
- **Fold 5**: Train 7053 | Test 1609 → RMSE = 0.9491, R² = 0.8444

Spot checks on well-known molecules (logS): naphthalene -4.05 (exp -3.61),
benzene -1.77 (exp -1.64), caffeine -0.95 (exp -0.96), aspirin -1.74 (exp -2.10).

---

## Development

```bash
nix develop ./config        # enter the dev shell (.venv, CPU torch, all deps)
python -m src.train                   # train the 5-model ensemble -> models/gnn_ensemble.pth
python -m src.evaluate                # 5-fold random-split CV (normalized ensemble)
python -m src.evaluate_scaffold_split # 5-fold scaffold-split CV (normalized ensemble)
python -m src.hyperparameter_tune     # Optuna search (N_TRIALS / TUNE_EPOCHS env-overridable)
python -m src.server                  # serve on :8080
pytest                                # run tests
```

GPU training/search (NixOS): the nix-provided torch is CPU-only, so a separate
CUDA venv is used. Create it once and run modules through the wrapper, which
clears the nix `PYTHONPATH` (it would shadow the CUDA torch) and sets up the
runtime libs:

```bash
uv venv .venv-gpu --python 3.12
uv pip install --python .venv-gpu torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-gpu torch_geometric rdkit pandas numpy scikit-learn scipy optuna
scripts/gpu-run.sh -m src.hyperparameter_tune   # or -m src.train, -m src.evaluate, ...
```

Models trained on GPU load fine on the CPU server (state dicts are device-agnostic
and loaded with `map_location`). Shared configuration (paths, hyperparameters,
dataset specs) lives in `src/config.py`; dataset loading/caching in `src/data.py`.
The dataset is downloaded once and cached under `data/`.

## Other Notes

- Supported atoms: C, N, O, F, S, Cl, Br, I.
- Web server runs on my k8s homelab.
