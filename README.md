# Soluble: A GNN Solubility Predictor
This project uses a Graph Neural Network (GNN) to treat molecules as graph structures and predict solubility. Trained on the Delaney dataset with ~1000 molecules.

<img width="2880" height="1800" alt="ss-2026-01-08-02-40-00" src="https://github.com/user-attachments/assets/6995f260-1354-40f4-9c67-26fab54a3a9d" />
Try it out yourself: https://soluble.rupan.dev

## Model Architecture
Uses a **GINEConv** backbone with edge attribute support:
- **Input Layer**: 16-dimensional Node Features + 4-dimensional Edge Features.
- **Hidden Layers**: 3× GINEConv layers with LayerNorm after each.
- **Residual Connections**: Skip connection from layer 2 → layer 3 to improve gradient flow.
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

## Model Performance Metrics

Metrics below are for the 5-model deep ensemble (Iteration 5). Each fold reserves
15% of its training data as a validation slice for early stopping.

**5-Fold Cross-Validation** (Random Split):
- **Average RMSE**: 0.5623 ± 0.0324  (was 0.5938 ± 0.0243 single-model)
- **Average R²**: 0.9271 (explains 92.71% of variance)
- **Fold 1**: RMSE = 0.5363, R² = 0.9392
- **Fold 2**: RMSE = 0.5331, R² = 0.9341
- **Fold 3**: RMSE = 0.6084, R² = 0.9107
- **Fold 4**: RMSE = 0.5946, R² = 0.9198
- **Fold 5**: RMSE = 0.5389, R² = 0.9315

**5-Fold Scaffold Split Cross-Validation** (More Rigorous - Prevents Data Leakage):
- **Average RMSE**: 0.8116 ± 0.0408  (was 0.7893 ± 0.0870 single-model)
- **Average R²**: 0.8157 (explains 81.57% of variance)
- **Fold 1**: Train Size: 811 | Test Size: 317 → RMSE = 0.7897, R² = 0.8122
- **Fold 2**: Train Size: 874 | Test Size: 254 → RMSE = 0.7543, R² = 0.7288
- **Fold 3**: Train Size: 942 | Test Size: 186 → RMSE = 0.8594, R² = 0.8720
- **Fold 4**: Train Size: 942 | Test Size: 186 → RMSE = 0.7975, R² = 0.8325
- **Fold 5**: Train Size: 943 | Test Size: 185 → RMSE = 0.8570, R² = 0.8329

On the scaffold split the headline RMSE is essentially unchanged (within fold
noise), but cross-fold variance roughly halved (±0.041 vs ±0.087), so the
ensemble is the more reliable predictor across unseen chemical scaffolds.

---

## Development

```bash
nix develop ./config        # enter the dev shell (.venv with all deps)
python -m src.train                   # train the 5-model ensemble -> models/gnn_ensemble.pth
python -m src.evaluate                # 5-fold random-split CV (normalized ensemble)
python -m src.evaluate_scaffold_split # 5-fold scaffold-split CV (normalized ensemble)
python -m src.hyperparameter_tune     # Optuna search
python -m src.server                  # serve on :8080
pytest                                # run tests
```

Shared configuration (paths, hyperparameters, dataset constants) lives in
`src/config.py`; dataset loading/caching in `src/data.py`. The Delaney CSV is
downloaded once and cached under `data/`.

## Other Notes

- Supported atoms: C, N, O, F, S, Cl, Br, I.
- Web server runs on my k8s homelab.
