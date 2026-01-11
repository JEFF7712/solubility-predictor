# Soluble: A GNN Solubility Predictor
This project uses a Graph Neural Network (GNN) to treat molecules as graph structures and predict solubility. Trained on the Delaney dataset with ~1000 molecules.

<img width="2880" height="1800" alt="ss-2026-01-08-02-40-00" src="https://github.com/user-attachments/assets/6995f260-1354-40f4-9c67-26fab54a3a9d" />
Try it out yourself: https://soluble.rupan.dev

## The Pipeline
1. **Input**: A SMILES string (e.g., `CCO` for Ethanol).
2. **Graph Conversion** (RDKit):
   - **Nodes (Atoms)**: Featurized by Atomic Symbol (one-hot), Hybridization (SP/SP2/SP3), Mass, Formal Charge, Aromaticity, Hydrogen Count, Degree.
   - **Edges (Bonds)**: Connectivity + Bond Type Attributes (Single, Double, Triple, Aromatic).
3. **Inference** (PyTorch Geometric): Graph passes through a 3-layer GINE architecture to predict Log Solubility.

---

## Model Architecture
Uses a **GINEConv** backbone with edge attribute support:
- **Input Layer**: 17-dimensional Node Features + 4-dimensional Edge Features.
- **Hidden Layers**: 3× GINEConv layers with BatchNorm1d after each.
- **Residual Connections**: Skip connection from layer 2 → layer 3 to improve gradient flow.
- **Global Pooling**: Concatenates sum, mean, and max pooling to capture size-dependent and functional group features.
- **Prediction Head**: MLP (3 × hidden_dim → hidden_dim → 1) for final solubility prediction.

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
- **Key changes**: Better normalization technique, improved optimizer, and proper train/val/test split for unbiased evaluation.

## Model Performance Metrics

**5-Fold Cross-Validation** (Most Reliable):
- **Average RMSE**: 0.5938 ± 0.0243
- **Average R²**: 0.9187 (explains 91.87% of variance)
- **Fold 1**: RMSE = 0.5668, R² = 0.9320
- **Fold 2**: RMSE = 0.5686, R² = 0.9251
- **Fold 3**: RMSE = 0.6326, R² = 0.9035
- **Fold 4**: RMSE = 0.5991, R² = 0.9186
- **Fold 5**: RMSE = 0.6021, R² = 0.9145

✅ **Verdict**: Model is solid and generalizes consistently across different data splits.

---

## Limitations

- **Known Atoms Only**: Molecules outside [C, N, O, F, S, Cl, Br, I] are dropped.
- **No Stereochemistry**: Cannot distinguish between geometric isomers.
- **Solubility Range**: Trained on log solubility ≈ -5 to +5; extrapolation outside this range unreliable.
- **Dataset Bias**: Model inherits any biases from Delaney dataset.

---

## Other Notes

- Supported atoms: C, N, O, F, S, Cl, Br, I.
- Web/Server runs on my k8s homelab.