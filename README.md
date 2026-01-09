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

## Final Model Metrics
- **MSE**: 0.5071
- **RMSE**: 0.7073
- **MAE**: 0.4829
- **R²**: 0.8780 (explains 87.8% of variance)
- **Pearson Correlation**: 0.9398 (p-value: 2.41e-107)
- **MAPE**: 55.92%

---

### Featurization
**Atom Features** (17-dim):
- Symbol: 8-dim one-hot (C, N, O, F, S, Cl, Br, I)
- Hybridization: 3-dim one-hot (SP, SP2, SP3)
- Mass: Normalized
- Formal Charge: [-2, 2]
- Aromaticity
- Hydrogen Count
- Degree

**Edge Features** (4-dim):
- Bond type: 4-dim one-hot (Single, Double, Triple, Aromatic)

---

## Limitations

- **Known Atoms Only**: Molecules outside [C, N, O, F, S, Cl, Br, I] are dropped.
- **No Stereochemistry**: Cannot distinguish between geometric isomers.
- **Solubility Range**: Trained on log solubility ≈ -5 to +5; extrapolation outside this range unreliable.
- **Dataset Bias**: Model inherits any biases from Delaney dataset.

---

## Other Notes

- Random seed set to 42 for reproducibility across runs.
- Supported atoms: C, N, O, F, S, Cl, Br, I.
- Server runs on personal k8s homelab.
