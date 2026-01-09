# Soluble: A GNN Solubility Predictor
This project uses a Graph Neural Network (GNN) to treat molecules as graph structures rather than simple images or text strings.

<img width="2880" height="1800" alt="ss-2026-01-08-02-40-00" src="https://github.com/user-attachments/assets/6995f260-1354-40f4-9c67-26fab54a3a9d" />
Try it out yourself: https://soluble.rupan.dev

### The Pipeline
1. Input: A SMILES string (e.g., `CCO` for Ethanol).
2. Graph Conversion (RDKit):
   - Nodes (Atoms): Featurized by Atomic Symbol (One-Hot), Hybridization (SP/SP2/SP3), and Mass.
   - Edges (Bonds): Connectivity matrix + Bond Type Attributes (Single, Double, Triple, Aromatic).
3. Inference (PyTorch Geometric): The graph is passed through a 3-layer GINE architecture to predict Log Solubility.

### Model Architecture
Uses a GINEConv backbone, which is specifically designed to handle edge attributes (bond types) natively.
- Input Layer: 12-dimensional Node Features + 4-dimensional Edge Features.
- Hidden Layers: 3x GINEConv layers.
- Global Pooling: global_sum_pool combines all atom vectors into a single molecule vector.
- Prediction Head: A final MLP projects the molecular embedding to a scalar solubility value.

---

## Model Notes

### Initial model
- Started with a shallow, single-layer GINE model.
- Issue: The model treated atoms only based on immediate neighbors (1-hop). This results in it failing to capture global molecular geometry.

### Updated model 1
Introduced four key optimizations to reduce error by ~65%:

1. Increased Depth (1 → 3 Layers): Allows information to propagate 3 hops across the molecule, capturing long-range dependencies (e.g., how a polar group on one end affects the whole structure).
2. Batch Normalization: Added 'BatchNorm1d' after every convolution to stabilize training and allow for higher learning rates.
3. Regularization:
    * Dropout (0.5): Randomly disables neurons in the prediction head to prevent overfitting and force redundancy.
    * Weight Decay (5e-4): Penalizes large weights in the optimizer to smooth the loss landscape.
4.  Learning Rate Scheduler: Implemented 'ReduceLROnPlateau' to dynamically lower the learning rate when
validation loss slows down. This allows the model to reach lower loss be more optimized.

### Updated model 2
Introduced a few more optimizations to reduce error by a further ~34%:

1. Hybrid Global Pooling: Refactored the readout phase to combine sum, mean, and max pooling outputs. This allows the model to capture size-dependent properties (sum), average molecular characteristics (mean), and dominant functional group features (max).
2. Residual Skip Connections: Implemented residual connections (x = x + residual) between GNN layers to mitigates the vanishing gradient problem and prevent oversmoothing.
3. Expanded Atom Featurization: Expanded input node features to include Formal Charge, Aromaticity, Number of Hydrogens, and Degree.
4. Simplified Regularization: Removed Dropout layers from the architecture.

## Model eval metrics
- MSE (Mean Squared Error): 0.5128
- RMSE (Root Mean Squared Error): 0.7118
- MAE (Mean Absolute Error): 0.4773
- R² (Coefficient of Determination): 0.8764
- Pearson Correlation: 0.9384 (p-value: 2.90e-106)
- MAPE (Mean Absolute Percentage Error): 58.40%

---
## Other Notes

- Known Atoms = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']
- Cannot distinguish between geometric isomers
