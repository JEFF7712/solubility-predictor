# GNN Solubility Predictor
This project uses a Graph Neural Network (GNN) to treat molecules as graph structures rather than simple images or text strings.

Try it out yourself [here](https://soluble.rupan.dev)

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
- Global Pooling: global_mean_pool combines all atom vectors into a single molecule vector.
- Prediction Head: A final MLP projects the molecular embedding to a scalar solubility value.

---

## Model Notes

### Initial model
- Started with a shallow, single-layer GINE model.
- Issue: The model treated atoms only based on immediate neighbors (1-hop). This results in it failing to capture global molecular geometry.

### Updated model
Introduced four key optimizations to reduce error by ~65%:

1. Increased Depth (1 → 3 Layers): Allows information to propagate 3 hops across the molecule, capturing long-range dependencies (e.g., how a polar group on one end affects the whole structure).
2. Batch Normalization: Added 'BatchNorm1d' after every convolution to stabilize training and allow for higher learning rates.
3. Regularization:
    * Dropout (0.5): Randomly disables neurons in the prediction head to prevent overfitting and force redundancy.
    * Weight Decay (5e-4): Penalizes large weights in the optimizer to smooth the loss landscape.
4.  Learning Rate Scheduler: Implemented 'ReduceLROnPlateau' to dynamically lower the learning rate when
validation loss slows down. This allows the model to reach lower loss be more optimized.



NOTE: Front end (static/) is mostly vibe coded as my main focus was on the model
