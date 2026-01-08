import torch
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d
from torch_geometric.nn import GINEConv, global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # 1. Edge Encoder (4 -> 12)
        self.edge_encoder = Linear(4, 12)
        
        # 2. GINE Layer 1
        mlp1 = Sequential(Linear(12, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv1 = GINEConv(mlp1)
        self.bn1 = BatchNorm1d(hidden_dim)
        
        # 3. GINE Layer 2
        self.edge_encoder2 = Linear(12, hidden_dim) 
        mlp2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv2 = GINEConv(mlp2)
        self.bn2 = BatchNorm1d(hidden_dim)

        # 4. GINE Layer 3
        self.edge_encoder3 = Linear(hidden_dim, hidden_dim)
        self.edge_lin = Linear(12, hidden_dim)

        mlp3 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv3 = GINEConv(mlp3)
        self.bn3 = BatchNorm1d(hidden_dim)

        # 5. Prediction Head
        self.head = Sequential(Linear(hidden_dim, hidden_dim // 2), ReLU(), Dropout(0.5), Linear(hidden_dim // 2, 1))
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Layer 1
        edge_attr_1 = self.edge_encoder(edge_attr)
        x = self.conv1(x, edge_index, edge_attr_1)
        x = self.bn1(x)
        x = x.relu()
        
        # Prepare edges for Layer 2 & 3 (Needs to match hidden_dim=64)
        edge_attr_deep = self.edge_lin(edge_attr_1)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr_deep)
        x = self.bn2(x)
        x = x.relu()
        
        # Layer 3
        x = self.conv3(x, edge_index, edge_attr_deep)
        x = self.bn3(x)
        x = x.relu()

        # Global Pooling
        x = global_mean_pool(x, batch)
        
        # Final Prediction
        return self.head(x)