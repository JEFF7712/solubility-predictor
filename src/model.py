import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool

class GNN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # GINE Layer 1
        self.edge_encoder = Linear(4, 16)
        mlp1 = Sequential(Linear(16, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv1 = GINEConv(mlp1)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # GINE Layer 2
        self.edge_encoder2 = Linear(16, hidden_dim) 
        mlp2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv2 = GINEConv(mlp2)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # GINE Layer 3
        self.edge_encoder3 = Linear(hidden_dim, hidden_dim)
        mlp3 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv3 = GINEConv(mlp3)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # Prediction Head
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Layer 1
        edge_attr_1 = self.edge_encoder(edge_attr)
        x = self.conv1(x, edge_index, edge_attr_1)
        x = self.ln1(x)
        x = x.relu()
        
        # Layer 2
        edge_attr_2 = self.edge_encoder2(edge_attr_1)
        x = self.conv2(x, edge_index, edge_attr_2)
        x = self.ln2(x)
        x = x.relu()
        x_residual = x

        # Layer 3
        edge_attr_3 = self.edge_encoder3(edge_attr_2)
        x = self.conv3(x, edge_index, edge_attr_3)
        x = x + x_residual
        x = self.ln3(x)
        x = x.relu()

        # Global Pooling
        x_sum = global_add_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        x_graph = torch.cat([x_sum, x_mean, x_max], dim=1)

        # Final Prediction
        return self.final_mlp(x_graph)