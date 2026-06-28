import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import (
    GINEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from .config import NUM_EDGE_FEATURES, NUM_NODE_FEATURES


class GNN(torch.nn.Module):
    """GINEConv backbone with a configurable width/depth/dropout.

    Node and edge features are projected to `hidden_dim` once, then passed
    through `num_layers` residual GINEConv blocks (LayerNorm + ReLU + dropout).
    Graph embeddings concatenate sum/mean/max pooling before the MLP head.
    """

    def __init__(
        self,
        hidden_dim=160,
        num_layers=3,
        dropout=0.0,
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
    ):
        super().__init__()
        self.dropout = dropout

        self.node_encoder = Linear(num_node_features, hidden_dim)
        self.edge_encoder = Linear(num_edge_features, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = Sequential(
                Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            x_residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_residual

        x_graph = torch.cat(
            [
                global_add_pool(x, batch),
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
            ],
            dim=1,
        )
        return self.final_mlp(x_graph)
