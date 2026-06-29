"""Feature and target standardization.

Standardization is fit on the training split only and persisted alongside the
model so the exact same transform is applied at inference. Node features are
standardized only on the continuous columns (CONTINUOUS_FEATURE_INDICES); the
element/hybridization/aromatic one-hot columns pass through unchanged. Targets
(logS) are standardized too, which stabilizes training; predictions are mapped
back to logS with `inverse_y`.
"""

import torch

from .config import CONTINUOUS_FEATURE_INDICES, NUM_NODE_FEATURES


class GraphStandardizer:
    def __init__(self, feature_mean, feature_std, y_mean, y_std):
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

    @classmethod
    def fit(cls, data_list, continuous_indices=CONTINUOUS_FEATURE_INDICES):
        x = torch.cat([d.x for d in data_list], dim=0)
        col_mean = x.mean(dim=0)
        col_std = x.std(dim=0).clamp_min(1e-6)

        # Pass-through (mean 0, std 1) for every column except the continuous ones.
        feature_mean = torch.zeros(NUM_NODE_FEATURES)
        feature_std = torch.ones(NUM_NODE_FEATURES)
        for i in continuous_indices:
            feature_mean[i] = col_mean[i]
            feature_std[i] = col_std[i]

        y = torch.cat([d.y for d in data_list])
        y_std = y.std().clamp_min(1e-6).item()
        return cls(feature_mean, feature_std, y.mean().item(), y_std)

    def transform_x(self, x):
        # Operate on x's device: a scaler reloaded from a GPU checkpoint holds
        # CUDA tensors, while freshly-featurized input is on CPU.
        mean = self.feature_mean.to(x.device)
        std = self.feature_std.to(x.device)
        return (x - mean) / std

    def standardize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y):
        return y * self.y_std + self.y_mean

    def transform_data(self, data):
        """Return a copy of `data` with standardized node features and (if
        present) standardized target."""
        out = data.clone()
        out.x = self.transform_x(out.x)
        if "y" in out:
            out.y = self.standardize_y(out.y)
        return out

    def transform_dataset(self, data_list):
        return [self.transform_data(d) for d in data_list]

    def state(self):
        return {
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }

    @classmethod
    def from_state(cls, state):
        return cls(
            state["feature_mean"],
            state["feature_std"],
            state["y_mean"],
            state["y_std"],
        )
