"""A small relation-aware GraphSAGE forecaster implemented with PyTorch only."""

import torch
from torch import nn
from torch.nn import functional as F


def encode_edge_types(edge_type_names, edge_type_order):
    """Encode stable string relation labels using an explicit, reproducible order."""
    if not isinstance(edge_type_order, (list, tuple)) or not edge_type_order:
        raise ValueError("edge_type_order must be a non-empty list")
    if len(set(edge_type_order)) != len(edge_type_order):
        raise ValueError("edge_type_order must not contain duplicate names")
    index = {name: position for position, name in enumerate(edge_type_order)}
    try:
        encoded = [index[name] for name in edge_type_names]
    except (KeyError, TypeError):
        raise ValueError("edge_type_names contains a relation missing from edge_type_order") from None
    return torch.tensor(encoded, dtype=torch.long), list(edge_type_order)


def aggregate_neighbors(x, edge_index, edge_weight=None, edge_type=None, *, num_edge_types):
    """Compute relation-wise weighted mean messages along source→target edges.

    Negative edge weights are treated as zero. This keeps cosine similarity
    edges with negative similarity from reversing a message; they contribute no
    neighbor signal, while the graph artifact still retains their original score.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or not x.is_floating_point():
        raise ValueError("x must be a 2D floating-point tensor")
    if not isinstance(num_edge_types, int) or isinstance(num_edge_types, bool) or num_edge_types < 1:
        raise ValueError("num_edge_types must be a positive integer")
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=x.device)
    else:
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    edge_count = edge_index.shape[1]
    if edge_count and (edge_index.min().item() < 0 or edge_index.max().item() >= x.shape[0]):
        raise ValueError("edge_index contains a node index outside x")

    if edge_weight is None:
        weights = torch.ones(edge_count, dtype=x.dtype, device=x.device)
    else:
        weights = torch.as_tensor(edge_weight, dtype=x.dtype, device=x.device)
    if weights.ndim != 1 or weights.shape[0] != edge_count:
        raise ValueError("edge_weight must contain one value per edge")
    if not torch.isfinite(weights).all().item():
        raise ValueError("edge_weight must contain only finite numbers")
    weights = weights.clamp_min(0.0)

    if edge_type is None:
        relations = torch.zeros(edge_count, dtype=torch.long, device=x.device)
    else:
        relations = torch.as_tensor(edge_type, dtype=torch.long, device=x.device)
    if relations.ndim != 1 or relations.shape[0] != edge_count:
        raise ValueError("edge_type must contain one relation index per edge")
    if edge_count and (relations.min().item() < 0 or relations.max().item() >= num_edge_types):
        raise ValueError("edge_type index is outside num_edge_types")

    source, target = edge_index
    relation_aggregates = []
    epsilon = torch.finfo(x.dtype).eps
    for relation in range(num_edge_types):
        mask = relations == relation
        sums = torch.zeros_like(x)
        normalizers = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        if mask.any().item():
            relation_weights = weights[mask]
            relation_source = source[mask]
            relation_target = target[mask]
            messages = x.index_select(0, relation_source) * relation_weights.unsqueeze(1)
            sums.index_add_(0, relation_target, messages)
            normalizers.index_add_(0, relation_target, relation_weights)
        relation_aggregates.append(sums / normalizers.clamp_min(epsilon).unsqueeze(1))
    return torch.stack(relation_aggregates, dim=1)


class HICGraphSAGEForecaster(nn.Module):
    """Two-layer relation-aware mean GraphSAGE head for a forecast horizon."""

    def __init__(self, input_dim, hidden_dim, horizon, edge_types, dropout=0.1):
        super().__init__()
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in (input_dim, hidden_dim, horizon)):
            raise ValueError("input_dim, hidden_dim, and horizon must be positive integers")
        if isinstance(edge_types, int) and not isinstance(edge_types, bool):
            relation_count = edge_types
            self.edge_type_names = None
        elif isinstance(edge_types, (list, tuple)) and edge_types:
            if len(set(edge_types)) != len(edge_types):
                raise ValueError("edge_types must not contain duplicate relation names")
            relation_count = len(edge_types)
            self.edge_type_names = list(edge_types)
        else:
            raise ValueError("edge_types must be a positive count or non-empty relation-name list")
        if relation_count < 1:
            raise ValueError("edge_types count must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.num_edge_types = relation_count
        self.input_self = nn.Linear(input_dim, hidden_dim)
        self.input_neighbors = nn.ModuleList(
            nn.Linear(input_dim, hidden_dim, bias=False) for _ in range(relation_count)
        )
        self.hidden_self = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_neighbors = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(relation_count)
        )
        self.output = nn.Linear(hidden_dim, horizon)
        self.dropout = float(dropout)

    def _message_layer(self, x, edge_index, edge_type, edge_weight, self_linear, neighbor_linears):
        aggregate = aggregate_neighbors(
            x,
            edge_index,
            edge_weight,
            edge_type,
            num_edge_types=self.num_edge_types,
        )
        output = self_linear(x)
        for relation, transform in enumerate(neighbor_linears):
            output = output + transform(aggregate[:, relation, :])
        return output

    def forward(self, x, edge_index, edge_type, edge_weight):
        hidden = self._message_layer(
            x, edge_index, edge_type, edge_weight, self.input_self, self.input_neighbors
        )
        hidden = F.dropout(F.relu(hidden), p=self.dropout, training=self.training)
        hidden = self._message_layer(
            hidden, edge_index, edge_type, edge_weight, self.hidden_self, self.hidden_neighbors
        )
        hidden = F.dropout(F.relu(hidden), p=self.dropout, training=self.training)
        return self.output(hidden)
