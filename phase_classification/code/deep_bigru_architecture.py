"""
Deep bidirectional GRU architecture for per-frame phase classification.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DeepBidirectionalGRU(nn.Module):
    """
    Three-layer bidirectional GRU with separate forward and backward stacks.

    Input shape is (batch, sequence_length, input_dim). Output shape is
    (batch, sequence_length, output_dim), so every frame receives a prediction.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (64, 128, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one GRU layer size")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = dropout

        forward_layers = []
        backward_layers = []
        in_dim = input_dim
        for hidden_dim in self.hidden_dims:
            forward_layers.append(nn.GRU(in_dim, hidden_dim, batch_first=True))
            backward_layers.append(nn.GRU(in_dim, hidden_dim, batch_first=True))
            in_dim = hidden_dim

        self.forward_layers = nn.ModuleList(forward_layers)
        self.backward_layers = nn.ModuleList(backward_layers)
        self.dropout = nn.Dropout(dropout)

        head_input_dim = self.hidden_dims[-1] * 2
        self.classifier = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(max(dropout - 0.1, 0.0)),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward_out = x
        backward_out = torch.flip(x, dims=[1])

        for layer_idx, (forward_gru, backward_gru) in enumerate(zip(self.forward_layers, self.backward_layers)):
            forward_out, _ = forward_gru(forward_out)
            backward_out, _ = backward_gru(backward_out)
            if layer_idx < len(self.forward_layers) - 1:
                forward_out = self.dropout(forward_out)
                backward_out = self.dropout(backward_out)

        backward_out = torch.flip(backward_out, dims=[1])
        combined = torch.cat([forward_out, backward_out], dim=-1)
        return self.classifier(combined)

    def get_total_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


def create_deep_bigru_model(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int] = (64, 128, 256),
    dropout: float = 0.3,
) -> DeepBidirectionalGRU:
    return DeepBidirectionalGRU(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
