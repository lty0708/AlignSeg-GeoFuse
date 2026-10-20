from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
from knosp_r_experiments import SliceAttentionEncoder

@dataclass(frozen=True)
class TargetSpec:
    classes: tuple[str, ...]
    ordinal: bool
TARGET_SPECS: OrderedDict[str, TargetSpec] = OrderedDict((('knosp_grade', TargetSpec(('0', '1', '2', '3', '4'), True)), ('hardy_grade', TargetSpec(('0', '1', '2', '3', '4'), True)), ('wilson_stage', TargetSpec(('A', 'B', 'C', 'D', 'E'), True)), ('left_invasion', TargetSpec(('0', '1'), False)), ('right_invasion', TargetSpec(('0', '1'), False)), ('left_compression', TargetSpec(('0', '1'), False)), ('right_compression', TargetSpec(('0', '1'), False))))

class MultiTaskR1(nn.Module):

    def __init__(self, geometry_dim: int, base_channels: int, embedding_dim: int, hidden_dim: int, dropout: float, target_names: tuple[str, ...] | None=None) -> None:
        super().__init__()
        self.target_names = tuple(TARGET_SPECS) if target_names is None else tuple(target_names)
        invalid_targets = [target for target in self.target_names if target not in TARGET_SPECS]
        if invalid_targets:
            raise ValueError(f'Unsupported target names: {invalid_targets}')
        if not self.target_names:
            raise ValueError('target_names cannot be empty.')
        self.image_encoder = SliceAttentionEncoder(base_channels, embedding_dim)
        self.geometry_encoder = nn.Sequential(nn.Linear(geometry_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.fusion_encoder = nn.Sequential(nn.Linear(embedding_dim + hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.fusion_heads = nn.ModuleDict({target: nn.Linear(hidden_dim, len(spec.classes)) for target, spec in TARGET_SPECS.items() if target in self.target_names})
        self.image_heads = nn.ModuleDict({target: nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, len(spec.classes))) for target, spec in TARGET_SPECS.items() if target in self.target_names})

    def forward(self, images: torch.Tensor, geometry: torch.Tensor) -> dict[str, Any]:
        image_embedding, attention = self.image_encoder(images)
        geometry_embedding = self.geometry_encoder(geometry)
        fused_embedding = self.fusion_encoder(torch.cat((image_embedding, geometry_embedding), dim=1))
        return {'logits': {target: head(fused_embedding) for target, head in self.fusion_heads.items()}, 'image_logits': {target: head(image_embedding) for target, head in self.image_heads.items()}, 'attention': attention}
