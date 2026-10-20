from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

def _valid_group_count(num_channels: int, requested_groups: int) -> int:
    group_count = min(num_channels, requested_groups)
    while num_channels % group_count != 0:
        group_count -= 1
    return group_count

def replace_batch_norm_with_group_norm(module: nn.Module, requested_groups: int=8) -> nn.Module:
    if requested_groups < 1:
        raise ValueError('requested_groups must be >= 1.')
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            group_count = _valid_group_count(child.num_features, requested_groups)
            group_norm = nn.GroupNorm(num_groups=group_count, num_channels=child.num_features, eps=child.eps, affine=child.affine)
            if child.affine:
                with torch.no_grad():
                    group_norm.weight.copy_(child.weight)
                    group_norm.bias.copy_(child.bias)
            setattr(module, child_name, group_norm)
        else:
            replace_batch_norm_with_group_norm(child, requested_groups)
    return module

class ConvBNReLU(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(ConvBNReLU(in_channels, out_channels), ConvBNReLU(out_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DownBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels)
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels))
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)

class ResDownBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = ResidualBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))

class ResUpBlock(nn.Module):

    def __init__(self, low_channels: int, skip_channels: int, out_channels: int, bilinear: bool=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            up_channels = low_channels
        else:
            self.up = nn.ConvTranspose2d(low_channels, low_channels // 2, kernel_size=2, stride=2)
            up_channels = low_channels // 2
        self.block = ResidualBlock(up_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)

class Encoder2D(nn.Module):

    def __init__(self, in_channels: int, base_channels: int, residual: bool=False):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16
        if residual:
            self.inc = ResidualBlock(in_channels, c1)
            self.down1 = ResDownBlock(c1, c2)
            self.down2 = ResDownBlock(c2, c3)
            self.down3 = ResDownBlock(c3, c4)
            self.down4 = ResDownBlock(c4, c5)
        else:
            self.inc = DoubleConv(in_channels, c1)
            self.down1 = DownBlock(c1, c2)
            self.down2 = DownBlock(c2, c3)
            self.down3 = DownBlock(c3, c4)
            self.down4 = DownBlock(c4, c5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return (x1, x2, x3, x4, x5)

class FeatureAlignmentFusion(nn.Module):

    def __init__(self, channels: int, max_offset: float=4.0, enable_alignment: bool=True) -> None:
        super().__init__()
        hidden_channels = max(channels // 2, 16)
        self.max_offset = float(max_offset)
        self.enable_alignment = bool(enable_alignment)
        self.flow_predictor = nn.Sequential(ConvBNReLU(channels * 2, hidden_channels), nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1))
        self.confidence_predictor = nn.Sequential(ConvBNReLU(channels * 2, hidden_channels), nn.Conv2d(hidden_channels, 1, kernel_size=1))
        flow_output = self.flow_predictor[-1]
        assert isinstance(flow_output, nn.Conv2d)
        nn.init.zeros_(flow_output.weight)
        nn.init.zeros_(flow_output.bias)

    @staticmethod
    def _warp(feature: torch.Tensor, flow_pixels: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = feature.shape
        y_coords = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x_coords = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        base_grid = base_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        flow_x = flow_pixels[:, 0] * (2.0 / max(width - 1, 1))
        flow_y = flow_pixels[:, 1] * (2.0 / max(height - 1, 1))
        sampling_grid = base_grid + torch.stack((flow_x, flow_y), dim=-1)
        return F.grid_sample(feature, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

    def _align_one(self, center: torch.Tensor, neighbor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.enable_alignment:
            flow = torch.tanh(self.flow_predictor(torch.cat((center, neighbor), dim=1)))
            flow = flow * self.max_offset
            aligned = self._warp(neighbor, flow)
        else:
            aligned = neighbor
        confidence = torch.sigmoid(self.confidence_predictor(torch.cat((center, aligned), dim=1)))
        return (aligned, confidence)

    def forward(self, minus_feature: torch.Tensor, center_feature: torch.Tensor, plus_feature: torch.Tensor) -> torch.Tensor:
        minus_aligned, minus_confidence = self._align_one(center_feature, minus_feature)
        plus_aligned, plus_confidence = self._align_one(center_feature, plus_feature)
        denominator = 1.0 + minus_confidence + plus_confidence
        return (center_feature + minus_confidence * minus_aligned + plus_confidence * plus_aligned) / denominator

class ViewConfidenceContextGate(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(channels // 4, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.b_confidence = nn.Sequential(nn.Conv2d(channels * 2, hidden_channels, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, 1, kernel_size=1))
        self.v_confidence = nn.Sequential(nn.Conv2d(channels * 2, hidden_channels, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, 1, kernel_size=1))
        self.modulator = nn.Sequential(nn.Conv2d(channels * 2, hidden_channels, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, channels * 2, kernel_size=1))
        modulation_output = self.modulator[-1]
        assert isinstance(modulation_output, nn.Conv2d)
        nn.init.zeros_(modulation_output.weight)
        nn.init.zeros_(modulation_output.bias)

    def forward(self, coronal: torch.Tensor, b_view: torch.Tensor, v_view: torch.Tensor, b_keep: torch.Tensor, v_keep: torch.Tensor) -> torch.Tensor:
        coronal_vector = self.pool(coronal)
        b_vector = self.pool(b_view) * b_keep
        v_vector = self.pool(v_view) * v_keep
        b_weight = torch.sigmoid(self.b_confidence(torch.cat((coronal_vector, b_vector), dim=1))) * b_keep
        v_weight = torch.sigmoid(self.v_confidence(torch.cat((coronal_vector, v_vector), dim=1))) * v_keep
        context = torch.cat((b_weight * b_vector, v_weight * v_vector), dim=1)
        gamma, beta = self.modulator(context).chunk(2, dim=1)
        return coronal * (1.0 + torch.tanh(gamma)) + beta

class AlignedMultiViewResUNet2D(nn.Module):

    def __init__(self, in_channels: int=3, out_channels: int=6, base_channels: int=32, bilinear: bool=True, view_dropout: float=0.0, alignment_max_offset: float=4.0, enable_alignment: bool=True) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError('AlignedMultiViewResUNet2D requires three coronal channels.')
        if not 0.0 <= view_dropout < 1.0:
            raise ValueError('view_dropout must be in [0, 1).')
        self.view_dropout = float(view_dropout)
        self.enable_alignment = bool(enable_alignment)
        channels = [base_channels * 2 ** level for level in range(5)]
        self.coronal_encoder = Encoder2D(1, base_channels, residual=True)
        self.b_encoder = Encoder2D(1, base_channels, residual=True)
        self.v_encoder = Encoder2D(1, base_channels, residual=True)
        self.alignment_fusions = nn.ModuleList((FeatureAlignmentFusion(channel, max(alignment_max_offset / 2 ** level, 0.25), enable_alignment=self.enable_alignment) for level, channel in enumerate(channels)))
        self.context_gates = nn.ModuleList((ViewConfidenceContextGate(channel) for channel in channels))
        c1, c2, c3, c4, c5 = channels
        self.up1 = ResUpBlock(c5, c4, c4, bilinear=bilinear)
        self.up2 = ResUpBlock(c4, c3, c3, bilinear=bilinear)
        self.up3 = ResUpBlock(c3, c2, c2, bilinear=bilinear)
        self.up4 = ResUpBlock(c2, c1, c1, bilinear=bilinear)
        self.outc = nn.Conv2d(c1, out_channels, kernel_size=1)

    @staticmethod
    def _split_inputs(x: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(x, dict):
            return (x['coronal'], x['b'], x['v'])
        if torch.is_tensor(x) and x.dim() == 4 and (x.size(1) == 5):
            return (x[:, :3], x[:, 3:4], x[:, 4:5])
        raise TypeError('AlignedMultiViewResUNet2D expects coronal/b/v dict inputs.')

    def _view_keep_masks(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = reference.shape[0]
        shape = (batch_size, 1, 1, 1)
        if not self.training or self.view_dropout <= 0:
            ones = reference.new_ones(shape)
            return (ones, ones)
        b_keep = (torch.rand(shape, device=reference.device) >= self.view_dropout).to(reference.dtype)
        v_keep = (torch.rand(shape, device=reference.device) >= self.view_dropout).to(reference.dtype)
        both_dropped = b_keep + v_keep == 0
        restore_b = (torch.rand(shape, device=reference.device) >= 0.5).to(reference.dtype)
        b_keep = torch.where(both_dropped, restore_b, b_keep)
        v_keep = torch.where(both_dropped, 1.0 - restore_b, v_keep)
        return (b_keep, v_keep)

    def encode_coronal(self, coronal: torch.Tensor) -> list[torch.Tensor]:
        minus_features = self.coronal_encoder(coronal[:, 0:1])
        center_features = self.coronal_encoder(coronal[:, 1:2])
        plus_features = self.coronal_encoder(coronal[:, 2:3])
        return [align(minus, center, plus) for align, minus, center, plus in zip(self.alignment_fusions, minus_features, center_features, plus_features)]

    def add_multiview_context(self, coronal_features: list[torch.Tensor], b_view: torch.Tensor, v_view: torch.Tensor, reference: torch.Tensor) -> list[torch.Tensor]:
        b_features = self.b_encoder(b_view)
        v_features = self.v_encoder(v_view)
        b_keep, v_keep = self._view_keep_masks(reference)
        return [context_gate(coronal_feature, b_feat, v_feat, b_keep, v_keep) for context_gate, coronal_feature, b_feat, v_feat in zip(self.context_gates, coronal_features, b_features, v_features)]

    def decode_features(self, fused_features: list[torch.Tensor]) -> torch.Tensor:
        x1, x2, x3, x4, x5 = fused_features
        decoded = self.up1(x5, x4)
        decoded = self.up2(decoded, x3)
        decoded = self.up3(decoded, x2)
        decoded = self.up4(decoded, x1)
        return decoded

    def forward_features(self, x: Any, *, coronal_only: bool=False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        coronal, b_view, v_view = self._split_inputs(x)
        fused_features = self.encode_coronal(coronal)
        if not coronal_only:
            fused_features = self.add_multiview_context(fused_features, b_view, v_view, coronal)
        return (self.decode_features(fused_features), fused_features)

    def forward_coronal_features(self, coronal: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fused_features = self.encode_coronal(coronal)
        return (self.decode_features(fused_features), fused_features)

    def forward(self, x: Any) -> torch.Tensor:
        decoded, _ = self.forward_features(x)
        return self.outc(decoded)
