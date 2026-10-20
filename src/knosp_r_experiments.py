from __future__ import annotations
import math
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
VIEW_FEATURES = ('tumor_width_over_ica_span', 'tumor_height_over_ica_span', 'tumor_axis_offset_abs', 'tumor_perp_offset_abs', 'signed_distance_min', 'signed_distance_max', 'signed_distance_abs_diff', 'contact_min', 'contact_max', 'contact_abs_diff', 'encasement_min', 'encasement_max', 'encasement_abs_diff', 'outer_extension_min', 'outer_extension_max', 'outer_extension_abs_diff')
GEOMETRY_FEATURES = tuple((f'm_{feature}' for feature in VIEW_FEATURES))

def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan

def serialize(value: float) -> float | str:
    return float(value) if math.isfinite(value) else ''

def centroid(mask: np.ndarray) -> np.ndarray | None:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None
    return coordinates.mean(axis=0).astype(np.float64)

def largest_component_fraction(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    labels, count = ndimage.label(mask)
    if count <= 1:
        return 1.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes.max() / max(sizes.sum(), 1))

def signed_gap(tumor: np.ndarray, vessel: np.ndarray) -> float:
    if not tumor.any() or not vessel.any():
        return math.nan
    overlap = int((tumor & vessel).sum())
    if overlap:
        return -math.sqrt(overlap / math.pi)
    distance = ndimage.distance_transform_edt(~vessel)
    return float(distance[tumor].min())

def contact_and_encasement(tumor: np.ndarray, vessel: np.ndarray) -> tuple[float, float]:
    if not tumor.any() or not vessel.any():
        return (math.nan, math.nan)
    near_tumor = ndimage.binary_dilation(tumor, iterations=2)
    contact = float((near_tumor & vessel).sum() / max(int(vessel.sum()), 1))
    eroded = ndimage.binary_erosion(vessel)
    boundary = vessel & ~eroded
    encasement = float((ndimage.binary_dilation(tumor, iterations=3) & boundary).sum() / max(int(boundary.sum()), 1))
    return (min(contact, 1.0), min(encasement, 1.0))

def pair_stats(right: float, left: float) -> tuple[float, float, float]:
    if not math.isfinite(right) or not math.isfinite(left):
        return (math.nan, math.nan, math.nan)
    return (min(right, left), max(right, left), abs(right - left))

def view_geometry(masks: dict[str, np.ndarray]) -> dict[str, float]:
    tumor = masks['tumor']
    right = masks['right_ica']
    left = masks['left_ica']
    right_center = centroid(right)
    left_center = centroid(left)
    tumor_center = centroid(tumor)
    output = {name: math.nan for name in VIEW_FEATURES}
    if right_center is None or left_center is None or tumor_center is None:
        return output
    axis = left_center - right_center
    span = float(np.linalg.norm(axis))
    if span <= 1e-06:
        return output
    unit_axis = axis / span
    unit_perp = np.asarray((-unit_axis[1], unit_axis[0]))
    midpoint = (right_center + left_center) / 2.0
    tumor_coordinates = np.argwhere(tumor).astype(np.float64)
    relative = tumor_coordinates - midpoint
    axis_projection = relative @ unit_axis / span
    centered_tumor = tumor_center - midpoint
    ys, xs = np.where(tumor)
    output['tumor_width_over_ica_span'] = float((xs.max() - xs.min() + 1) / span)
    output['tumor_height_over_ica_span'] = float((ys.max() - ys.min() + 1) / span)
    output['tumor_axis_offset_abs'] = float(abs(centered_tumor @ unit_axis) / span)
    output['tumor_perp_offset_abs'] = float(abs(centered_tumor @ unit_perp) / span)
    right_distance = signed_gap(tumor, right) / span
    left_distance = signed_gap(tumor, left) / span
    output['signed_distance_min'], output['signed_distance_max'], output['signed_distance_abs_diff'] = pair_stats(right_distance, left_distance)
    right_contact, right_encasement = contact_and_encasement(tumor, right)
    left_contact, left_encasement = contact_and_encasement(tumor, left)
    output['contact_min'], output['contact_max'], output['contact_abs_diff'] = pair_stats(right_contact, left_contact)
    output['encasement_min'], output['encasement_max'], output['encasement_abs_diff'] = pair_stats(right_encasement, left_encasement)
    right_extension = max(0.0, -0.5 - float(axis_projection.min()))
    left_extension = max(0.0, float(axis_projection.max()) - 0.5)
    output['outer_extension_min'], output['outer_extension_max'], output['outer_extension_abs_diff'] = pair_stats(right_extension, left_extension)
    return output

def erosion_survival(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    eroded = ndimage.binary_erosion(mask)
    ratio = float(eroded.sum() / max(int(mask.sum()), 1))
    return min(1.0, ratio / 0.5)

def case_geometry_and_quality(masks_by_view: dict[str, dict[str, np.ndarray]]) -> dict[str, float | str]:
    output: dict[str, float | str] = {}
    values = view_geometry(masks_by_view['m'])
    for name, value in values.items():
        output[f'm_{name}'] = serialize(value)
    right = masks_by_view['m']['right_ica']
    left = masks_by_view['m']['left_ica']
    presence = 0.5 * float(right.any()) + 0.5 * float(left.any())
    component = 0.5 * (largest_component_fraction(right) + largest_component_fraction(left))
    survival = 0.5 * (erosion_survival(right) + erosion_survival(left))
    non_overlap = 1.0 if not right.any() or not left.any() else 1.0 - float((right & left).sum() / max(min(right.sum(), left.sum()), 1))
    non_overlap = float(np.clip(non_overlap, 0.0, 1.0))
    output['qc_m_presence'] = presence
    output['qc_m_component'] = component
    output['qc_m_erosion_survival'] = survival
    output['qc_m_left_right_non_overlap'] = non_overlap
    output['ica_quality_score'] = float(np.clip(0.4 * presence + 0.25 * component + 0.25 * survival + 0.1 * non_overlap, 0.0, 1.0))
    return output

class GeometryPreprocessor:

    def __init__(self, names: list[str], stats: dict[str, dict[str, list[float]]]):
        self.names = names
        self.stats = stats

    @classmethod
    def fit(cls, names: list[str], feature_by_case: dict[str, dict[str, str]], manifest_by_case: dict[str, dict[str, str]], train_ids: list[str]) -> 'GeometryPreprocessor':
        stats: dict[str, dict[str, list[float]]] = {}
        domains = sorted({manifest_by_case[case_id]['domain'] for case_id in train_ids})
        for domain in domains:
            domain_ids = [case_id for case_id in train_ids if manifest_by_case[case_id]['domain'] == domain]
            matrix = np.asarray([[finite_float(feature_by_case[case_id].get(name, '')) for name in names] for case_id in domain_ids], dtype=np.float32)
            medians = np.nanmedian(matrix, axis=0)
            medians = np.where(np.isfinite(medians), medians, 0.0)
            filled = np.where(np.isfinite(matrix), matrix, medians)
            means = filled.mean(axis=0)
            stds = filled.std(axis=0)
            stds[stds < 1e-06] = 1.0
            stats[domain] = {'medians': medians.tolist(), 'means': means.tolist(), 'stds': stds.tolist()}
        return cls(names, stats)

    def transform(self, row: dict[str, str], domain: str) -> np.ndarray:
        values = np.asarray([finite_float(row.get(name, '')) for name in self.names], dtype=np.float32)
        missing = ~np.isfinite(values)
        stats = self.stats[domain]
        medians = np.asarray(stats['medians'], dtype=np.float32)
        means = np.asarray(stats['means'], dtype=np.float32)
        stds = np.asarray(stats['stds'], dtype=np.float32)
        filled = np.where(missing, medians, values)
        normalized = (filled - means) / stds
        return np.concatenate([normalized, missing.astype(np.float32)])

    def to_dict(self) -> dict[str, Any]:
        return {'names': self.names, 'stats': self.stats}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'GeometryPreprocessor':
        return cls(list(value['names']), dict(value['stats']))

class ConvBlock(nn.Sequential):

    def __init__(self, input_channels: int, output_channels: int, stride: int=1):
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        super().__init__(nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(groups, output_channels), nn.GELU())

class SliceAttentionEncoder(nn.Module):

    def __init__(self, base_channels: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(ConvBlock(1, base_channels, 2), ConvBlock(base_channels, base_channels), ConvBlock(base_channels, base_channels * 2, 2), ConvBlock(base_channels * 2, base_channels * 2), ConvBlock(base_channels * 2, base_channels * 4, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base_channels * 4, embedding_dim), nn.GELU())
        self.attention = nn.Sequential(nn.Linear(embedding_dim, embedding_dim // 2), nn.Tanh(), nn.Linear(embedding_dim // 2, 1))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, views, channels, height, width = images.shape
        embeddings = self.encoder(images.reshape(batch * views, channels, height, width)).reshape(batch, views, -1)
        weights = torch.softmax(self.attention(embeddings).squeeze(-1), dim=1)
        pooled = (embeddings * weights.unsqueeze(-1)).sum(dim=1)
        return (pooled, weights)
