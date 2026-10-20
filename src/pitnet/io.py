from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output

@dataclass(frozen=True)
class AxisCrop:
    source_start: int
    source_end: int
    output_start: int
    output_end: int

def load_canonical_image(path: Path) -> nib.spatialimages.SpatialImage:
    return nib.as_closest_canonical(nib.load(str(path)))

def voxel_sizes(image: nib.spatialimages.SpatialImage) -> Tuple[float, float, float]:
    return tuple((float(value) for value in image.header.get_zooms()[:3]))

def prepare_working_image(path: Path, target_spacing: Sequence[float]) -> Tuple[nib.spatialimages.SpatialImage, nib.spatialimages.SpatialImage, nib.spatialimages.SpatialImage, bool]:
    original = nib.load(str(path))
    canonical = nib.as_closest_canonical(original)
    spacing = np.asarray(voxel_sizes(canonical), dtype=np.float64)
    target = np.asarray(target_spacing, dtype=np.float64)
    needs_resample = not np.allclose(spacing, target, rtol=0.0, atol=0.001)
    working = resample_to_output(canonical, voxel_sizes=tuple(target), order=1) if needs_resample else canonical
    return (original, canonical, working, needs_resample)

def align_mask_to_reference(mask_path: Path, canonical_reference: nib.spatialimages.SpatialImage, working_reference: nib.spatialimages.SpatialImage) -> Tuple[np.ndarray, str, float]:
    mask_image = load_canonical_image(mask_path)
    affine_difference = float(np.max(np.abs(np.asarray(mask_image.affine) - np.asarray(canonical_reference.affine))))
    if mask_image.shape == canonical_reference.shape:
        data = (np.asanyarray(mask_image.dataobj) > 0).astype(np.uint8)
        if affine_difference <= 0.0001:
            aligned = nib.Nifti1Image(data, canonical_reference.affine)
            mode = 'exact'
        else:
            aligned = nib.Nifti1Image(data, canonical_reference.affine)
            mode = 'voxel_shape_match'
    else:
        aligned = resample_from_to(mask_image, canonical_reference, order=0)
        mode = 'affine_resample'
    if aligned.shape != working_reference.shape or not np.allclose(aligned.affine, working_reference.affine, atol=0.0001):
        aligned = resample_from_to(aligned, working_reference, order=0)
        mode = f'{mode}+spacing_resample'
    array = (np.asanyarray(aligned.dataobj) > 0).astype(np.uint8)
    return (array, mode, affine_difference)

def compute_axis_crop(length: int, size: int, center: int) -> AxisCrop:
    start = int(center) - size // 2
    end = start + int(size)
    source_start = max(0, start)
    source_end = min(length, end)
    output_start = source_start - start
    output_end = output_start + (source_end - source_start)
    return AxisCrop(source_start, source_end, output_start, output_end)

def crop_and_pad(array: Any, crop_axes: Sequence[AxisCrop], crop_size: Sequence[int], dtype: np.dtype) -> np.ndarray:
    source_slices = tuple((slice(axis.source_start, axis.source_end) for axis in crop_axes))
    output_slices = tuple((slice(axis.output_start, axis.output_end) for axis in crop_axes))
    output = np.zeros(tuple((int(value) for value in crop_size)), dtype=dtype)
    output[output_slices] = np.asarray(array[source_slices], dtype=dtype)
    return output

def robust_normalize(image: np.ndarray) -> Tuple[np.ndarray, float, float]:
    finite = np.isfinite(image)
    values = image[finite & (image != 0)]
    if values.size == 0:
        raise ValueError('T1 crop contains no finite nonzero voxels.')
    lower, upper = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(f'Invalid intensity range: lower={lower}, upper={upper}')
    normalized = np.zeros(image.shape, dtype=np.float32)
    clipped = np.clip(image[finite], lower, upper)
    normalized[finite] = (clipped - lower) / (upper - lower)
    return (normalized, float(lower), float(upper))
