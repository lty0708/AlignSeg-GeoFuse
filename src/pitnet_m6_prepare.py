from __future__ import annotations
import numpy as np
from scipy import ndimage

def split_bilateral_carotids(mask_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary = np.asarray(mask_xyz > 0, dtype=np.uint8)
    labels, count = ndimage.label(binary, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count < 2:
        raise ValueError(f'Expected bilateral ICA components, found {count}.')
    component_sizes = np.bincount(labels.ravel())[1:]
    largest = np.argsort(component_sizes)[-2:] + 1
    centroids = {int(component): ndimage.center_of_mass(binary, labels, int(component))[0] for component in largest}
    right_component = max(centroids, key=centroids.get)
    left_component = min(centroids, key=centroids.get)
    right_center = float(centroids[right_component])
    left_center = float(centroids[left_component])
    midpoint = (right_center + left_center) / 2.0
    x_coordinates = np.arange(binary.shape[0], dtype=np.float32)[:, None, None]
    right = (labels == right_component) | (binary > 0) & (x_coordinates >= midpoint)
    left = (labels == left_component) | (binary > 0) & (x_coordinates < midpoint)
    overlap = right & left
    if np.any(overlap):
        raise RuntimeError('Right/left ICA split produced overlapping voxels.')
    if not np.array_equal(right | left, binary > 0):
        raise RuntimeError('Right/left ICA split lost foreground voxels.')
    return (right.astype(np.uint8), left.astype(np.uint8))
