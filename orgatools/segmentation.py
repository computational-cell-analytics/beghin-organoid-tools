"""Post-processing and export of the instance segmentations."""

import os

import tifffile
import numpy as np
from scipy import ndimage

from bioimage_cpp.segmentation import relabel_sequential

# Objects below LARGE_SIZE are fragments or noise, nuclei are >= 3500 voxels in 90%.
LARGE_SIZE = 2000
FRAGMENT_MIN_ENCLOSED = 0.6
NEAR_XY, NEAR_Z = 15, 5
SMALL_MIN_SIZE = 300
SMALL_MAX_CV = 0.45
# Flat blobs no brighter than their surrounding ring, and grainy, thin speckles below the spheroid.
NOISE_MIN_CONTRAST = 1.1
SPECKLE_MIN_CV = 0.5
SPECKLE_MAX_Z_EXTENT = 9
NOISE_RING_WIDTH = 3
# Small grit inside the cells of the membrane segmentation.
GRIT_MAX_SIZE = 1000
GRIT_MIN_ENCLOSED = 0.5


def remove_noise_objects(segmentation, raw):
    """Remove the flat noise blobs and the speckles."""
    raw = raw.astype("float32")
    bboxes = ndimage.find_objects(segmentation)
    removed = []
    for object_id, bbox in enumerate(bboxes, 1):
        if bbox is None:
            continue
        bbox = tuple(slice(max(axis.start - NOISE_RING_WIDTH, 0), axis.stop + NOISE_RING_WIDTH) for axis in bbox)
        labels = segmentation[bbox]
        mask = labels == object_id
        ring = ndimage.binary_dilation(mask, iterations=NOISE_RING_WIDTH) & (labels == 0)
        values = raw[bbox][mask]
        contrast = values.mean() / raw[bbox][ring].mean() if ring.any() else np.inf
        z_extent = int(mask.any(axis=(1, 2)).sum())
        is_speckle = values.std() / values.mean() >= SPECKLE_MIN_CV and z_extent <= SPECKLE_MAX_Z_EXTENT
        if contrast < NOISE_MIN_CONTRAST or is_speckle:
            removed.append(object_id)
    if removed:
        lut = np.arange(len(bboxes) + 1, dtype=segmentation.dtype)
        lut[removed] = 0
        segmentation = lut[segmentation]
    return segmentation


def clean_spheroid_segmentation(segmentation, raw):
    """Merge the fragments into their nuclei and remove the frame, small and grainy objects and the noise."""
    raw = raw.astype("float32")
    bboxes = ndimage.find_objects(segmentation)
    lut = np.arange(len(bboxes) + 1, dtype=segmentation.dtype)
    for object_id, bbox in enumerate(bboxes, 1):
        if bbox is None or (segmentation[bbox] == object_id).sum() >= LARGE_SIZE:
            continue
        padded = tuple(slice(max(axis.start - 1, 0), axis.stop + 1) for axis in bbox)
        labels = segmentation[padded]
        mask = labels == object_id
        border = labels[ndimage.binary_dilation(mask) & ~mask]
        neighbors, counts = np.unique(border[border != 0], return_counts=True)
        if len(neighbors) and counts.max() / len(border) >= FRAGMENT_MIN_ENCLOSED:
            lut[object_id] = neighbors[counts.argmax()]
    segmentation = lut[segmentation]

    ids, counts = np.unique(segmentation, return_counts=True)
    near = np.isin(segmentation, ids[(counts >= LARGE_SIZE) & (ids != 0)])
    near = ndimage.binary_dilation(near, structure=np.ones((2 * NEAR_Z + 1, 1, 1), bool))
    near = ndimage.binary_dilation(near, structure=np.ones((1, 3, 3), bool), iterations=NEAR_XY)
    bboxes = ndimage.find_objects(segmentation)
    lut = np.arange(len(bboxes) + 1, dtype=segmentation.dtype)
    for object_id, bbox in enumerate(bboxes, 1):
        if bbox is None:
            continue
        mask = segmentation[bbox] == object_id
        size = int(mask.sum())
        if size >= LARGE_SIZE:
            continue
        values = raw[bbox][mask]
        if size < SMALL_MIN_SIZE or near[bbox][mask].mean() < 0.5 or values.std() / values.mean() >= SMALL_MAX_CV:
            lut[object_id] = 0
    return remove_noise_objects(lut[segmentation], raw)


def clean_grit(segmentation):
    """Merge the objects below GRIT_MAX_SIZE into the neighbor that encloses them, or remove them."""
    segmentation = segmentation.copy()
    object_ids, sizes = np.unique(segmentation[segmentation > 0], return_counts=True)
    bboxes = ndimage.find_objects(segmentation)
    for object_id in object_ids[sizes < GRIT_MAX_SIZE]:
        bbox = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in bboxes[object_id - 1])
        crop = segmentation[bbox]
        mask = crop == object_id
        if not mask.any():
            continue
        border = crop[ndimage.binary_dilation(mask) & ~mask]
        neighbors, counts = np.unique(border, return_counts=True)
        best = np.argmax(np.where(neighbors > 0, counts, 0))
        enclosed = neighbors[best] > 0 and counts[best] >= GRIT_MIN_ENCLOSED * border.size
        crop[mask] = neighbors[best] if enclosed else 0
    return relabel_sequential(segmentation)[0]


def write_prediction(path, segmentation):
    """Write uint16 instance labels as a deflate compressed tif, like the Fiji files, and check it."""
    assert segmentation.max() <= np.iinfo("uint16").max, f"{path}: {segmentation.max()} labels do not fit uint16."
    tifffile.imwrite(path, segmentation.astype("uint16"), compression="zlib")
    assert np.array_equal(tifffile.imread(path), segmentation), f"{path} differs after reading it back."
    os.chmod(path, 0o664)
    print(f"Saved {path}: {len(np.unique(segmentation)) - 1} objects")
