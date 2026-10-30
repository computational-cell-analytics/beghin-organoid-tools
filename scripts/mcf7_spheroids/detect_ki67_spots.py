"""Detection of the Ki67 spots in the nuclei of the MCF7 spheroids (dapi_ki67), in the micro-sam2 nuclei.

First the nuclei that are noise of the DAPI channel at the z-borders are removed, see filter_nuclei; the filtered
segmentation is saved as '<output>/nuclei/<stack>_nucleus_filtered.tif', the micro-sam2 segmentation is not changed.
Every nucleus is processed in its bounding box: the Ki67 channel is normalized to the median and MAD of the nucleus,
the spots are detected with one of METHODS and kept if they are in the nucleus eroded by EROSION, their contrast above
the local background is at least MIN_CONTRAST counts, their peak is at least MIN_PEAK counts and they are smaller than
the largest scale. Spots that are the signal of a neighbouring cell and spots that split one focus are removed.

The methods were compared on 5 stacks with synthetic spots, a cytoplasm control and visual checks:
'tophat_peaks' had the best recall at the same false positive rate, 'log' found the most faint spots,
'dog' agreed best with the other methods. All three are close, after the visual inspection 'dog' is used for the
further analysis.

The spots are saved as '<output>/<method>/<stack>_ki67_spots.csv' with the columns
z, y, x (voxels), nucleus_id, diameter_um (in xy, from the best LoG scale), peak, background, contrast (counts)
and snr (contrast in units of the nucleus MAD).

Usage:
    python detect_ki67_spots.py -o /path/to/output  # all methods, all stacks
    python detect_ki67_spots.py -o /path/to/output -m log -n s10  # one method, one stack
"""

import os
import time
import argparse
from glob import glob

import tifffile
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.feature import blob_dog, blob_log, peak_local_max


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/mcf7_spheroids"
OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/beghin_organoids/mcf7_ki67_spots"
SUBSET = "dapi_ki67"
DAPI_CHANNEL, KI67_CHANNEL = 0, 1
# The voxel size (z, y, x) in micrometer.
SCALE = np.array([1.5, 0.414, 0.414])

# The spots have sigma_xy 1.7 - 5 px and sigma_z 1.1 - 2.4 slices (Gaussian fits), the LoG scale is ~0.8 x that.
MIN_SIGMA, MAX_SIGMA, NUM_SIGMA = np.array([1.0, 1.5, 1.5]), np.array([2.5, 5.0, 5.0]), 8
SIGMAS = np.linspace(MIN_SIGMA, MAX_SIGMA, NUM_SIGMA)
# The detection thresholds in units of the nucleus MAD, see METHODS.
THRESHOLDS = {"log": 1.0, "dog": 0.5, "tophat_peaks": 2.0}
TOPHAT_SIZE = (5, 15, 15)
SMOOTH = (0.7, 1.0, 1.0)
PEAK_FOOTPRINT = np.ones((3, 5, 5), bool)
# The minimal spot contrast in counts: the Ki67 negative nuclei have noise peaks of 35 - 65 counts.
MIN_CONTRAST = 100
# The minimal (smoothed) spot intensity in counts, the camera offset is 150: dimmer spots look like noise.
MIN_PEAK = 450
# The spots are only kept at least EROSION + 1 px (in xy) inside the nucleus: at the rim they are the gradient from
# the nucleus to the dimmer cytoplasm.
EROSION = 2
# Spots closer than NMS_RADIUS (z, xy in voxels) are duplicates, spots closer than MERGE_DISTANCE (um) without
# an intensity dip of 10% between them are one focus.
NMS_RADIUS = (1.5, 2.5)
MERGE_DISTANCE, MERGE_MIN_DIP = 3.5, 0.9
SPILL_WINDOW = (1, 3, 3)
PAD = (2, 6, 6)
COLUMNS = ["z", "y", "x", "nucleus_id", "diameter_um", "peak", "background", "contrast", "snr"]

# The noise nuclei: a speckled DAPI layer below the spheroids and a few objects at the bottom. They have their centre
# within BORDER_SLICES of the first or last slice, a DAPI texture above MAX_TEXTURE and a DAPI median below MAX_DAPI.
# The texture is the std of the high-pass filtered DAPI without the camera noise, relative to the median above the
# offset; real nuclei at the z-borders have up to 0.16, or are brighter (nucleoli, chromatin).
BORDER_SLICES, MAX_TEXTURE, MAX_DAPI = 15, 0.2, 700
# The DAPI camera offset and noise: variance = READ_VARIANCE + GAIN * (intensity - OFFSET).
DAPI_OFFSET, DAPI_READ_VARIANCE, DAPI_GAIN = 158, 41, 12


def get_stack_paths(data_root):
    paths = sorted(glob(os.path.join(data_root, SUBSET, "*.tif")))
    return [path for path in paths if "_prediction" not in os.path.basename(path)]


def compute_dapi_texture(dapi, segmentation, ids):
    """The std of the high-pass filtered DAPI of each nucleus without the camera noise, relative to its median above
    the offset, returns the texture and the median."""
    dapi = dapi.astype("float32")
    residual = dapi - ndimage.gaussian_filter(dapi, (0, 1, 1))
    # The fraction of the noise variance that remains after the high-pass filter.
    noise_fraction = 0.759
    mean = ndimage.mean(dapi, segmentation, ids)
    median = ndimage.median(dapi, segmentation, ids)
    noise_variance = noise_fraction * (DAPI_READ_VARIANCE + DAPI_GAIN * np.maximum(mean - DAPI_OFFSET, 0))
    variance = ndimage.mean(residual ** 2, segmentation, ids) - ndimage.mean(residual, segmentation, ids) ** 2
    signal_variance = np.maximum(variance - noise_variance, 0)
    return np.sqrt(signal_variance) / np.maximum(median - DAPI_OFFSET, 1), median


def filter_nuclei(dapi, segmentation):
    """Remove the noise nuclei at the z-borders, returns the filtered segmentation and the removed ids."""
    ids = np.unique(segmentation)[1:]
    texture, median = compute_dapi_texture(dapi, segmentation, ids)
    z_center = np.array(ndimage.center_of_mass(np.ones_like(segmentation), segmentation, ids))[:, 0]
    at_border = (z_center < BORDER_SLICES) | (z_center > segmentation.shape[0] - 1 - BORDER_SLICES)
    removed = ids[at_border & (texture > MAX_TEXTURE) & (median < MAX_DAPI)]
    return np.where(np.isin(segmentation, removed), 0, segmentation), removed


def normalize_nucleus(crop, mask):
    """Normalize to the median and MAD of the nucleus and set the outside (beyond 2 px in xy) to the median."""
    values = crop[mask]
    median = np.median(values)
    mad = max(1.4826 * np.median(np.abs(values - median)), 1.0)
    inside = ndimage.binary_dilation(mask, structure=np.ones((1, 3, 3), bool), iterations=2)
    return np.where(inside, (crop - median) / mad, 0).astype("float32"), median, mad


def detect_log(norm):
    blobs = blob_log(norm, min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA, num_sigma=NUM_SIGMA,
                     threshold=THRESHOLDS["log"], overlap=0.5, exclude_border=False)
    return blobs[:, :3]


def detect_dog(norm):
    blobs = blob_dog(norm, min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA, sigma_ratio=1.6,
                     threshold=THRESHOLDS["dog"], overlap=0.5, exclude_border=False)
    return blobs[:, :3]


def detect_tophat_peaks(norm):
    """Local maxima of the smoothed white top-hat, which removes the background larger than the spots."""
    tophat = norm - ndimage.grey_opening(norm, size=TOPHAT_SIZE)
    tophat = ndimage.gaussian_filter(tophat, SMOOTH)
    return peak_local_max(tophat, footprint=PEAK_FOOTPRINT, threshold_abs=THRESHOLDS["tophat_peaks"],
                          exclude_border=False).astype("float64")


METHODS = {"log": detect_log, "dog": detect_dog, "tophat_peaks": detect_tophat_peaks}


def measure_spots(norm, mask, points):
    """The size from the best scale-normalized LoG, the peak, the local background in a ring and the contrast."""
    log = np.stack([-ndimage.gaussian_laplace(norm, sigma) * np.mean(sigma) ** 2 for sigma in SIGMAS])
    smoothed = ndimage.gaussian_filter(norm, (0.5, 1, 1))
    idx = np.round(points).astype(int)
    sigmas = SIGMAS[log[(slice(None),) + tuple(idx.T)].argmax(axis=0)]
    zz, yy, xx = np.indices(norm.shape)
    background = []
    for (z, y, x), (sigma_z, sigma_xy, _) in zip(idx, sigmas):
        distance = ((zz - z) / sigma_z) ** 2 + ((yy - y) / sigma_xy) ** 2 + ((xx - x) / sigma_xy) ** 2
        ring = (distance >= 4) & (distance <= 9) & mask
        background.append(np.median(norm[ring]) if ring.sum() > 5 else 0.0)
    peak = smoothed[tuple(idx.T)]
    return sigmas, peak, np.array(background)


def suppress_duplicates(points, scores):
    """Keep the highest scoring point within the ellipsoid NMS_RADIUS around each point."""
    order = np.argsort(-scores)
    points = points[order]
    tree = cKDTree(points / np.array([NMS_RADIUS[0], NMS_RADIUS[1], NMS_RADIUS[1]]))
    keep = np.ones(len(points), bool)
    for i, neighbors in enumerate(tree.query_ball_point(tree.data, 1.0)):
        if keep[i]:
            keep[[j for j in neighbors if j > i]] = False
    return order[keep]


def detect_nucleus(ki67, segmentation, nucleus_id, bbox, method):
    """Detect and measure the spots of one nucleus, returns a DataFrame in global voxel coordinates."""
    mask = segmentation[bbox] == nucleus_id
    norm, median, mad = normalize_nucleus(ki67[bbox], mask)
    points = METHODS[method](norm)
    if len(points) == 0:
        return None
    idx = np.clip(np.round(points).astype(int), 0, np.array(mask.shape) - 1)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((1, 3, 3), bool), iterations=EROSION)
    points = points[eroded[tuple(idx.T)]]
    if len(points) == 0:
        return None
    sigmas, peak, background = measure_spots(norm, mask, points)
    keep = suppress_duplicates(points, peak)
    offset = np.array([axis.start for axis in bbox])
    spots = pd.DataFrame(points[keep] + offset, columns=["z", "y", "x"])
    spots["nucleus_id"] = nucleus_id
    # The LoG scale is ~sqrt(2/3) of the Gaussian sigma of a 3D blob, the diameter is the FWHM in xy.
    spots["diameter_um"] = 2.355 * np.sqrt(1.5) * sigmas[keep, 1] * SCALE[1]
    spots["peak"] = median + peak[keep] * mad
    spots["background"] = median + background[keep] * mad
    spots["contrast"] = (peak[keep] - background[keep]) * mad
    spots["snr"] = peak[keep] - background[keep]
    spots["nucleus_median"] = median
    # A spot at the largest scale is not a resolved blob but a broad structure, the large foci are below it.
    return spots[(spots.contrast >= MIN_CONTRAST) & (spots.peak >= MIN_PEAK) & (sigmas[keep, 1] < MAX_SIGMA[1])]


def remove_spill(spots, smoothed, segmentation):
    """Remove the spots that are brighter just outside their nucleus than at the spot: signal of a neighbour."""
    keep = np.ones(len(spots), bool)
    idx = np.round(spots[["z", "y", "x"]].to_numpy()).astype(int)
    for i, (point, nucleus_id) in enumerate(zip(idx, spots.nucleus_id.to_numpy())):
        bbox = tuple(slice(max(c - w, 0), c + w + 1) for c, w in zip(point, SPILL_WINDOW))
        outside = segmentation[bbox] != nucleus_id
        if outside.any() and smoothed[bbox][outside].max() > smoothed[tuple(point)]:
            keep[i] = False
    return spots[keep]


def merge_split_spots(spots, smoothed, n_samples=16):
    """Merge the spots of a nucleus that are closer than MERGE_DISTANCE without an intensity dip between them,
    the dip is the minimum on the line relative to the dimmer spot, both above the nucleus median."""
    points = spots[["z", "y", "x"]].to_numpy()
    peak = ndimage.map_coordinates(smoothed, points.T, order=1)
    nucleus_ids, medians = spots.nucleus_id.to_numpy(), spots.nucleus_median.to_numpy()
    tree = cKDTree(points * SCALE)
    alive = np.ones(len(spots), bool)
    t = np.linspace(0, 1, n_samples)[1:-1, None]
    for i in np.argsort(-peak):
        if not alive[i]:
            continue
        for j in tree.query_ball_point(points[i] * SCALE, MERGE_DISTANCE):
            if j == i or not alive[j] or nucleus_ids[j] != nucleus_ids[i] or peak[j] > peak[i]:
                continue
            line = ndimage.map_coordinates(smoothed, (points[i] + t * (points[j] - points[i])).T, order=1)
            if (line.min() - medians[j]) / max(peak[j] - medians[j], 1.0) >= MERGE_MIN_DIP:
                alive[j] = False
    return spots[alive]


def detect_spots(ki67, segmentation, method):
    """Detect the Ki67 spots in all nuclei of a stack."""
    ki67 = ki67.astype("float32")
    spots = []
    for nucleus_id, bbox in enumerate(ndimage.find_objects(segmentation), 1):
        if bbox is None:
            continue
        bbox = tuple(slice(max(b.start - p, 0), min(b.stop + p, n)) for b, p, n in zip(bbox, PAD, ki67.shape))
        nucleus_spots = detect_nucleus(ki67, segmentation, nucleus_id, bbox, method)
        if nucleus_spots is not None and len(nucleus_spots):
            spots.append(nucleus_spots)
    if not spots:
        return pd.DataFrame(columns=COLUMNS)
    spots = pd.concat(spots, ignore_index=True)
    smoothed = ndimage.gaussian_filter(ki67, (0.5, 1, 1))
    spots = merge_split_spots(remove_spill(spots, smoothed, segmentation), smoothed)
    return spots[COLUMNS].reset_index(drop=True)


def run_stack(path, output_folder, methods):
    name = os.path.splitext(os.path.basename(path))[0]
    stack = tifffile.imread(path)
    ki67 = stack[:, KI67_CHANNEL]
    segmentation = tifffile.imread(os.path.join(os.path.dirname(path), f"{name}_nucleus_prediction.tif"))
    assert segmentation.shape == ki67.shape, f"{name}: the segmentation shape does not match the stack."
    segmentation, removed = filter_nuclei(stack[:, DAPI_CHANNEL], segmentation)
    os.makedirs(os.path.join(output_folder, "nuclei"), exist_ok=True)
    tifffile.imwrite(os.path.join(output_folder, "nuclei", f"{name}_nucleus_filtered.tif"), segmentation,
                     compression="zlib")
    n_nuclei = len(np.unique(segmentation)) - 1
    print(f"{name}: removed {len(removed)} noise nuclei at the z-borders, {n_nuclei} nuclei left", flush=True)
    for method in methods:
        start = time.time()
        spots = detect_spots(ki67, segmentation, method)
        os.makedirs(os.path.join(output_folder, method), exist_ok=True)
        spots.round(3).to_csv(os.path.join(output_folder, method, f"{name}_ki67_spots.csv"), index=False)
        n_with_spots = spots.nucleus_id.nunique()
        print(f"{name} {method}: {len(spots)} spots in {n_with_spots} / {n_nuclei} nuclei, {time.time() - start:.1f} s",
              flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default=DATA_ROOT, help="The folder with the spheroid sets.")
    parser.add_argument("-o", "--output_folder", default=OUTPUT_ROOT, help="The folder for the spot tables.")
    parser.add_argument("-m", "--method", choices=list(METHODS) + ["all"], default="all")
    parser.add_argument("-n", "--name", help="The name of a single stack, such as 's10'.")
    args = parser.parse_args()

    methods = list(METHODS) if args.method == "all" else [args.method]
    paths = get_stack_paths(args.input_folder)
    if args.name is not None:
        paths = [path for path in paths if os.path.basename(path) == f"{args.name}.tif"]
        assert paths, f"Could not find the stack {args.name}."
    for path in paths:
        run_stack(path, args.output_folder, methods)


if __name__ == "__main__":
    main()
