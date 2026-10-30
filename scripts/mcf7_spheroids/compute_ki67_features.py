"""Features of the MCF7 spheroids (dapi_ki67) on three levels: the spheroid, the nuclei and the Ki67 foci.

Uses the filtered nuclei and the dog spots of detect_ki67_spots.py. The spheroid is the closed union of the nuclei,
the foci are grown from the dog spots: a seeded watershed of the smoothed Ki67 in the nucleus, cut at half the spot
contrast above its local background. The focus ids are the row index + 1 of the dog table, so every spot has a focus.

The outputs in '<output>' are:
    spheroid/<stack>_spheroid.tif  the spheroid mask
    foci/<stack>_ki67_foci.tif     the foci labels, the label is the focus_id
    tables/<stack>_spheroid.csv    one row per stack
    tables/<stack>_nuclei.csv      one row per nucleus, the key is (stack, nucleus_id)
    tables/<stack>_foci.csv        one row per focus, the key is (stack, focus_id), nucleus_id links to the nuclei
Read all stacks with: pd.concat(map(pd.read_csv, glob("<output>/tables/*_nuclei.csv")), ignore_index=True).

The columns (besides stack):
    spheroid: volume_um3, surface_area_um2, sphericity, equivalent_diameter_um, axis_major/middle/minor_um,
        elongation, flatness, z_first, z_last, touches_z_border, n_nuclei, n_nuclei_with_foci, n_foci
    nuclei: nucleus_id, z, y, x, z_um, y_um, x_um, the shape as for the spheroid, solidity, touches_border,
        dapi_mean/median/std/integrated/texture, ki67_mean/median/integrated/mad, n_foci,
        foci_volume_total/mean/max_um3, foci_volume_fraction, foci_intensity_fraction, foci_peak_rel_mean,
        foci_radial_position_mean (NaN without foci), nn_ratio, nn_pvalue_clustered, nn_pvalue_dispersed,
        radial_ratio, foci_gyration_ratio (NaN below MIN_FOCI_STATS foci), in_spheroid, dist_to_spheroid_surface_um,
        depth_um, dist_to_spheroid_center_um, normalized_radius, n_neighbors_20um
    foci: focus_id, nucleus_id, z, y, x (the spot), z_um, y_um, x_um, volume_um3, n_voxels, equivalent_diameter_um,
        diameter_um (dog), axis_major/middle/minor_um, elongation, flatness, extent_z_um, peak, mean, integrated,
        background, contrast, snr (dog), peak_rel (to the nucleus Ki67 median), dapi_rel (the DAPI of the focus
        relative to the nucleus DAPI median), dist_to_nucleus_surface_um, radial_position, nn_distance_um (to the
        nearest focus of the nucleus)

Coordinates z, y, x are in voxels, *_um in micrometer. Intensities are in counts, 'integrated' is the sum above the
camera offset. The axes are the full lengths of the ellipsoid with the same second moments, elongation is major /
middle and flatness middle / minor. radial_position is 0 at the deepest point of the nucleus and 1 at its surface.

The spatial distribution of the foci (nuclei with >= MIN_FOCI_STATS foci) is compared to N_RANDOM random placements
of the same number of foci in the region where the foci are detected (the nucleus eroded by EROSION), with the
minimal distance NMS_RADIUS of the detection: nn_ratio is the mean nearest neighbour distance relative to the random
placements (< 1 clustered, > 1 dispersed) with the p-values nn_pvalue_clustered / nn_pvalue_dispersed, radial_ratio
the mean radial position relative to the random placements (< 1 central, > 1 peripheral).

The position in the spheroid: dist_to_spheroid_surface_um (the stack borders are not a surface, so for a spheroid cut
at a z-border it is the distance to the remaining surface, see touches_z_border), depth_um below the first slice of
the spheroid (the signal is attenuated with depth), dist_to_spheroid_center_um, normalized_radius
(0 at the centre, 1 at the surface) and n_neighbors_20um.

Usage:
    python compute_ki67_features.py  # all stacks
    python compute_ki67_features.py -n s10  # one stack
"""

import os
import time
import argparse

import tifffile
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes, mesh_surface_area, regionprops
from skimage.segmentation import watershed

from detect_ki67_spots import (
    DATA_ROOT, SUBSET, DAPI_CHANNEL, KI67_CHANNEL, SCALE, EROSION, NMS_RADIUS, PAD, DAPI_OFFSET,
    compute_dapi_texture,
)


SPOT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/beghin_organoids/mcf7_ki67_spots"
SPOT_METHOD = "dog"
OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/beghin_organoids/mcf7_ki67_features"
KI67_OFFSET = 150
VOXEL_VOLUME = float(np.prod(SCALE))
# The radius (um) of the closing of the nuclei to the spheroid.
SPHEROID_CLOSING = 10.0
# A focus is the part of its watershed basin above background + FOCUS_LEVEL * contrast, connected to its spot.
FOCUS_LEVEL = 0.5
SMOOTH = (0.5, 1, 1)
MIN_FOCI_STATS, N_RANDOM = 3, 200
NEIGHBOR_RADIUS = 20.0


def segment_spheroid(nuclei):
    """Close the union of the nuclei with a ball of SPHEROID_CLOSING (via distance transforms), fill the holes and
    keep the largest component."""
    pad = np.ceil(SPHEROID_CLOSING / SCALE).astype(int) + 1
    mask = np.pad(nuclei > 0, [(p, p) for p in pad])
    dilated = ndimage.distance_transform_edt(~mask, sampling=SCALE) <= SPHEROID_CLOSING
    closed = ndimage.distance_transform_edt(dilated, sampling=SCALE) > SPHEROID_CLOSING
    closed = closed[tuple(slice(p, -p) for p in pad)] | (nuclei > 0)
    closed = ndimage.binary_fill_holes(closed)
    closed = np.stack([ndimage.binary_fill_holes(plane) for plane in closed])
    components, n_components = ndimage.label(closed)
    if n_components > 1:
        sizes = np.bincount(components.ravel())[1:]
        closed = components == np.argmax(sizes) + 1
    return closed


def ellipsoid_axes(coords):
    """The full axis lengths (major, middle, minor) of the ellipsoid with the second moments of the voxels, each voxel
    a box (variance spacing^2 / 12), so that the small foci have no zero axes."""
    covariance = np.diag(SCALE ** 2 / 12)
    if len(coords) > 1:
        covariance = covariance + np.cov((coords * SCALE).T, bias=True)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    return 2 * np.sqrt(5 * np.maximum(eigenvalues, 0))


def surface_area(mask):
    vertices, faces, _, _ = marching_cubes(np.pad(mask, 1).astype("float32"), 0.5, spacing=tuple(SCALE))
    return mesh_surface_area(vertices, faces)


def shape_features(mask):
    """Volume, surface, sphericity and the ellipsoid axes of a mask."""
    volume = mask.sum() * VOXEL_VOLUME
    area = surface_area(mask)
    major, middle, minor = ellipsoid_axes(np.argwhere(mask))
    return {
        "volume_um3": volume, "surface_area_um2": area,
        "sphericity": np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area,
        "equivalent_diameter_um": (6 * volume / np.pi) ** (1 / 3),
        "axis_major_um": major, "axis_middle_um": middle, "axis_minor_um": minor,
        "elongation": major / max(middle, 1e-6), "flatness": middle / max(minor, 1e-6),
    }


def segment_foci(smoothed, mask, spots):
    """Grow the foci of one nucleus from its spots, returns the labels (row index of the spot + 1)."""
    seeds = np.zeros(mask.shape, "int32")
    idx = np.round(spots[["z", "y", "x"]].to_numpy()).astype(int)
    seeds[tuple(idx.T)] = spots.index.to_numpy() + 1
    basins = watershed(-smoothed, seeds, mask=mask)
    foci = np.zeros(mask.shape, "int32")
    for focus_id, point, background, contrast in zip(
        spots.index + 1, idx, spots.background.to_numpy(), spots.contrast.to_numpy()
    ):
        threshold = min(background + FOCUS_LEVEL * contrast, smoothed[tuple(point)])
        components, _ = ndimage.label((basins == focus_id) & (smoothed >= threshold))
        foci[components == components[tuple(point)]] = focus_id
    return foci


def random_placement(candidates, n, rng, max_tries=100):
    """n random voxels of the candidates without two closer than NMS_RADIUS, None if they do not fit."""
    radius = np.array([NMS_RADIUS[0], NMS_RADIUS[1], NMS_RADIUS[1]])
    points = np.zeros((n, 3))
    n_placed = 0
    for index in rng.integers(len(candidates), size=max_tries * n):
        point = candidates[index]
        if n_placed == 0 or (((points[:n_placed] - point) / radius) ** 2).sum(axis=1).min() > 1:
            points[n_placed] = point
            n_placed += 1
            if n_placed == n:
                return points
    return None


def mean_nn_distance(points):
    distances, _ = cKDTree(points * SCALE).query(points * SCALE, k=2)
    return distances[:, 1].mean()


def spatial_statistics(points, mask, radial, rng):
    """Compare the nearest neighbour distances and the radial positions of the foci to random placements."""
    eroded = ndimage.binary_erosion(mask, structure=np.ones((1, 3, 3), bool), iterations=EROSION)
    candidates = np.argwhere(eroded)
    nn_observed, radial_observed = mean_nn_distance(points), radial[tuple(points.T)].mean()
    nn_random, radial_random = [], []
    for _ in range(N_RANDOM):
        placement = random_placement(candidates, len(points), rng)
        if placement is None:
            continue
        placement = placement.astype(int)
        nn_random.append(mean_nn_distance(placement))
        radial_random.append(radial[tuple(placement.T)].mean())
    if not nn_random:
        return {}
    nn_random = np.array(nn_random)
    nucleus_coords = np.argwhere(mask) * SCALE
    nucleus_gyration = np.sqrt(((nucleus_coords - nucleus_coords.mean(axis=0)) ** 2).sum(axis=1).mean())
    foci_coords = points * SCALE
    foci_gyration = np.sqrt(((foci_coords - foci_coords.mean(axis=0)) ** 2).sum(axis=1).mean())
    return {
        "nn_ratio": nn_observed / nn_random.mean(),
        "nn_pvalue_clustered": (1 + (nn_random <= nn_observed).sum()) / (len(nn_random) + 1),
        "nn_pvalue_dispersed": (1 + (nn_random >= nn_observed).sum()) / (len(nn_random) + 1),
        "radial_ratio": radial_observed / np.mean(radial_random),
        "foci_gyration_ratio": foci_gyration / nucleus_gyration,
    }


def measure_foci(foci, mask, ki67, dapi, radial, distance, spots, offset, nucleus):
    """The features of the foci of one nucleus."""
    rows = []
    for focus_id, spot in zip(spots.index + 1, spots.itertuples()):
        focus = foci == focus_id
        coords = np.argwhere(focus)
        major, middle, minor = ellipsoid_axes(coords)
        point = tuple(np.round([spot.z, spot.y, spot.x]).astype(int) - offset)
        volume = len(coords) * VOXEL_VOLUME
        rows.append({
            "focus_id": focus_id, "nucleus_id": spot.nucleus_id, "z": spot.z, "y": spot.y, "x": spot.x,
            "volume_um3": volume, "n_voxels": len(coords),
            "equivalent_diameter_um": (6 * volume / np.pi) ** (1 / 3), "diameter_um": spot.diameter_um,
            "axis_major_um": major, "axis_middle_um": middle, "axis_minor_um": minor,
            "elongation": major / max(middle, 1e-6), "flatness": middle / max(minor, 1e-6),
            "extent_z_um": len(np.unique(coords[:, 0])) * SCALE[0],
            "peak": spot.peak, "mean": ki67[focus].mean(), "integrated": (ki67[focus] - KI67_OFFSET).sum(),
            "background": spot.background, "contrast": spot.contrast, "snr": spot.snr,
            "peak_rel": (spot.peak - KI67_OFFSET) / max(nucleus["ki67_median"] - KI67_OFFSET, 1),
            "dapi_rel": (dapi[focus].mean() - DAPI_OFFSET) / max(nucleus["dapi_median"] - DAPI_OFFSET, 1),
            "dist_to_nucleus_surface_um": distance[point], "radial_position": radial[point],
        })
    foci_table = pd.DataFrame(rows)
    if len(foci_table) > 1:
        coords = foci_table[["z", "y", "x"]].to_numpy() * SCALE
        foci_table["nn_distance_um"] = cKDTree(coords).query(coords, k=2)[0][:, 1]
    else:
        foci_table["nn_distance_um"] = np.nan
    return foci_table


def measure_nucleus(nuclei, ki67, dapi, smoothed, spots, nucleus_id, bbox, rng):
    """Segment the foci of one nucleus and measure the nucleus and its foci, returns the nucleus row, the foci table
    and the foci labels of the bounding box."""
    mask = nuclei[bbox] == nucleus_id
    ki67_values, dapi_values = ki67[bbox][mask], dapi[bbox][mask]
    offset = np.array([axis.start for axis in bbox])
    centroid = np.argwhere(mask).mean(axis=0) + offset
    ki67_median = np.median(ki67_values)
    nucleus = {
        "nucleus_id": nucleus_id, "z": centroid[0], "y": centroid[1], "x": centroid[2],
        **shape_features(mask), "solidity": regionprops(mask.astype("uint8"))[0].solidity,
        "touches_border": any(start + axis.start == 0 or start + axis.stop == size for start, axis, size in zip(
            offset, ndimage.find_objects(mask.astype("uint8"))[0], ki67.shape)),
        "dapi_mean": dapi_values.mean(), "dapi_median": np.median(dapi_values), "dapi_std": dapi_values.std(),
        "dapi_integrated": (dapi_values - DAPI_OFFSET).sum(),
        "ki67_mean": ki67_values.mean(), "ki67_median": ki67_median,
        "ki67_integrated": (ki67_values - KI67_OFFSET).sum(),
        "ki67_mad": 1.4826 * np.median(np.abs(ki67_values - ki67_median)),
        "n_foci": len(spots),
    }
    if len(spots) == 0:
        nucleus.update({"foci_volume_total_um3": 0.0, "foci_volume_fraction": 0.0, "foci_intensity_fraction": 0.0})
        return nucleus, None, None

    local_spots = spots.copy()
    local_spots[["z", "y", "x"]] -= offset
    foci = segment_foci(smoothed[bbox], mask, local_spots)
    distance = ndimage.distance_transform_edt(mask, sampling=SCALE)
    radial = 1 - distance / distance.max()
    foci_table = measure_foci(foci, mask, ki67[bbox], dapi[bbox], radial, distance, spots, offset, nucleus)
    in_foci = foci[mask] > 0
    nucleus.update({
        "foci_volume_total_um3": foci_table.volume_um3.sum(), "foci_volume_mean_um3": foci_table.volume_um3.mean(),
        "foci_volume_max_um3": foci_table.volume_um3.max(),
        "foci_volume_fraction": foci_table.volume_um3.sum() / nucleus["volume_um3"],
        "foci_intensity_fraction": (ki67_values[in_foci] - KI67_OFFSET).sum() / nucleus["ki67_integrated"],
        "foci_peak_rel_mean": foci_table.peak_rel.mean(),
        "foci_radial_position_mean": foci_table.radial_position.mean(),
    })
    if len(spots) >= MIN_FOCI_STATS:
        points = np.round(local_spots[["z", "y", "x"]].to_numpy()).astype(int)
        nucleus.update(spatial_statistics(points, mask, radial, rng))
    return nucleus, foci_table, foci


def add_um_coordinates(table):
    for axis, scale in zip("zyx", SCALE):
        table.insert(table.columns.get_loc("x") + 1 + "zyx".index(axis), f"{axis}_um", table[axis] * scale)
    return table


def spheroid_position(nuclei_table, spheroid):
    """The position of the nuclei in the spheroid."""
    centroids = nuclei_table[["z", "y", "x"]].to_numpy()
    idx = tuple(np.round(centroids).astype(int).T)
    surface_distance = ndimage.distance_transform_edt(spheroid, sampling=SCALE)[idx]
    center_distance = np.linalg.norm((centroids - np.argwhere(spheroid).mean(axis=0)) * SCALE, axis=1)
    tree = cKDTree(centroids * SCALE)
    return pd.DataFrame({
        "in_spheroid": spheroid[idx], "dist_to_spheroid_surface_um": surface_distance,
        "depth_um": (centroids[:, 0] - np.argwhere(spheroid.any(axis=(1, 2)))[0, 0]) * SCALE[0],
        "dist_to_spheroid_center_um": center_distance,
        "normalized_radius": center_distance / np.maximum(center_distance + surface_distance, 1e-6),
        "n_neighbors_20um": np.array([len(n) - 1 for n in tree.query_ball_point(centroids * SCALE, NEIGHBOR_RADIUS)]),
    }, index=nuclei_table.index)


def run_stack(name, input_folder, output_folder):
    start = time.time()
    stack = tifffile.imread(os.path.join(input_folder, SUBSET, f"{name}.tif"))
    ki67, dapi = stack[:, KI67_CHANNEL].astype("float32"), stack[:, DAPI_CHANNEL].astype("float32")
    nuclei = tifffile.imread(os.path.join(SPOT_ROOT, "nuclei", f"{name}_nucleus_filtered.tif"))
    spots = pd.read_csv(os.path.join(SPOT_ROOT, SPOT_METHOD, f"{name}_ki67_spots.csv"))
    assert nuclei.shape == ki67.shape, f"{name}: the nuclei shape does not match the stack."
    smoothed = ndimage.gaussian_filter(ki67, SMOOTH)
    spheroid = segment_spheroid(nuclei)

    rng = np.random.default_rng(0)
    foci = np.zeros(nuclei.shape, "uint16")
    nuclei_rows, foci_tables = [], []
    spots_per_nucleus = dict(list(spots.groupby("nucleus_id")))
    for nucleus_id, bbox in enumerate(ndimage.find_objects(nuclei), 1):
        if bbox is None:
            continue
        bbox = tuple(slice(max(b.start - p, 0), min(b.stop + p, n)) for b, p, n in zip(bbox, PAD, ki67.shape))
        nucleus_spots = spots_per_nucleus.get(nucleus_id, spots.iloc[:0])
        nucleus, foci_table, nucleus_foci = measure_nucleus(
            nuclei, ki67, dapi, smoothed, nucleus_spots, nucleus_id, bbox, rng
        )
        nuclei_rows.append(nucleus)
        if foci_table is not None:
            foci_tables.append(foci_table)
            foci[bbox][nucleus_foci > 0] = nucleus_foci[nucleus_foci > 0]

    nuclei_table = add_um_coordinates(pd.DataFrame(nuclei_rows))
    nuclei_table["dapi_texture"] = compute_dapi_texture(dapi, nuclei, nuclei_table.nucleus_id.to_numpy())[0]
    nuclei_table = pd.concat([nuclei_table, spheroid_position(nuclei_table, spheroid)], axis=1)
    foci_table = add_um_coordinates(pd.concat(foci_tables, ignore_index=True)) if foci_tables else pd.DataFrame()
    assert len(foci_table) == len(spots), f"{name}: {len(foci_table)} foci for {len(spots)} spots."

    z_planes = np.where(spheroid.any(axis=(1, 2)))[0]
    spheroid_table = pd.DataFrame([{
        **shape_features(spheroid), "z_first": z_planes[0], "z_last": z_planes[-1],
        "touches_z_border": z_planes[0] == 0 or z_planes[-1] == spheroid.shape[0] - 1,
        "n_nuclei": len(nuclei_table), "n_nuclei_with_foci": int((nuclei_table.n_foci > 0).sum()),
        "n_foci": len(foci_table),
    }])

    for folder in ("spheroid", "foci", "tables"):
        os.makedirs(os.path.join(output_folder, folder), exist_ok=True)
    tifffile.imwrite(os.path.join(output_folder, "spheroid", f"{name}_spheroid.tif"), spheroid.astype("uint8"),
                     compression="zlib")
    tifffile.imwrite(os.path.join(output_folder, "foci", f"{name}_ki67_foci.tif"), foci, compression="zlib")
    for level, table in (("spheroid", spheroid_table), ("nuclei", nuclei_table), ("foci", foci_table)):
        table.insert(0, "stack", name)
        table.round(4).to_csv(os.path.join(output_folder, "tables", f"{name}_{level}.csv"), index=False)
    print(f"{name}: spheroid diameter {spheroid_table.equivalent_diameter_um[0]:.0f} um, {len(nuclei_table)} nuclei, "
          f"{len(foci_table)} foci, {time.time() - start:.1f} s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default=DATA_ROOT, help="The folder with the spheroid sets.")
    parser.add_argument("-o", "--output_folder", default=OUTPUT_ROOT, help="The folder for the features.")
    parser.add_argument("-n", "--name", help="The name of a single stack, such as 's10'.")
    args = parser.parse_args()

    if args.name is None:
        names = [path.replace("_ki67_spots.csv", "") for path in os.listdir(os.path.join(SPOT_ROOT, SPOT_METHOD))]
        names = sorted(names, key=lambda name: int(name[1:]))
    else:
        names = [args.name]
    for name in names:
        run_stack(name, args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
