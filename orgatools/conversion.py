"""Convert the tif stacks of the Beghin lab to OME-Zarr."""

import os
from glob import glob

import numpy as np
import tifffile
import zarr
from zarr.codecs import BloscCodec
from ome_zarr.format import FormatV05
from ome_zarr.writer import write_image

# OME-Zarr 0.5 (zarr v3) with blosc zstd and byte shuffle: lossless and 2.7 - 3x smaller than the uncompressed tifs,
# smaller and faster than h5 (gzip) and compressed tifs. Byte shuffle beats bitshuffle on the 12 bit data.
COMPRESSOR = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
# Spatial chunks of 32 x 256 x 256 (4 MB in uint16) and shards of up to 4 x 4 x 4 chunks (256 MB).
CHUNKS = {"c": 1, "z": 32, "y": 256, "x": 256}
CHUNKS_PER_SHARD = 4
# The pyramid halves y and x (not z, it is coarse already) until the smallest level is below this size.
MIN_LEVEL_SIZE = 256


def read_tif(path):
    """Read a tif stack and its voxel size in micrometer from the ImageJ metadata.

    Returns the data with the axes in the OME-Zarr order (c, z, y, x), its axes and the voxel size of z, y and x.
    """
    with tifffile.TiffFile(path) as f:
        series = f.series[0]
        data = series.asarray()
        axes = series.axes.lower().replace("q", "z").replace("i", "z")
        metadata = f.imagej_metadata or {}
        x_resolution = series.pages[0].tags.get("XResolution")
    assert set(axes) <= set("czyx"), f"{path}: unsupported axes {series.axes}."
    order = [axes.index(axis) for axis in "czyx" if axis in axes]
    data = data.transpose(order)
    axes = "".join(axes[i] for i in order)

    voxel_size = {"z": float(metadata.get("spacing", 1.0)), "y": 1.0, "x": 1.0}
    if x_resolution is not None and x_resolution.value[0] > 0:
        numerator, denominator = x_resolution.value
        voxel_size["y"] = voxel_size["x"] = denominator / numerator
    return data, axes, voxel_size


def get_chunks(shape, axes):
    return tuple(min(size, CHUNKS[axis]) for size, axis in zip(shape, axes))


def get_shards(shape, axes, chunks):
    """Shards of up to CHUNKS_PER_SHARD chunks along each spatial axis, a multiple of the chunks."""
    return tuple(
        chunk if axis == "c" else min(-(-size // chunk), CHUNKS_PER_SHARD) * chunk
        for size, chunk, axis in zip(shape, chunks, axes)
    )


def get_scale_factors(shape, axes):
    y, x = shape[axes.index("y")], shape[axes.index("x")]
    factors = []
    while min(y, x) // 2 ** (len(factors) + 1) >= MIN_LEVEL_SIZE:
        factor = 2 ** (len(factors) + 1)
        factors.append({"y": factor, "x": factor})
    return factors


def write_ome_zarr(data, axes, path, voxel_size=None, pyramid=False, labels=False):
    """Write an image as OME-Zarr 0.5.

    Args:
        data: The image, with the axes in the order (c, z, y, x).
        axes: The axes of the image, e.g. 'zyx' or 'czyx'.
        path: The output path, by convention ending in '.ome.zarr'.
        voxel_size: The voxel size in micrometer of the axes 'z', 'y' and 'x'.
        pyramid: Whether to add downsampled levels for viewing, about 30% more space.
        labels: Whether the data holds labels, which are then downsampled with nearest neighbor interpolation.
    """
    chunks = get_chunks(data.shape, axes)
    storage_options = {"chunks": chunks, "shards": get_shards(data.shape, axes, chunks), "compressors": COMPRESSOR}
    scale_factors = get_scale_factors(data.shape, axes) if pyramid else []
    voxel_size = {axis: size for axis, size in (voxel_size or {}).items() if axis in axes}

    group = zarr.open_group(path, mode="w", zarr_format=3)
    write_image(
        data, group, scale_factors=scale_factors, method="nearest" if labels else "resize", axes=list(axes),
        fmt=FormatV05(), storage_options=storage_options, scale=voxel_size,
        axes_units={axis: "micrometer" for axis in voxel_size},
    )


def open_ome_zarr(path, level=0):
    """Open one level of an OME-Zarr image as a zarr array, 0 is the full resolution."""
    group = zarr.open_group(path, mode="r")
    return group[group.attrs["ome"]["multiscales"][0]["datasets"][level]["path"]]


def read_ome_zarr(path, level=0):
    """Read one level of an OME-Zarr image, 0 is the full resolution."""
    return open_ome_zarr(path, level)[:]


def convert_tif(input_path, output_path, labels=False, pyramid=False):
    """Convert a tif stack to OME-Zarr and check that the full resolution is identical to the tif."""
    data, axes, voxel_size = read_tif(input_path)
    write_ome_zarr(data, axes, output_path, voxel_size, pyramid=pyramid, labels=labels)
    assert np.array_equal(read_ome_zarr(output_path), data), f"{output_path} differs from {input_path}."
    return data.shape, axes, voxel_size


def find_tifs(input_path, output_folder):
    """Pairs of each tif and its OME-Zarr output path, a folder is searched recursively and its layout is kept."""
    if os.path.isdir(input_path):
        paths = sorted(glob(os.path.join(input_path, "**", "*.tif*"), recursive=True))
        relative_paths = [os.path.relpath(path, input_path) for path in paths]
    else:
        paths, relative_paths = [input_path], [os.path.basename(input_path)]
    return [
        (path, os.path.join(output_folder, relative_path.rsplit(".tif", 1)[0] + ".ome.zarr"))
        for path, relative_path in zip(paths, relative_paths)
    ]
