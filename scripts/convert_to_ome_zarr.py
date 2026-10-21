"""Convert tif stacks to OME-Zarr 0.5, lossless and checked against the tif after writing.

The input can be tif files or folders, a folder is searched recursively and its layout is kept in the output folder.

Usage:
    python convert_to_ome_zarr.py -i /path/to/stacks -o /path/to/output
"""

import os
import time
import argparse
from glob import glob

from orgatools.conversion import convert_tif


def find_tifs(inputs):
    """Pairs of each tif and its path relative to the input folder."""
    paths = []
    for input_path in inputs:
        if os.path.isdir(input_path):
            files = sorted(glob(os.path.join(input_path, "**", "*.tif*"), recursive=True))
            paths.extend((path, os.path.relpath(path, input_path)) for path in files)
        else:
            paths.append((input_path, os.path.basename(input_path)))
    return paths


def convert(inputs, output_folder, overwrite):
    for input_path, relative_path in find_tifs(inputs):
        name = relative_path.rsplit(".tif", 1)[0]
        output_path = os.path.join(output_folder, f"{name}.ome.zarr")
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping {output_path}, it exists already.")
            continue
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        start = time.time()
        shape, axes, voxel_size = convert_tif(input_path, output_path)
        print(f"Saved {output_path}: {axes} {shape}, voxel size {voxel_size}, {time.time() - start:.1f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputs", nargs="+", required=True, help="The tif files or folders with tif files.")
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert(args.inputs, args.output_folder, args.overwrite)


if __name__ == "__main__":
    main()
