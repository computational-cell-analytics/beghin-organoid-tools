"""Instance segmentation of the MCF7 spheroid stacks with micro-sam2, the stacks have no ground truth.

The stacks are (Z, C, Y, X) with z 1.5 um and xy 0.414 um, channel 0 is DAPI in both sets.

Usage:
    python segment_mcf7_spheroids.py  # the nuclei on DAPI, as '<stack>_nucleus_prediction.tif'
    python segment_mcf7_spheroids.py --membrane  # the cells of the composites, as '<stack>_membrane_prediction.tif'
"""

import os
import time
import shutil
import argparse
import tempfile
from glob import glob

import tifffile
import numpy as np

from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter

from orgatools.segmentation import clean_grit, clean_spheroid_segmentation, write_prediction


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/mcf7_spheroids"
EMBEDDING_ROOT = os.environ.get("TMPDIR", tempfile.gettempdir())
MODEL_TYPE = "hvit_l_cells"
SPHEROID_SETS = ("dapi_ki67", "dapi_ecadh_btub")
DAPI_CHANNEL = 0
# The cells are segmented on (beta-tubulin, DAPI, E-cadherin) as RGB, which covers the most of the spheroid.
MEMBRANE_SET = "dapi_ecadh_btub"
MEMBRANE_CHANNELS = (2, 0, 1)


def get_stack_paths(data_root, spheroid_set):
    paths = sorted(glob(os.path.join(data_root, spheroid_set, "*.tif")))
    return [path for path in paths if "_prediction" not in os.path.basename(path)]


def segment(predictor, segmenter, raw):
    embedding_dir = tempfile.mkdtemp(dir=EMBEDDING_ROOT)
    try:
        return automatic_instance_segmentation(
            predictor, segmenter, input_path=raw, ndim=3, batch_size=None, verbose=False,
            embedding_path=os.path.join(embedding_dir, "embeddings.zarr"),
        )
    finally:
        shutil.rmtree(embedding_dir, ignore_errors=True)


def run_nuclei(data_root):
    """Segment the nuclei on the DAPI channel of every stack."""
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="ais", ndim=3)
    for spheroid_set in SPHEROID_SETS:
        for path in get_stack_paths(data_root, spheroid_set):
            name = os.path.splitext(os.path.basename(path))[0]
            raw = tifffile.imread(path)[:, DAPI_CHANNEL]
            start = time.time()
            segmentation = segment(predictor, segmenter, raw)
            cleaned = clean_spheroid_segmentation(segmentation, raw)
            n_objects, n_cleaned = len(np.unique(segmentation)) - 1, len(np.unique(cleaned)) - 1
            print(f"{spheroid_set} {name}: {n_objects} -> {n_cleaned} objects, {time.time() - start:.1f} s", flush=True)
            write_prediction(os.path.join(data_root, spheroid_set, f"{name}_nucleus_prediction.tif"), cleaned)


def run_membranes(data_root):
    """Segment the cells of the composite stacks on MEMBRANE_CHANNELS."""
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="ais", ndim=3)
    for path in get_stack_paths(data_root, MEMBRANE_SET):
        name = os.path.splitext(os.path.basename(path))[0]
        raw = np.moveaxis(tifffile.imread(path)[:, list(MEMBRANE_CHANNELS)], 1, -1)
        start = time.time()
        segmentation = segment(predictor, segmenter, raw)
        cleaned = clean_grit(segmentation)
        n_objects, n_cleaned = len(np.unique(segmentation)) - 1, len(np.unique(cleaned)) - 1
        print(f"{MEMBRANE_SET} {name}: {n_objects} -> {n_cleaned} objects, {time.time() - start:.1f} s", flush=True)
        write_prediction(os.path.join(data_root, MEMBRANE_SET, f"{name}_membrane_prediction.tif"), cleaned)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default=DATA_ROOT, help="The folder with the spheroid sets.")
    parser.add_argument("--membrane", action="store_true", help="Segment the cells instead of the nuclei.")
    args = parser.parse_args()

    if args.membrane:
        run_membranes(args.input_folder)
    else:
        run_nuclei(args.input_folder)


if __name__ == "__main__":
    main()
