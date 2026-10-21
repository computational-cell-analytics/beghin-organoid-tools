"""Instance segmentation of the nus_organoids benchmark with micro-sam2 AIS or cellposeSAM, scored on the ground truth.

The data is the benchmark of the Digitalized Organoid paper: four uint16 volumes (ZYX) with an instance ground truth.
"""

import os
import time
import shutil
import argparse
import tempfile

import tifffile
import numpy as np

from elf.evaluation import matching, mean_segmentation_accuracy

from bioimage_py.evaluation import symmetric_best_dice_score

from bioimage_cpp.filters import gaussian_smoothing
from bioimage_cpp.transformation import affine_transform

from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter

from orgatools.segmentation import clean_spheroid_segmentation, write_prediction


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/nus_organoids"
OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/beghin_data"
EMBEDDING_ROOT = os.environ.get("TMPDIR", tempfile.gettempdir())
MODEL_TYPE = "hvit_l_cells"
# 'scale' resamples the input (z, y, x), 'anisotropy' is the z over xy spacing and 'min_size' is in original voxels.
SETTINGS = {
    "HCT116_cells_monolayer": {"scale": (1, 3, 3), "min_size": 500},
    "Human_colon_organoid": {"scale": (2, 2, 2), "anisotropy": 4, "min_size": 500},
    "ZeroG_breast_cancer_spheroid": {"clean": True},
    # The background subtraction removes the dim cell clusters around the nuclei.
    "TM00099_cell": {
        "background_sigma": 8, "norm_percentiles": (99, 99.99), "tile_shape": (4, 256, 256), "halo": (2, 52, 52),
    },
}


def evaluate(segmentation, labels):
    msa, accuracies = mean_segmentation_accuracy(segmentation, labels, return_accuracies=True)
    scores = matching(segmentation, labels, threshold=0.5)
    return {
        "objects": len(np.unique(segmentation)) - 1, "gt_objects": len(np.unique(labels)) - 1,
        "mSA": msa, "SA50": accuracies[0], "SA75": accuracies[5],
        "precision": scores["precision"], "recall": scores["recall"], "F1": scores["f1"],
        "SBD": symmetric_best_dice_score(segmentation, labels),
    }


def resample(data, shape, order):
    """Resample to 'shape' with the corners of the grids aligned, like 'scipy.ndimage.zoom'."""
    factors = [(size - 1) / max(new_size - 1, 1) for size, new_size in zip(data.shape, shape)]
    matrix = np.concatenate([np.diag(factors), np.zeros((data.ndim, 1))], axis=1)
    return affine_transform(data, matrix, bounding_box=tuple(slice(0, size) for size in shape), order=order)


def segment(predictor, segmenter, raw, settings):
    scale = settings.get("scale", (1, 1, 1))
    if "background_sigma" in settings:
        background = np.stack([gaussian_smoothing(plane, settings["background_sigma"]) for plane in raw])
        raw = np.clip(raw.astype("float32") - background, 0, None)
    shape = tuple(round(size * factor) for size, factor in zip(raw.shape, scale))
    volume = raw if scale == (1, 1, 1) else resample(raw.astype("float32"), shape, order=1)
    norm_bounds = None
    if "norm_percentiles" in settings:
        lower, upper = np.percentile(raw, settings["norm_percentiles"])
        norm_bounds = (np.array([lower], dtype="float32"), np.array([upper], dtype="float32"))
    kwargs = {}
    if "min_size" in settings:
        kwargs["min_size"] = int(settings["min_size"] * np.prod(scale))
    spacing_z = settings.get("anisotropy", 1) / scale[0]
    if spacing_z > 1:
        kwargs["spacing"] = (spacing_z, 1.0, 1.0)

    embedding_dir = tempfile.mkdtemp(dir=EMBEDDING_ROOT)
    try:
        segmentation = automatic_instance_segmentation(
            predictor, segmenter, input_path=volume, ndim=3, batch_size=None, verbose=False,
            embedding_path=os.path.join(embedding_dir, "embeddings.zarr"), norm_bounds=norm_bounds,
            tile_shape=settings.get("tile_shape"), halo=settings.get("halo"), **kwargs,
        )
    finally:
        shutil.rmtree(embedding_dir, ignore_errors=True)
    if segmentation.shape != raw.shape:
        segmentation = resample(segmentation.astype("uint32"), raw.shape, order=0)
    return segmentation


def run(data_root, names, output_folder, cellpose=False):
    os.makedirs(output_folder, exist_ok=True)
    if cellpose:
        from cellpose import models

        model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    else:
        predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="ais", ndim=3)
    for name in names:
        raw = tifffile.imread(os.path.join(data_root, f"{name}.tif"))
        labels = tifffile.imread(os.path.join(data_root, f"{name}_groundtruth.tif"))
        start = time.time()
        if cellpose:
            segmentation = model.eval(raw, z_axis=0, do_3D=True)[0].astype("uint32")
        else:
            segmentation = segment(predictor, segmenter, raw, SETTINGS[name])
            if SETTINGS[name].get("clean", False):
                segmentation = clean_spheroid_segmentation(segmentation, raw)
        runtime = time.time() - start
        scores = ", ".join(f"{key} {value:.3f}" for key, value in evaluate(segmentation, labels).items())
        print(f"{name}: {scores}, {runtime:.1f} s", flush=True)
        suffix = "_cellposeSAM" if cellpose else ""
        write_prediction(os.path.join(output_folder, f"{name}_prediction{suffix}.tif"), segmentation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default=DATA_ROOT, help="The folder with the volumes.")
    parser.add_argument("-o", "--output_folder", default=OUTPUT_ROOT, help="The folder for the segmentations.")
    parser.add_argument("-n", "--names", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    parser.add_argument("--cellpose", action="store_true", help="Segment with cellposeSAM in 3D instead.")
    args = parser.parse_args()
    run(args.input_folder, args.names, args.output_folder, args.cellpose)


if __name__ == "__main__":
    main()
