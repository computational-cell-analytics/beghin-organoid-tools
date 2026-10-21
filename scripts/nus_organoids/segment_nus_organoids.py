"""Instance segmentation of the nus_organoids benchmark with micro-sam2 and cellposeSAM, scored on the ground truth.

The data is the benchmark of the Digitalized Organoid paper: four uint16 volumes (ZYX) with an instance ground truth.

Usage:
    python segment_nus_organoids.py  # AIS on every volume
    python segment_nus_organoids.py --best  # AIS with the best settings of each volume
    python segment_nus_organoids.py -e ais apg -n ZeroG_breast_cancer_spheroid  # AIS and APG on one volume
    python segment_nus_organoids.py --clean -n HCT116_cells_monolayer  # the spheroid cleanup, stored as 'ais_clean'
    python segment_nus_organoids.py --cellpose  # cellposeSAM on every volume
    python segment_nus_organoids.py --sweep -n HCT116_cells_monolayer --scale 1 2 2  # the APG variants at this scale
    python segment_nus_organoids.py --export  # the segmentations as tifs in the data folder
    python segment_nus_organoids.py --view -n TM00099_cell  # the data and segmentations in napari
"""

import os
import time
import shutil
import argparse
import tempfile

import h5py
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
ENGINES = ("ais", "apg")
NAMES = ("HCT116_cells_monolayer", "Human_colon_organoid", "TM00099_cell", "ZeroG_breast_cancer_spheroid")
# 'scale' resamples the input (z, y, x), 'anisotropy' is the z over xy spacing and 'min_size' is in original voxels.
DEFAULT_SETTINGS = {
    "scale": (1, 1, 1), "anisotropy": 1, "min_size": None, "norm_lower": None, "norm_upper": None,
    "norm_percentiles": None, "background_sigma": None, "tile_shape": None, "halo": None, "clean": False, "apg": None,
}
BEST_SETTINGS = {
    "HCT116_cells_monolayer": {"scale": (1, 3, 3), "min_size": 500},  # mSA 0.648 -> 0.750
    "Human_colon_organoid": {"scale": (2, 2, 2), "anisotropy": 4, "min_size": 500},  # mSA 0.390 -> 0.493
    "ZeroG_breast_cancer_spheroid": {"clean": True},  # mSA 0.129 -> 0.417
    # The background subtraction removes the dim cell clusters around the nuclei, mSA 0.038 -> 0.210 on a crop.
    "TM00099_cell": {
        "background_sigma": 8, "norm_percentiles": (99, 99.99), "tile_shape": (4, 256, 256), "halo": (2, 52, 52),
    },
}
# The APG variants of '--sweep', each one change of the 'generate' defaults, on top of the best settings.
APG_VARIANTS = {
    "default": {},
    "refine": {"refinement": "points+boxes"},
    "score0.5": {"score_threshold": 0.5},
    "score0.7": {"score_threshold": 0.7},
    "score0.8": {"score_threshold": 0.8},
    "score0.9": {"score_threshold": 0.9},
    "overlap0.3": {"max_overlap": 0.3},
    "overlap0.7": {"max_overlap": 0.7},
    "sigma0.5": {"sigma": 0.5},
    "sigma1": {"sigma": 1.0},
    "sigma2": {"sigma": 2.0},
    "ladder_low": {"candidate_threshold": (1.0, 5.0)},
    "ladder_high": {"candidate_threshold": (3.0, 10.0)},
    "fg0.3": {"foreground_threshold": 0.3},
    "fg0.5": {"foreground_threshold": 0.5},
    "flow_long": {"n_iter": 100, "dt": 1.0},
    # 'min_size' and 'spacing' are on the scaled grid here.
    "min100": {"min_size": 100},
    "min1000": {"min_size": 1000},
    "spacing1": {"spacing": None},
    "spacing2": {"spacing": (2.0, 1.0, 1.0)},
    "spacing4": {"spacing": (4.0, 1.0, 1.0)},
    "spacing6": {"spacing": (6.0, 1.0, 1.0)},
    "nomulti": {"multimasking": False},
    "maxsize1.5": {"max_size_factor": 1.5},
    "maxsize2": {"max_size_factor": 2.0},
    "maxsize3": {"max_size_factor": 3.0},
}
CELLPOSE_MODES = ("3d", "2d_stitch")
CELLPOSE_STITCH_THRESHOLD = 0.5


def load_data(data_root, name):
    raw = tifffile.imread(os.path.join(data_root, f"{name}.tif"))
    labels = tifffile.imread(os.path.join(data_root, f"{name}_groundtruth.tif"))
    assert raw.shape == labels.shape, f"{name}: raw {raw.shape} and ground truth {labels.shape} differ."
    return raw, labels


def evaluate(segmentation, labels):
    msa, accuracies = mean_segmentation_accuracy(segmentation, labels, return_accuracies=True)
    scores = matching(segmentation, labels, threshold=0.5)
    return {
        "objects": len(np.unique(segmentation)) - 1, "gt_objects": len(np.unique(labels)) - 1,
        "mSA": msa, "SA50": accuracies[0], "SA75": accuracies[5],
        "precision": scores["precision"], "recall": scores["recall"], "F1": scores["f1"],
        "SBD": symmetric_best_dice_score(segmentation, labels),
    }


def format_value(value):
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def format_scores(scores):
    return ", ".join(f"{key} {format_value(value)}" for key, value in scores.items())


def print_table(results):
    print("\n" + "\t".join(results[0].keys()))
    for result in results:
        print("\t".join(format_value(value) for value in result.values()))


def save_segmentation(output_path, raw, labels, key, segmentation, attrs):
    with h5py.File(output_path, "a") as f:
        for name, data in (("raw", raw), ("gt", labels)):
            if name not in f:
                f.create_dataset(name, data=data, compression="gzip")
        if key in f:
            del f[key]
        f.create_dataset(key, data=segmentation, compression="gzip")
        f[key].attrs.update(attrs)


def resample(data, shape, order):
    """Resample to 'shape' with the corners of the grids aligned, like 'scipy.ndimage.zoom'."""
    factors = [(size - 1) / max(new_size - 1, 1) for size, new_size in zip(data.shape, shape)]
    matrix = np.concatenate([np.diag(factors), np.zeros((data.ndim, 1))], axis=1)
    return affine_transform(data, matrix, bounding_box=tuple(slice(0, size) for size in shape), order=order)


def segment(raw, engine, settings, models=None, embedding_path=None):
    """Segment a volume, the embeddings go to a tmp folder unless 'embedding_path' is given."""
    settings = {**DEFAULT_SETTINGS, **settings}
    scale = tuple(settings["scale"])
    if models is None:
        models = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode=engine, ndim=3)
    predictor, segmenter = models
    start = time.time()

    if settings["background_sigma"] is not None:
        background = np.stack([gaussian_smoothing(plane, settings["background_sigma"]) for plane in raw])
        raw = np.clip(raw.astype("float32") - background, 0, None)
    shape = tuple(round(size * factor) for size, factor in zip(raw.shape, scale))
    volume = raw if scale == (1, 1, 1) else resample(raw.astype("float32"), shape, order=1)
    norm_bounds = None
    if settings["norm_percentiles"] is not None:
        lower, upper = np.percentile(raw, settings["norm_percentiles"])
        norm_bounds = (np.array([lower], dtype="float32"), np.array([upper], dtype="float32"))
    elif settings["norm_lower"] is not None or settings["norm_upper"] is not None:
        lower = np.percentile(raw, 2) if settings["norm_lower"] is None else settings["norm_lower"]
        upper = np.percentile(raw, 98) if settings["norm_upper"] is None else settings["norm_upper"]
        norm_bounds = (np.array([lower], dtype="float32"), np.array([upper], dtype="float32"))
    generate_kwargs = {}
    if settings["min_size"] is not None:
        generate_kwargs["min_size"] = int(settings["min_size"] * np.prod(scale))
    spacing_z = settings["anisotropy"] / scale[0]
    if spacing_z > 1:
        generate_kwargs["spacing"] = (spacing_z, 1.0, 1.0)
    if engine == "apg" and settings["apg"] is not None:
        generate_kwargs.update(settings["apg"])

    embedding_dir = tempfile.mkdtemp(dir=EMBEDDING_ROOT) if embedding_path is None else None
    try:
        segmentation = automatic_instance_segmentation(
            predictor, segmenter, input_path=volume, ndim=3, batch_size=None, verbose=False,
            embedding_path=os.path.join(embedding_dir, "embeddings.zarr") if embedding_path is None else embedding_path,
            norm_bounds=norm_bounds, tile_shape=settings["tile_shape"], halo=settings["halo"], **generate_kwargs,
        )
    finally:
        if embedding_dir is not None:
            shutil.rmtree(embedding_dir, ignore_errors=True)
    if segmentation.shape != raw.shape:
        segmentation = resample(segmentation.astype("uint32"), raw.shape, order=0)
    return segmentation, time.time() - start


def run(data_root, names, engines, best, output_folder, clean=False, scale=None):
    """Segment and score the volumes, '--clean' stores the result as '<engine>_clean'."""
    os.makedirs(output_folder, exist_ok=True)
    results = []
    for name in names:
        raw, labels = load_data(data_root, name)
        settings = {**DEFAULT_SETTINGS, **(BEST_SETTINGS[name] if best else {})}
        if scale is not None:
            settings["scale"] = tuple(scale)
        if clean or settings["clean"]:
            settings.update({"min_size": None, "clean": True})
        for engine in (engines or ["ais"]):
            segmentation, runtime = segment(raw, engine, settings)
            if settings["clean"]:
                start = time.time()
                segmentation = clean_spheroid_segmentation(segmentation, raw)
                runtime += time.time() - start
            key = f"{engine}_clean" if clean else engine
            scores = evaluate(segmentation, labels)
            results.append({"name": name, "engine": key, "time [s]": runtime, **scores})
            print(f"{name} {key.upper()}: {format_scores(scores)}, {runtime:.1f} s", flush=True)
            output_path = os.path.join(output_folder, f"{name}_{MODEL_TYPE}.h5")
            attrs = {**scores, "settings": str({**settings, "engine": engine})}
            save_segmentation(output_path, raw, labels, key, segmentation, attrs)
    print_table(results)


def parse_variant(variant):
    """A '+'-joined combination of 'APG_VARIANTS' keys, e.g. 'refine+score0.7', applied from left to right."""
    kwargs = {}
    for key in variant.split("+"):
        kwargs.update(APG_VARIANTS[key])
    return kwargs


def sweep(data_root, name, scale, variants, output_folder, tag=None):
    """Run the APG variants on one volume from the same embeddings, and skip the variants that exist already."""
    raw, labels = load_data(data_root, name)
    settings = {**DEFAULT_SETTINGS, **BEST_SETTINGS[name], "scale": tuple(scale)}
    os.makedirs(os.path.join(output_folder, "apg_sweep"), exist_ok=True)
    file_name = f"{name}_s{''.join(str(x) for x in scale)}" + ("" if tag is None else f"_{tag}")
    output_path = os.path.join(output_folder, "apg_sweep", f"{file_name}.h5")
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            variants = [variant for variant in variants if variant not in f]
    if not variants:
        return
    models = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="apg", ndim=3)
    embedding_dir = tempfile.mkdtemp(dir=EMBEDDING_ROOT)
    try:
        for variant in variants:
            variant_settings = {**settings, "apg": parse_variant(variant)}
            embedding_path = os.path.join(embedding_dir, "embeddings.zarr")
            segmentation, runtime = segment(raw, "apg", variant_settings, models=models, embedding_path=embedding_path)
            results = [(variant, segmentation)]
            if settings["clean"]:
                cleaned = clean_spheroid_segmentation(segmentation, raw)
                results = [(variant, cleaned), (f"{variant}_noclean", segmentation)]
            for key, result in results:
                scores = evaluate(result, labels)
                print(f"{name} scale {scale} APG {key}: {format_scores(scores)}, {runtime:.1f} s", flush=True)
                with h5py.File(output_path, "a") as f:
                    f.create_dataset(key, data=result, compression="gzip")
                    f[key].attrs.update({**scores, "time": runtime, "settings": str(variant_settings)})
    finally:
        shutil.rmtree(embedding_dir, ignore_errors=True)


def run_cellpose(data_root, names, output_folder):
    """Segment and score the volumes with cellposeSAM in both modes, stored as 'cellposeSAM_<mode>'."""
    from cellpose import models

    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    os.makedirs(output_folder, exist_ok=True)
    results = []
    for name in names:
        raw, labels = load_data(data_root, name)
        for mode in CELLPOSE_MODES:
            start = time.time()
            stitch_threshold = CELLPOSE_STITCH_THRESHOLD if mode == "2d_stitch" else 0.0
            segmentation = model.eval(raw, z_axis=0, do_3D=mode == "3d", stitch_threshold=stitch_threshold)[0]
            segmentation = segmentation.astype("uint32")
            runtime = time.time() - start
            scores = evaluate(segmentation, labels)
            results.append({"name": name, "mode": mode, "time [s]": runtime, **scores})
            print(f"{name} cellposeSAM {mode}: {format_scores(scores)}, {runtime:.1f} s", flush=True)
            output_path = os.path.join(output_folder, f"{name}_{MODEL_TYPE}.h5")
            attrs = {**scores, "time": runtime}
            save_segmentation(output_path, raw, labels, f"cellposeSAM_{mode}", segmentation, attrs)
    print_table(results)


def export_predictions(data_root, names, output_folder):
    """Export the micro-sam2 AIS, both cellposeSAM modes and the better one by mSA as '_cellposeSAM'."""
    for name in names:
        with h5py.File(os.path.join(output_folder, f"{name}_{MODEL_TYPE}.h5"), "r") as f:
            modes = [f"cellposeSAM_{mode}" for mode in CELLPOSE_MODES]
            best = max(modes, key=lambda key: f[key].attrs["mSA"])
            for key, suffix in [("ais", ""), (best, "_cellposeSAM")] + [(key, f"_{key}") for key in modes]:
                write_prediction(os.path.join(data_root, f"{name}_prediction{suffix}.tif"), f[key][:])


def view(name, output_folder):
    import napari

    viewer = napari.Viewer(title=name)
    with h5py.File(os.path.join(output_folder, f"{name}_{MODEL_TYPE}.h5"), "r") as f:
        viewer.add_image(f["raw"][:], name="raw")
        for key in f:
            if key != "raw":
                viewer.add_labels(f[key][:], name=key)
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default=DATA_ROOT, help="The folder with the volumes.")
    parser.add_argument("-o", "--output_folder", default=OUTPUT_ROOT, help="The folder for the h5 files.")
    parser.add_argument("-n", "--names", nargs="+", choices=NAMES, default=list(NAMES))
    parser.add_argument("-e", "--engines", nargs="+", choices=ENGINES, default=None, help="By default AIS.")
    parser.add_argument("--best", action="store_true", help="Use the best AIS settings of each volume.")
    parser.add_argument("--clean", action="store_true", help="Use 'clean_spheroid_segmentation' instead of min_size.")
    parser.add_argument("--cellpose", action="store_true", help="Segment the volumes with cellposeSAM in both modes.")
    parser.add_argument("--sweep", action="store_true", help="Run the APG variants on the first volume.")
    parser.add_argument("--scale", nargs=3, type=int, default=None, help="The scale (z, y, x), overrides the settings.")
    parser.add_argument("--variants", nargs="+", default=list(APG_VARIANTS), help="The variants of '--sweep'.")
    parser.add_argument("--tag", default=None, help="Appended to the file name of '--sweep', for parallel jobs.")
    parser.add_argument("--export", action="store_true", help="Export the segmentations as tifs to the data folder.")
    parser.add_argument("--view", action="store_true", help="Show the first volume and its segmentations in napari.")
    args = parser.parse_args()

    if args.sweep:
        scale = args.scale or BEST_SETTINGS[args.names[0]].get("scale", (1, 1, 1))
        sweep(args.input_folder, args.names[0], scale, args.variants, args.output_folder, args.tag)
    elif args.cellpose:
        run_cellpose(args.input_folder, args.names, args.output_folder)
    elif args.export:
        export_predictions(args.input_folder, args.names, args.output_folder)
    elif args.view:
        view(args.names[0], args.output_folder)
    else:
        run(args.input_folder, args.names, args.engines, args.best, args.output_folder, args.clean, args.scale)


if __name__ == "__main__":
    main()
