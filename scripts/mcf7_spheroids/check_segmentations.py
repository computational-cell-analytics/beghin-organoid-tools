"""Show the MCF7 spheroid stacks and their micro-sam2 segmentations in napari.

Usage:
    python check_segmentations.py -s dapi_ki67 -n s10  # the nuclei of one stack of 'dapi_ki67'
    python check_segmentations.py -s dapi_ecadh_btub -n Composite_s21  # the nuclei and the cells of one composite
    python check_segmentations.py -s dapi_ki67 -n s10 --spots /path/to/spots  # and the Ki67 spots of every method
    python check_segmentations.py -s dapi_ki67 -n s10 --spots /path/to/spots --features /path/to/features  # and foci
"""

import os
import argparse
from glob import glob

import imageio.v3 as imageio
import numpy as np
import pandas as pd

import napari


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/mcf7_spheroids"
# The voxel size (z, y, x) in micrometer.
SCALE = (1.5, 0.414, 0.414)
# The half depth in micrometer (3 slices) around the current z-slice in which the spots are shown, shrinking to zero
# at its edge, so a spot is shown in the +-2 slices of its extent in z.
SPOT_DEPTH = 4.5
# The colors of the spots of each method, to compare them in the same view.
SPOT_COLORS = ("cyan", "magenta", "yellow", "lime", "orange")


def load_stack(data_root, subset, name, prediction_names):
    """Load a stack (Z, C, Y, X) and its micro-sam2 segmentations '<name>_<prediction>_prediction.tif'.

    Returns the stack and a dictionary that maps each prediction name to its segmentation (Z, Y, X).
    """
    raw = imageio.imread(os.path.join(data_root, subset, f"{name}.tif"))
    segmentations = {
        prediction_name: imageio.imread(os.path.join(data_root, subset, f"{name}_{prediction_name}_prediction.tif"))
        for prediction_name in prediction_names
    }
    for prediction_name, segmentation in segmentations.items():
        assert segmentation.shape == raw[:, 0].shape, f"{name}: the {prediction_name} shape does not match the stack."
    return raw, segmentations


def load_spots(spot_folder, name):
    """Load the spot tables '<method>/<name>_ki67_spots.csv' of detect_ki67_spots.py for every method."""
    paths = sorted(glob(os.path.join(spot_folder, "*", f"{name}_ki67_spots.csv")))
    assert paths, f"{name}: no spot tables in {spot_folder}."
    return {os.path.basename(os.path.dirname(path)): pd.read_csv(path) for path in paths}


def load_filtered_nuclei(spot_folder, name):
    """Load the nucleus segmentation without the noise nuclei of detect_ki67_spots.py, if it exists."""
    path = os.path.join(spot_folder, "nuclei", f"{name}_nucleus_filtered.tif")
    return imageio.imread(path) if os.path.exists(path) else None


def load_feature_masks(feature_folder, name):
    """Load the spheroid mask and the foci labels of compute_ki67_features.py."""
    return {
        "spheroid": imageio.imread(os.path.join(feature_folder, "spheroid", f"{name}_spheroid.tif")),
        "ki67_foci": imageio.imread(os.path.join(feature_folder, "foci", f"{name}_ki67_foci.tif")),
    }


def show_stack(data_root, subset, name, channel_names, prediction_names, spots=None, filtered_nuclei=None,
               feature_masks=None):
    """Load a stack and its micro-sam2 segmentations, and show them in napari, with the spots of each method and
    the filtered nuclei they were detected in instead of the micro-sam2 nuclei (still available as hidden layer),
    and the spheroid mask and the foci."""
    raw, segmentations = load_stack(data_root, subset, name, prediction_names)
    assert raw.shape[1] == len(channel_names), f"{name}: expected {len(channel_names)} channels, got {raw.shape}."
    print(f"{subset} {name}: stack {raw.shape}, " + ", ".join(
        f"{key} {segmentation.max()} objects" for key, segmentation in segmentations.items()
    ))

    viewer = napari.Viewer(title=f"{subset} {name}")
    for channel, channel_name in enumerate(channel_names):
        viewer.add_image(
            raw[:, channel], name=channel_name, scale=SCALE, blending="additive", visible=channel == 0,
            projection_mode="none",  # Show the current z-slice, not the mean over the slice depth of the spots.
        )
    for prediction_name, segmentation in segmentations.items():
        viewer.add_labels(segmentation, name=f"{prediction_name}_prediction", scale=SCALE,
                          visible=filtered_nuclei is None)
    if filtered_nuclei is not None:
        n_removed = len(set(np.unique(segmentations["nucleus"])) - set(np.unique(filtered_nuclei)))
        print(f"nucleus_filtered: {n_removed} noise nuclei removed")
        viewer.add_labels(filtered_nuclei, name="nucleus_filtered", scale=SCALE)
    for mask_name, mask in (feature_masks or {}).items():
        viewer.add_labels(mask, name=mask_name, scale=SCALE, visible=mask_name != "spheroid")
    for (method, table), color in zip((spots or {}).items(), SPOT_COLORS):
        print(f"{method}: {len(table)} spots in {table.nucleus_id.nunique()} nuclei")
        coordinates = ["z", "y", "x"]
        viewer.add_points(
            table[coordinates].to_numpy(), name=f"{method}_spots", scale=SCALE,
            features=table.drop(columns=coordinates),
            size=table.diameter_um.to_numpy() / SCALE[1], face_color="transparent", border_color=color,
            border_width=0.1, visible=False, projection_mode="rescale_linear",
        )
    if spots:
        # Show the spots within SPOT_DEPTH of the current z-slice, shrinking with their distance to it.
        viewer.dims.margin_left = viewer.dims.margin_right = (SPOT_DEPTH, 0.0, 0.0)
    napari.run()


def check_dapi_ki67(data_root, name, spot_folder=None, feature_folder=None):
    """Show a stack of 'dapi_ki67', its nucleus segmentation and optionally the Ki67 spots and the feature masks."""
    spots = None if spot_folder is None else load_spots(spot_folder, name)
    filtered_nuclei = None if spot_folder is None else load_filtered_nuclei(spot_folder, name)
    feature_masks = None if feature_folder is None else load_feature_masks(feature_folder, name)
    show_stack(
        data_root, "dapi_ki67", name, channel_names=("DAPI", "Ki67"), prediction_names=("nucleus",), spots=spots,
        filtered_nuclei=filtered_nuclei, feature_masks=feature_masks,
    )


def check_dapi_ecadh_btub(data_root, name):
    """Show a stack of 'dapi_ecadh_btub' and its nucleus and membrane segmentations."""
    show_stack(
        data_root, "dapi_ecadh_btub", name, channel_names=("DAPI", "E-cadherin", "beta-tubulin"),
        prediction_names=("nucleus", "membrane"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default=DATA_ROOT, help="The folder with the spheroid sets.")
    parser.add_argument("-s", "--subset", choices=("dapi_ki67", "dapi_ecadh_btub"), required=True)
    parser.add_argument("-n", "--name", required=True, help="The name of the stack, such as 's10' or 'Composite_s21'.")
    parser.add_argument("--spots", help="The output folder of detect_ki67_spots.py, only for 'dapi_ki67'.")
    parser.add_argument("--features", help="The output folder of compute_ki67_features.py, only for 'dapi_ki67'.")
    args = parser.parse_args()

    if args.subset == "dapi_ki67":
        check_dapi_ki67(args.input_folder, args.name, args.spots, args.features)
    else:
        check_dapi_ecadh_btub(args.input_folder, args.name)


if __name__ == "__main__":
    main()
