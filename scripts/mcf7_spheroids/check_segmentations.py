"""Show the MCF7 spheroid stacks and their micro-sam2 segmentations in napari.

Usage:
    python check_segmentations.py -s dapi_ki67 -n s10  # the nuclei of one stack of 'dapi_ki67'
    python check_segmentations.py -s dapi_ecadh_btub -n Composite_s21  # the nuclei and the cells of one composite
"""

import os
import argparse

import imageio.v3 as imageio

import napari


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/mcf7_spheroids"
# The voxel size (z, y, x) in micrometer.
SCALE = (1.5, 0.414, 0.414)


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


def show_stack(data_root, subset, name, channel_names, prediction_names):
    """Load a stack and its micro-sam2 segmentations, and show them in napari."""
    raw, segmentations = load_stack(data_root, subset, name, prediction_names)
    assert raw.shape[1] == len(channel_names), f"{name}: expected {len(channel_names)} channels, got {raw.shape}."
    print(f"{subset} {name}: stack {raw.shape}, " + ", ".join(
        f"{key} {segmentation.max()} objects" for key, segmentation in segmentations.items()
    ))

    viewer = napari.Viewer(title=f"{subset} {name}")
    for channel, channel_name in enumerate(channel_names):
        viewer.add_image(raw[:, channel], name=channel_name, scale=SCALE, blending="additive", visible=channel == 0)
    for prediction_name, segmentation in segmentations.items():
        viewer.add_labels(segmentation, name=f"{prediction_name}_prediction", scale=SCALE)
    napari.run()


def check_dapi_ki67(data_root, name):
    """Show a stack of 'dapi_ki67' and its nucleus segmentation."""
    show_stack(data_root, "dapi_ki67", name, channel_names=("DAPI", "Ki67"), prediction_names=("nucleus",))


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
    args = parser.parse_args()

    if args.subset == "dapi_ki67":
        check_dapi_ki67(args.input_folder, args.name)
    else:
        check_dapi_ecadh_btub(args.input_folder, args.name)


if __name__ == "__main__":
    main()
