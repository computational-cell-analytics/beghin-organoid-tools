"""View nucleus segmentations and spot detections in napari, one sample at a time.

Files from the different folders are matched by a sample key that is extracted from the file name.
By default, the key is the number in a token like "s12" (e.g. "s12_nucleus_filtered.tif").
If a file name does not contain such a token, the first number in the file name is used.
"""

import argparse
import os
import re
from glob import glob

import napari
import pandas as pd
import tifffile
from magicgui.widgets import Container, Label, PushButton

DEFAULT_KEY_PATTERN = r"(?<![a-z0-9])s0*(\d+)(?![0-9])"
COORDINATE_COLUMNS = ["z", "y", "x"]
# The raw stacks are ZCYX, with DAPI in channel 0 and Ki67 in channel 1.
CHANNEL_AXIS = 1
CHANNEL_NAMES = ("DAPI", "Ki67")
CHANNEL_COLORMAPS = ("blue", "green")


def _extract_key(path, key_pattern):
    name = os.path.splitext(os.path.basename(path))[0]
    match = re.search(key_pattern, name, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"(\d+)", name)
    if match is None:
        raise ValueError(f"Cannot extract a sample key from the file name {name}.")
    key = match.group(1) if match.groups() else match.group(0)
    return int(key) if key.isdigit() else key


def _files_by_key(folder, extensions, key_pattern, exclude=None):
    paths = sorted(p for ext in extensions for p in glob(os.path.join(folder, f"*{ext}")))
    if exclude is not None:
        paths = [p for p in paths if exclude not in os.path.basename(p)]
    if not paths:
        raise ValueError(f"No files with extension {extensions} in {folder}.")
    files = {}
    for path in paths:
        key = _extract_key(path, key_pattern)
        if key in files:
            raise ValueError(f"The files {files[key]} and {path} have the same sample key {key}.")
        files[key] = path
    return files


def _sort_key(key):
    return (0, key, "") if isinstance(key, int) else (1, 0, key)


def match_files(seg_folder, spot_folder, raw_folder, key_pattern):
    """Return a list of (key, seg_path, spot_path, raw_path) for all samples with a file in each folder."""
    file_sets = {
        "segmentation": _files_by_key(seg_folder, [".tif", ".tiff"], key_pattern),
        "spots": _files_by_key(spot_folder, [".csv"], key_pattern),
    }
    if raw_folder is not None:
        # The raw folder also contains the micro-sam2 segmentations ("<stack>_nucleus_prediction.tif").
        file_sets["raw"] = _files_by_key(raw_folder, [".tif", ".tiff"], key_pattern, exclude="_prediction")

    common_keys = set.intersection(*(set(files) for files in file_sets.values()))
    for name, files in file_sets.items():
        unmatched = sorted(set(files) - common_keys, key=_sort_key)
        if unmatched:
            print(f"WARNING: Skip {name} files without a match: {[os.path.basename(files[k]) for k in unmatched]}")
    if not common_keys:
        raise ValueError("No sample has a file in each folder.")

    samples = []
    for key in sorted(common_keys, key=_sort_key):
        raw_path = file_sets["raw"][key] if raw_folder is not None else None
        samples.append((key, file_sets["segmentation"][key], file_sets["spots"][key], raw_path))
    return samples


def load_spots(path):
    spots = pd.read_csv(path)
    coordinates = spots[COORDINATE_COLUMNS].to_numpy()
    features = spots.drop(columns=COORDINATE_COLUMNS)
    return coordinates, features


class SampleViewer:
    def __init__(self, samples, spot_size):
        self.samples = samples
        self.spot_size = spot_size
        self.index = 0
        self.viewer = napari.Viewer()
        self.raw_layers = None
        self.seg_layer = None
        self.spot_layer = None

        self.label = Label()
        previous_button = PushButton(text="Previous")
        next_button = PushButton(text="Next")
        previous_button.changed.connect(lambda: self.show(self.index - 1))
        next_button.changed.connect(lambda: self.show(self.index + 1))
        widget = Container(widgets=[self.label, previous_button, next_button])
        self.viewer.window.add_dock_widget(widget, name="Samples", area="right")

        self.show(0)

    def show(self, index):
        if not 0 <= index < len(self.samples):
            return
        self.index = index
        key, seg_path, spot_path, raw_path = self.samples[index]

        segmentation = tifffile.imread(seg_path)
        coordinates, features = load_spots(spot_path)
        raw = None if raw_path is None else tifffile.imread(raw_path)
        if raw is not None and raw.shape[CHANNEL_AXIS] != len(CHANNEL_NAMES):
            raise ValueError(
                f"Expected {len(CHANNEL_NAMES)} channels in axis {CHANNEL_AXIS} of {raw_path}, got shape {raw.shape}."
            )

        if self.seg_layer is None:
            if raw is not None:
                self.raw_layers = self.viewer.add_image(
                    raw, channel_axis=CHANNEL_AXIS, name=list(CHANNEL_NAMES), colormap=list(CHANNEL_COLORMAPS),
                    blending="additive",
                )
            self.seg_layer = self.viewer.add_labels(segmentation, name="nuclei")
            self.spot_layer = self.viewer.add_points(
                coordinates, features=features, size=self.spot_size, face_color="red", border_width=0, name="spots",
            )
        else:
            shape_changed = segmentation.shape != self.seg_layer.data.shape
            if raw is not None:
                for channel, layer in enumerate(self.raw_layers):
                    layer.data = raw.take(channel, axis=CHANNEL_AXIS)
                    layer.reset_contrast_limits()
            self.seg_layer.data = segmentation
            self.spot_layer.data = coordinates
            self.spot_layer.features = features
            if shape_changed:
                self.viewer.reset_view()

        names = [os.path.basename(p) for p in (seg_path, spot_path, raw_path) if p is not None]
        self.label.value = f"Sample {key} ({index + 1} / {len(self.samples)})\n" + "\n".join(names)
        self.viewer.title = f"Sample {key}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-s", "--segmentation_folder", default="mcf7_nuclei_filtered",
        help="Folder with the nucleus segmentations as tif files.",
    )
    parser.add_argument(
        "-p", "--spot_folder", default="mcf7_ki67_spots",
        help="Folder with the spot detections as csv files with z, y, x columns in pixel coordinates.",
    )
    parser.add_argument(
        "-r", "--raw_folder",
        help="Optional folder with the raw data as ZCYX tif files (DAPI, Ki67). Files with '_prediction' are skipped.",
    )
    parser.add_argument("--spot_size", type=float, default=5, help="Display size of the spots in pixels.")
    parser.add_argument(
        "--key_pattern", default=DEFAULT_KEY_PATTERN,
        help="Regular expression that extracts the sample key from a file name. The first group is the key.",
    )
    args = parser.parse_args()

    samples = match_files(args.segmentation_folder, args.spot_folder, args.raw_folder, args.key_pattern)
    print(f"Found {len(samples)} samples.")
    SampleViewer(samples, args.spot_size)
    napari.run()


if __name__ == "__main__":
    main()
