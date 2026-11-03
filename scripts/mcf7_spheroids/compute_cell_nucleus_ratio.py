"""The volume ratio of the cells to their nuclei in the MCF7 spheroids with membrane stain (dapi_ecadh_btub).

Uses the micro-sam2 segmentations of segment_mcf7_spheroids.py: the cells ('<stack>_membrane_prediction.tif') and the
nuclei ('<stack>_nucleus_prediction.tif'). Every nucleus is assigned to the cell that contains most of its voxels.
The nucleus volume of a cell is the full volume of all nuclei assigned to it, so a nucleus split into fragments, two
cells merged together with their nuclei and binucleated cells all give a sensible ratio; n_nuclei flags these cases.

The output is '<output>/<stack>_cells.csv', one row per cell, the key is (stack, cell_id). The columns are:
    cell_id, z, y, x (centroid in voxels), z_um, y_um, x_um, cell_volume_um3, n_nuclei, nucleus_ids (';' separated),
    nucleus_volume_um3, cell_to_nucleus_ratio (NaN without nucleus), nucleus_in_cell_fraction (the fraction of the
    nucleus voxels in the cell), touches_z_border, touches_xy_border, nucleus_touches_z_border, valid.
A cell is valid if it has exactly one nucleus, at least MIN_NUCLEUS_IN_CELL of the nucleus is in the cell and neither
the cell nor the nucleus touches the stack border. s21 and s23 are cut at the first slice (the spheroid is widest
there), so many of their cells are truncated and not valid.
Read all stacks with: pd.concat(map(pd.read_csv, glob("<output>/*_cells.csv")), ignore_index=True).

The plot '<output>/plots/cell_to_nucleus_ratio.png' shows the distribution of the ratio of the valid cells, one panel
per stack and one for all stacks combined, from all tables in '<output>'.

Usage:
    python compute_cell_nucleus_ratio.py  # all stacks
    python compute_cell_nucleus_ratio.py -n Composite_s21  # one stack
"""

import os
import argparse
from glob import glob

import tifffile
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage

from detect_ki67_spots import DATA_ROOT, SCALE
from plot_ki67_spots_vs_nucleus_size import DOT_COLOR, LINE_COLOR, TEXT_COLOR, style_axis

matplotlib.use("Agg")

OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/beghin_organoids/mcf7_cell_nucleus_ratio"
SUBSET = "dapi_ecadh_btub"
VOXEL_VOLUME = float(np.prod(SCALE))
MIN_NUCLEUS_IN_CELL = 0.9
BIN_WIDTH = 0.2


def assign_nuclei(cells, nuclei):
    """The cell with the most voxels of each nucleus, the fraction of the nucleus in it and the nucleus volume."""
    rows = []
    for nucleus_id, bbox in enumerate(ndimage.find_objects(nuclei), 1):
        if bbox is None:
            continue
        mask = nuclei[bbox] == nucleus_id
        overlap = np.bincount(cells[bbox][mask])
        overlap[0] = 0
        touches_z = bbox[0].start == 0 or bbox[0].stop == nuclei.shape[0]
        rows.append({
            "nucleus_id": nucleus_id, "cell_id": int(overlap.argmax()), "in_cell": overlap.max() / mask.sum(),
            "volume_um3": mask.sum() * VOXEL_VOLUME, "touches_z_border": touches_z,
        })
    return pd.DataFrame(rows)


def run_stack(name, input_folder, output_folder):
    cells = tifffile.imread(os.path.join(input_folder, SUBSET, f"{name}_membrane_prediction.tif"))
    nuclei = tifffile.imread(os.path.join(input_folder, SUBSET, f"{name}_nucleus_prediction.tif"))
    assert cells.shape == nuclei.shape, f"{name}: the cell and nucleus shapes do not match."
    assigned = assign_nuclei(cells, nuclei)
    assigned = assigned[assigned.cell_id > 0]
    nuclei_per_cell = dict(list(assigned.groupby("cell_id")))

    rows = []
    for cell_id, bbox in enumerate(ndimage.find_objects(cells), 1):
        if bbox is None:
            continue
        coords = np.argwhere(cells[bbox] == cell_id) + [b.start for b in bbox]
        cell_nuclei = nuclei_per_cell.get(cell_id, assigned.iloc[:0])
        cell_volume, nucleus_volume = len(coords) * VOXEL_VOLUME, cell_nuclei.volume_um3.sum()
        touches_xy = any(b.start == 0 or b.stop == n for b, n in zip(bbox[1:], cells.shape[1:]))
        rows.append({
            "cell_id": cell_id, **dict(zip("zyx", coords.mean(axis=0))),
            **{f"{axis}_um": value for axis, value in zip("zyx", coords.mean(axis=0) * SCALE)},
            "cell_volume_um3": cell_volume, "n_nuclei": len(cell_nuclei),
            "nucleus_ids": ";".join(map(str, cell_nuclei.nucleus_id)), "nucleus_volume_um3": nucleus_volume,
            "cell_to_nucleus_ratio": cell_volume / nucleus_volume if len(cell_nuclei) else np.nan,
            "nucleus_in_cell_fraction": cell_nuclei.in_cell.min() if len(cell_nuclei) else np.nan,
            "touches_z_border": bbox[0].start == 0 or bbox[0].stop == cells.shape[0],
            "touches_xy_border": touches_xy, "nucleus_touches_z_border": bool(cell_nuclei.touches_z_border.any()),
        })
    table = pd.DataFrame(rows)
    table["valid"] = (
        (table.n_nuclei == 1) & (table.nucleus_in_cell_fraction >= MIN_NUCLEUS_IN_CELL) & ~table.touches_z_border
        & ~table.touches_xy_border & ~table.nucleus_touches_z_border
    )
    table.insert(0, "stack", name)

    os.makedirs(output_folder, exist_ok=True)
    table.round(4).to_csv(os.path.join(output_folder, f"{name}_cells.csv"), index=False)
    valid = table[table.valid]
    print(f"{name}: {len(table)} cells, {len(assigned)} nuclei in cells ({len(np.unique(nuclei)) - 1} total), "
          f"{len(valid)} valid, ratio median {valid.cell_to_nucleus_ratio.median():.2f} "
          f"(IQR {valid.cell_to_nucleus_ratio.quantile(0.25):.2f} - {valid.cell_to_nucleus_ratio.quantile(0.75):.2f})",
          flush=True)


def plot_ratio(output_folder):
    """Histograms of the ratio of the valid cells, per stack and combined, with shared bins."""
    tables = pd.concat(map(pd.read_csv, sorted(glob(os.path.join(output_folder, "*_cells.csv")))), ignore_index=True)
    valid = tables[tables.valid]
    groups = [(stack, valid[valid["stack"] == stack]) for stack in valid["stack"].unique()]
    groups.append((f"All {len(groups)} stacks", valid))
    bins = np.arange(np.floor(valid.cell_to_nucleus_ratio.min()), valid.cell_to_nucleus_ratio.max() + BIN_WIDTH,
                     BIN_WIDTH)

    n_cols = 2
    n_rows = int(np.ceil(len(groups) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.8 * n_rows), sharex=True, layout="constrained",
                             squeeze=False)
    for ax, (title, group) in zip(axes.flat, groups):
        ratio = group.cell_to_nucleus_ratio
        ax.hist(ratio, bins=bins, color=DOT_COLOR, edgecolor="white", linewidth=1)
        ax.axvline(ratio.median(), color=LINE_COLOR, linewidth=1.5, linestyle="--")
        ax.set_title(f"{title}: n = {len(ratio)}, median {ratio.median():.2f} "
                     f"(IQR {ratio.quantile(0.25):.2f} - {ratio.quantile(0.75):.2f})", fontsize=9, loc="left")
        style_axis(ax)
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    fig.supxlabel("Cell volume / nucleus volume (valid cells, dashed line: median)", color=TEXT_COLOR, fontsize=10)
    fig.supylabel("Number of cells", color=TEXT_COLOR, fontsize=10)
    os.makedirs(os.path.join(output_folder, "plots"), exist_ok=True)
    path = os.path.join(output_folder, "plots", "cell_to_nucleus_ratio.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Plot of {len(valid)} valid cells in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default=DATA_ROOT, help="The folder with the spheroid sets.")
    parser.add_argument("-o", "--output_folder", default=OUTPUT_ROOT, help="The folder for the cell tables.")
    parser.add_argument("-n", "--name", help="The name of a single stack, such as 'Composite_s21'.")
    args = parser.parse_args()

    if args.name is None:
        paths = sorted(glob(os.path.join(args.input_folder, SUBSET, "*_membrane_prediction.tif")))
        names = [os.path.basename(path).replace("_membrane_prediction.tif", "") for path in paths]
    else:
        names = [args.name]
    for name in names:
        run_stack(name, args.input_folder, args.output_folder)
    plot_ratio(args.output_folder)


if __name__ == "__main__":
    main()
