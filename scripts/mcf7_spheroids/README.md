# MCF7 spheroids

Segmentation of the MCF7 spheroids, detection of the Ki67 spots in the nuclei of `dapi_ki67` (39 stacks, s2 - s40) and
the cell to nucleus volume ratio of `dapi_ecadh_btub` (3 stacks). The usage is in the docstring of each script.

- `segment_mcf7_spheroids.py`: micro-sam2 nucleus (DAPI) and cell segmentation.
- `detect_ki67_spots.py`: Ki67 spot detection in the nuclei, one table per stack and method.
- `compute_ki67_features.py`: spheroid mask, Ki67 foci masks and the features of the spheroids, nuclei and foci.
- `plot_ki67_spots_vs_nucleus_size.py`: the number of Ki67 spots per nucleus against the nucleus volume, pooled and
  per spheroid.
- `compute_cell_nucleus_ratio.py`: the volume ratio of the cells to their nuclei in `dapi_ecadh_btub`, one table per
  stack and the histograms of the ratio.
- `check_segmentations.py`: napari viewer for the stacks, segmentations and spots.
- `show_ki67_spots.sh`: opens the stacks with their spots one after the other.

## Ki67 spots

**We use the `dog` spots for the further analysis.** The results are in
`/mnt/vast-nhr/projects/cidas/cca/experiments/beghin_organoids/mcf7_ki67_spots/dog/<stack>_ki67_spots.csv`.

Seven methods were compared without ground truth: skimage `blob_log`, `blob_dog`, a 3D determinant of Hessian,
white top-hat + `peak_local_max`, `h_maxima`, threshold + watershed, and spotiflow as a learned reference. The criteria
were the recall of synthetic spots, the false positives in the cytoplasm, the agreement between the methods and visual
checks. `tophat_peaks`, `log` and `dog` were close and best; after the visual inspection `dog` was chosen.

The detection, per nucleus:
- The noise nuclei of a speckled DAPI layer at the z-borders are removed first (high DAPI texture, dim, near the
  first or last slice). The filtered segmentation is saved in `.../mcf7_ki67_spots/nuclei/`, the micro-sam2
  segmentation is not changed.
- The Ki67 channel is normalized to the median and MAD of the nucleus, then the spots are detected.
- A spot is kept if it is at least 3 px inside the nucleus, has a contrast of at least 100 counts above its local
  background and a peak of at least 450 counts. Spill-over from neighbouring nuclei and split foci are removed.

The main settings are `MIN_PEAK`, `MIN_CONTRAST` and `EROSION` in `detect_ki67_spots.py`.

## Features

`compute_ki67_features.py` computes the features on three levels from the filtered nuclei and the `dog` spots, in
`/mnt/vast-nhr/projects/cidas/cca/experiments/beghin_organoids/mcf7_ki67_features`:

```
spheroid/<stack>_spheroid.tif  # spheroid mask: closing (10 um) of the nuclei, holes filled
foci/<stack>_ki67_foci.tif     # foci labels, grown from the dog spots to half their contrast; label = focus_id
tables/<stack>_spheroid.csv    # one row: size, shape, z-range, touches_z_border, counts
tables/<stack>_nuclei.csv      # one row per nucleus: shape, DAPI and Ki67 intensity, foci summary, spatial
                               # distribution of the foci, position in the spheroid
tables/<stack>_foci.csv        # one row per focus: size, shape, intensity, position in the nucleus
```

The tables join on `(stack, nucleus_id)`; read one level for all stacks with
`pd.concat(map(pd.read_csv, glob(".../tables/*_nuclei.csv")), ignore_index=True)`. The columns are described in the
docstring of the script. Note for the analysis: the signal drops 5 - 8x from the top to the bottom of a spheroid, so
the position in the spheroid is confounded with the depth (`depth_um`); some spheroids are cut at a z-border
(`touches_z_border`), and a few nuclei are outside of the spheroid mask (`in_spheroid`).

## Cell to nucleus volume ratio

`compute_cell_nucleus_ratio.py` assigns every nucleus to the cell that contains most of it and writes one row per cell
to `/mnt/vast-nhr/projects/cidas/cca/experiments/beghin_organoids/mcf7_cell_nucleus_ratio/<stack>_cells.csv`. The
nucleus volume of a cell is the sum of its nuclei. Use the `valid` cells for the analysis: exactly one nucleus, at least
90% of the nucleus in the cell, and neither touches the stack border. s21 and s23 are cut at the first slice, so many
of their cells are truncated. The histograms of the ratio of the valid cells, per stack and combined, are in
`.../mcf7_cell_nucleus_ratio/plots/cell_to_nucleus_ratio.png`.
