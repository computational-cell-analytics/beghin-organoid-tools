# MCF7 spheroids

Segmentation of the MCF7 spheroids and detection of the Ki67 spots in the nuclei of `dapi_ki67` (39 stacks, s2 - s40).
The usage is in the docstring of each script.

- `segment_mcf7_spheroids.py`: micro-sam2 nucleus (DAPI) and cell segmentation.
- `detect_ki67_spots.py`: Ki67 spot detection in the nuclei, one table per stack and method.
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
