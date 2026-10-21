# beghin-organoid-tools

Tools for Anne Beghin group's data.

## Installation

Requires Python 3.11 or newer. In any Python environment (conda, venv, ...):

```bash
pip install git+https://github.com/computational-cell-analytics/beghin-organoid-tools.git
```

## Converting tif to OME-Zarr

```bash
orgatools data convert -i /path/to/stack.tif -o /path/to/output/folder  # a single tif file
orgatools data convert -i /path/to/tif/folder -o /path/to/output/folder  # every tif in a folder and its subfolders
```

Each tif becomes a lossless `.ome.zarr` folder, about 2.5 - 3x smaller than the uncompressed tif. To send the results,
put them in one archive: `tar -cf data.tar *.ome.zarr`.
