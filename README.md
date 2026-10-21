# beghin-organoid-tools

Scripts for Anne Beghin group's data.

## Installation

```bash
git clone https://github.com/computational-cell-analytics/beghin-organoid-tools.git
cd beghin-organoid-tools
conda env create -f environment.yaml
conda activate orgatools
pip install -e .
```

Check that it works:

```bash
python -c "import orgatools; print(orgatools.__version__)"
```

## Converting tif stacks to OME-Zarr

```bash
python scripts/convert_to_ome_zarr.py -i /path/to/tif/folder -o /path/to/output/folder
```

Every tif in the folder (and its subfolders) becomes a `.ome.zarr` folder with the same name: lossless, about 2.5 - 3x
smaller than the uncompressed tif, with the voxel size from the ImageJ metadata. The script checks every converted
file against its tif.

To send the results, put the `.ome.zarr` folders in one archive, e.g. `tar -cf data.tar *.ome.zarr`. They are compressed
already, so the archive does not need compression.
