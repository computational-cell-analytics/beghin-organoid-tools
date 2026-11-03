#!/bin/bash
# View the filtered nuclei, the dog Ki67 spots and the DAPI and Ki67 channels of the dapi_ki67 stacks in napari,
# with buttons to go through the stacks (s2 to s40).
# Usage: ./view_nuclei_and_spots.sh [further arguments of view_nuclei_and_spots.py, e.g. --spot_size 3]

SPOT_FOLDER=/mnt/vast-nhr/projects/cidas/cca/experiments/beghin_organoids/mcf7_ki67_spots
RAW_FOLDER=/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/mcf7_spheroids/dapi_ki67

python "$(dirname "$0")/view_nuclei_and_spots.py" \
    -s "$SPOT_FOLDER/nuclei" \
    -p "$SPOT_FOLDER/dog" \
    -r "$RAW_FOLDER" \
    "$@"
