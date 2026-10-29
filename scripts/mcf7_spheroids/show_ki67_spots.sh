#!/bin/bash
# Show dapi_ki67 stacks, their nuclei and the Ki67 spots of every method in napari, one stack after the other:
# closing the napari window opens the next stack, Ctrl+C in the terminal stops after the current one.
# Usage: ./show_ki67_spots.sh [stack ...]  (default: all stacks, s2 to s40)

STACKS=${@:-$(seq -f "s%g" 2 40)}
SPOT_FOLDER=/mnt/vast-nhr/projects/cidas/cca/experiments/beghin_organoids/mcf7_ki67_spots

# No 'set -e': napari can exit with an error code when its window is closed, which must not stop the loop.
trap "echo 'Stopped.'; exit 130" INT

for STACK in $STACKS; do
    echo "Showing $STACK"
    python "$(dirname "$0")/check_segmentations.py" -s dapi_ki67 -n "$STACK" --spots "$SPOT_FOLDER"
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "napari exited with code $STATUS for $STACK, continuing."
    fi
done
