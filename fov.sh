#!/bin/bash


if [ -z "$1" ]; then
    echo "Error: You must provide a variant."
    echo "Usage: ./fov.sh <variant> <model_name> <angles>"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: You must provide a model name."
    echo "Usage: ./fov.sh <variant> <model_name> <angles>"
    exit 1
fi

if [ -z "$3" ]; then
    echo "Error: You must provide angles."
    echo "Usage: ./fov.sh <variant> <model_name> <angles>"
    exit 1
fi

VARIANT=$1
MODEL_NAME=$2
ANGLES=$3
TARGET_DIR="matrices/$MODEL_NAME"
SAVE_DIR="figures/$MODEL_NAME"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

if [ ! -d "$SAVE_DIR" ]; then
    mkdir -p "$SAVE_DIR"
fi

echo "Starting batch processing for model: $MODEL_NAME"
echo "------------------------------------------------"

if [ "$VARIANT" = "scipy" ]; then
    for filepath in "$TARGET_DIR"/*.dat; do

        # Safety check: if the folder is empty, the loop will literally return "*.dat"
        # This prevents the script from trying to process a file named "*.dat"
        if [ ! -e "$filepath" ]; then
            echo "No matrices found in $TARGET_DIR!"
            break
        fi

        echo ">>> Executing: python scipy_fov.py --model=$MODEL_NAME --angles=$ANGLES $filepath"
        echo "------------------------------------------------"

        # 4. Run your Python script!
        python3 scipy_fov.py --model="$MODEL_NAME" --angles="$ANGLES" "$filepath"

        echo ">>> Finished processing $filepath"
        echo "------------------------------------------------"
    done
elif [ "$VARIANT" = "numpy" ]; then
    for filepath in "$TARGET_DIR"/*.mtx; do
        if [ ! -e "$filepath" ]; then
            echo "No matrices found in $TARGET_DIR!"
            break
        fi

        echo ">>> Executing: python numpy_fov.py --model=$MODEL_NAME --angles=$ANGLES $filepath"
        echo "------------------------------------------------"

        # 4. Run your Python script!
        python3 numpy_fov.py --model="$MODEL_NAME" --angles="$ANGLES" "$filepath"

        echo ">>> Finished processing $filepath"
        echo "------------------------------------------------"
    done
fi

echo "All matrices for $MODEL_NAME have been processed!"
