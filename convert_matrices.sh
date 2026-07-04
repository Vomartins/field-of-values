#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: You must provide a model name."
    echo "Usage: ./convert_matrices.sh <model_name>"
    exit 1
fi

MODEL_NAME=$1

echo "Converting matrices for model: $MODEL_NAME"

TARGET_DIR="matrices/$MODEL_NAME"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Matrix directory does not exist: $TARGET_DIR"
    exit 1
fi

for filepath in "$TARGET_DIR"/*.mtx; do
    echo "Converting: $filepath"

    if [ ! -e "$filepath" ]; then
        echo "No matrices found in $TARGET_DIR!"
        break
    fi

    clean_name=$(basename "$filepath" .mtx)

    python3 convert_bin.py "$clean_name" "$MODEL_NAME"

    echo ">>> Finished converting $filepath to binary format"
    echo "------------------------------------------------"
done

echo ">>> All matrices converted for model: $MODEL_NAME"
