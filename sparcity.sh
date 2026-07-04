#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: You must provide a model name."
    echo "Usage: ./convert_matrices.sh <model_name>"
    exit 1
fi

MODEL_NAME=$1

echo "Calculating sparcity of matrices for model: $MODEL_NAME"

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

    python3 diagnose_sparcity.py "$filepath"

    echo ">>> Finished calculating sparsity for $filepath"
    echo "------------------------------------------------"
done

echo ">>> All matrices sparity calculated for model: $MODEL_NAME"
