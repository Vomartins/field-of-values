import sys

import numpy as np

if len(sys.argv) != 3:
    print("Usage: python convert_bin.py <mtx_filename> <model_name>")
    sys.exit(1)

filename = sys.argv[1]
modelname = sys.argv[2]


def convert_mtx_to_memmap(txt_filename, bin_filename, rows, cols, dtype="float64"):
    print(f"Creating binary file on disk... \n {txt_filename} -> {bin_filename}")
    # This creates a massive file on your drive, but uses 0 RAM
    disk_matrix = np.memmap(bin_filename, dtype=dtype, mode="w+", shape=(rows, cols))

    chunk_size = 10_000_000
    chunk_data = []

    print("Parsing text and writing to disk in chunks...")
    with open(txt_filename, "r") as f:
        # Skip header
        for line in f:
            if not line.startswith("%"):
                break

        # We already know dimensions, so we can ignore this line
        idx = 0
        for line in f:
            parts = line.split()
            # If your file has complex numbers, change to: complex(float(parts[0]), float(parts[1]))
            val = float(parts[0])

            chunk_data.append(val)

            # When the chunk is full, write it to the disk
            if len(chunk_data) == chunk_size:
                # Calculate flat indices to 2D coordinates (Column-major order for MTX)
                start_idx = idx - chunk_size + 1
                end_idx = idx + 1

                # Reshape and slot the chunk into the disk array
                # (This logic assumes a flat column-major read)
                flat_array = np.array(chunk_data, dtype=dtype)

                # Write to disk piece by piece
                np.put(disk_matrix, range(start_idx, end_idx), flat_array)
                disk_matrix.flush()  # Force save to SSD

                chunk_data = []
                print(f"Processed {idx + 1} elements...")

            idx += 1

        # Write any remaining elements in the final incomplete chunk
        if chunk_data:
            start_idx = idx - len(chunk_data)
            end_idx = idx
            flat_array = np.array(chunk_data, dtype=dtype)
            np.put(disk_matrix, range(start_idx, end_idx), flat_array)
            disk_matrix.flush()

    print("Conversion complete! Your matrix is now on disk.")


# Run the converter (replace with your exact dimensions if they differ slightly)
convert_mtx_to_memmap(
    f"matrices/{modelname}/{filename}.mtx",
    f"matrices/{modelname}/{filename}.dat",
    rows=42346,
    cols=42346,
)
