import sys

filename = sys.argv[1]


def analyze_density(filename, threshold=1e-15):
    survivors = 0
    total = 0

    print("Scanning file... ")

    with open(filename, "r") as f:
        for line in f:
            if not line.startswith("%"):
                break

        rows, cols = map(int, line.split())

        for line in f:
            parts = line.split()
            if len(parts) == 1:
                val = float(parts[0])
            else:
                val = complex(float(parts[0]), float(parts[1]))

            if abs(val) >= threshold:
                survivors += 1
            total += 1

            # Print progress every 10 million lines so you know it isn't frozen
            if total % 10_000_000 == 0:
                print(f"Scanned {total} lines...")

    density = (survivors / total) * 100

    print("\n--- RESULTS ---")
    print(f"Total elements: {total}")
    print(f"Elements above threshold: {survivors}")
    print(f"Matrix Density: {density:.2f}%")


# Run the diagnostic
analyze_density(filename, threshold=1e-15)
