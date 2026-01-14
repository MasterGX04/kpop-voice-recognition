import json
import sys
from pathlib import Path

def remove_jhope_backing(input_path: str):
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    cleaned_labels = []
    removed_count = 0

    for entry in labels:
        # Expected format: [member, start, end, backing, adlib]
        member = entry[0]
        backing = entry[3]

        if member == "J-Hope" and backing is True:
            removed_count += 1
            continue

        cleaned_labels.append(entry)

    output_path = input_path.with_name(input_path.stem + "_no_jhope_backing.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_labels, f, indent=2)

    print(f"Done.")
    print(f"Removed {removed_count} J-Hope backing entries.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remove_jhope_backing.py <labels.json>")
        sys.exit(1)

    remove_jhope_backing(sys.argv[1])