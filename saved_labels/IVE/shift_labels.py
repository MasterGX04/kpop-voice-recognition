import json
import argparse
import os
import sys

def shiftLabels(songName: str, shift: int, inDir: str = ".", outSuffix: str = "_shifted"):
    """
    Shift all label chunk indices by `shift`.

    Input file:  <songName>_labels.json
    Output file: <songName>_labels_shifted.json
    """
    inPath = os.path.join(inDir, f"{songName}_labels.json")
    if not os.path.exists(inPath):
        print(f"❌ Label file not found: {inPath}")
        sys.exit(1)

    with open(inPath, "r", encoding="utf-8") as f:
        labels = json.load(f)

    shifted = []
    for entry in labels:
        # Expect: [member, startChunk, endChunk, isBacking, isAdlib]
        member, start, end, isBacking, isAdlib = entry

        newStart = start + shift
        newEnd = end + shift

        if newStart < 0 or newEnd < 0:
            print(f"⚠️ Skipping negative chunk after shift: {entry}")
            continue

        shifted.append([member, newStart, newEnd, isBacking, isAdlib])

    outPath = os.path.join(inDir, f"{songName}_labels{outSuffix}.json")
    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(shifted, f, indent=2)

    print(f"✅ Shifted {len(shifted)} labels by {shift} chunks")
    print(f"📄 Output written to: {outPath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shift label chunk indices by a fixed amount.")
    parser.add_argument("songName", type=str, help="Song name (e.g. Attitude)")
    parser.add_argument("shift", type=int, help="Chunk shift amount (e.g. 200)")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing label file")

    args = parser.parse_args()
    shiftLabels(args.songName, args.shift, args.dir)
