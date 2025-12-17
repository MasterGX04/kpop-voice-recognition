#!/usr/bin/env python3
"""
Check *_frame_labels.json files for invalid lead rows with more than one '1'.

Usage:
  python check_lead_uniqueness.py IVE
  python check_lead_uniqueness.py IVE --max-examples 20

Looks in:
  ./training_data/{group}/*_frame_labels.json

Outputs:
  - Songs with any lead rows where sum(row) > 1
  - Count of offending rows
  - A few example chunk indices and the row contents
"""

import argparse
import glob
import json
import os
from typing import List, Tuple


def find_offending_rows(lead: List[List[int]]) -> List[Tuple[int, int]]:
    """Returns list of (row_index, row_sum) where row_sum > 1."""
    bad = []
    for i, row in enumerate(lead):
        s = 0
        for v in row:
            try:
                s += int(round(float(v)))
            except Exception:
                s = 999
                break
        if s > 1:
            bad.append((i, s))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group", help="Group folder name under ./training_data (e.g., IVE)")
    ap.add_argument("--max-examples", type=int, default=10,
                    help="How many offending rows to print per song (default: 10)")
    args = ap.parse_args()

    base_dir = os.path.join(".", "training_data", args.group)
    pattern = os.path.join(base_dir, "*_frame_labels.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[check] No files found matching: {pattern}")
        return

    total_files = 0
    total_bad_rows = 0
    songs_with_issues = 0

    print(f"[check] Scanning {len(files)} file(s) in {base_dir}")

    for path in files:
        total_files += 1
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"\n[ERROR] Failed to read JSON: {path}\n  {e}")
            continue

        song = meta.get("song", os.path.basename(path))
        lead = meta.get("lead", None)

        if lead is None:
            print(f"\n[WARN] Missing 'lead' key in {path} (song={song})")
            continue
        if not isinstance(lead, list) or (lead and not isinstance(lead[0], list)):
            print(f"\n[WARN] 'lead' is not a 2D list in {path} (song={song})")
            continue

        bad = find_offending_rows(lead)
        if bad:
            songs_with_issues += 1
            total_bad_rows += len(bad)

            members = meta.get("members", [])
            print(f"\n❌ {song}  ({os.path.basename(path)})")
            print(f"   Bad lead rows (sum>1): {len(bad)} / {len(lead)}")

            max_ex = max(0, args.max_examples)
            for (i, s) in bad[:max_ex]:
                row = lead[i]
                ones = [j for j, v in enumerate(row) if int(round(float(v))) == 1]
                if members and all(j < len(members) for j in ones):
                    who = ", ".join(members[j] for j in ones)
                else:
                    who = ", ".join(map(str, ones))
                print(f"   - chunk {i:>6} | sum={s} | ones={ones} | {who}")
                print(f"             row={row}")

            if len(bad) > max_ex:
                print(f"   ... {len(bad) - max_ex} more offending rows not shown (use --max-examples)")

    print("\n[summary]")
    print(f"  files_scanned:      {total_files}")
    print(f"  songs_with_issues:  {songs_with_issues}")
    print(f"  total_bad_rows:     {total_bad_rows}")
    if songs_with_issues == 0:
        print("  ✅ No lead rows with more than one '1' found.")


if __name__ == "__main__":
    main()
