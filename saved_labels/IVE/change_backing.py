import argparse
import json
import shutil
from pathlib import Path
from typing import List, Any


def remove_backing_labels(song_name: str, member: str, file_path: str) -> tuple[Path, int]:
    """
    Remove all labels matching:
      [member, start, end, True, False]

    Excludes labels where both booleans are True.

    Saves a backup copy as:
      {original_stem}_old.json

    Then overwrites the original JSON file with the filtered labels.

    Returns:
      (json_path, removed_count)
    """
    json_path = Path(file_path) if file_path else Path(f"{song_name}_labels.json")

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        labels: List[List[Any]] = json.load(f)

    filtered_labels = []
    removed_count = 0

    for entry in labels:
        if (
            isinstance(entry, list)
            and len(entry) >= 5
            and entry[0] == member
            and entry[3] is True
            and entry[4] is False
        ):
            removed_count += 1
            continue

        filtered_labels.append(entry)

    backup_path = json_path.with_name(f"{json_path.stem}_old.json")
    shutil.copy2(json_path, backup_path)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(filtered_labels, f, indent=4, ensure_ascii=False)

    return json_path, removed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove backing-vocal labels like [member, start, end, true, false] from a song label JSON."
    )
    parser.add_argument("song_name", help="Song name, e.g. ELEVEN")
    parser.add_argument("member", help="Member name, e.g. Yujin")
    parser.add_argument(
        "--file-path",
        dest="file_path",
        default=None,
        help="Optional explicit path to the JSON file. Defaults to <song_name>_labels.json in the current directory.",
    )

    args = parser.parse_args()

    json_path, removed_count = remove_backing_labels(
        song_name=args.song_name,
        member=args.member,
        file_path=args.file_path,
    )

    print(f"Updated file: {json_path}")
    print(f"Backup created: {json_path.with_name(f'{json_path.stem}_old.json')}")
    print(f"Removed {removed_count} matching labels for member '{args.member}'.")


if __name__ == "__main__":
    main()
