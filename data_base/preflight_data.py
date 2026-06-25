from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"

sys.path.insert(0, str(PROJECT_ROOT))
from configs.models import MODELS  # noqa: E402


def required_files(stage: str) -> list[str]:
    files: list[str] = []
    if stage == "populate":
        files += [spec.pt_file for spec in MODELS.values()]
        files.append(os.getenv("METADATA_CSV_NAME", "vox1_meta.csv"))
    elif stage == "evaluate":
        files += [spec.test_pt_file for spec in MODELS.values()]
    else:
        raise ValueError(f"Unknown stage: {stage}")
    # de-dup while preserving order
    return list(dict.fromkeys(files))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify required data files exist before populate/evaluate."
    )
    parser.add_argument("--stage", required=True, choices=["populate", "evaluate"])
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")

    missing = [
        name for name in required_files(args.stage) if not (DATA_DIR / name).is_file()
    ]
    if missing:
        print(
            f"[preflight] Missing required data files for stage '{args.stage}' in {DATA_DIR}:",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print(
            "\nDownload the data bundle (see the 'Reproduce with Docker' section in README.md)\n"
            "and unzip it so the files above land in data_base/data/.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[preflight] All required '{args.stage}' data files present in {DATA_DIR}.")


if __name__ == "__main__":
    main()
