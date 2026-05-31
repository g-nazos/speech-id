"""Carve a held-out test (probe) set out of the VoxCeleb dev set.

Speaker identification needs two disjoint sets:
  * enrollment / gallery  -> reference embeddings stored in the database
  * test / probe          -> utterances the system has NEVER enrolled

This script materialises that split on disk instead of inside the extractor.
For every speaker that has at least two videos, the *last video* (by sorted
folder name) is MOVED in its entirety from the dev directory into the test
directory, preserving the `<speaker_id>/<video_id>/*.wav` nesting so that:

  * the extractor's speaker-id parsing keeps working, and
  * `vox1_meta.csv` metadata / database lookups still line up
    (every test speaker keeps >=1 video in dev, so it remains enrolled).

Notes / guarantees:
  * Selection is fully deterministic (sorted()[-1]), no randomness.
  * Folders are MOVED, not copied, so no utterance can leak into both sets.
  * Speakers with a single video are left untouched (no fallback).
  * Re-running is safe: a video already moved is skipped.
  * Use --dry-run first to preview what would move.
"""

import os
import shutil
import argparse
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _subdirs(path):
    """Return sorted names of immediate sub-directories of `path`."""
    return sorted(
        name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
    )


def make_split(source_dir, dest_dir, dry_run=False):
    if not os.path.isdir(source_dir):
        logger.error("Source directory does not exist: %s", source_dir)
        return

    speakers = _subdirs(source_dir)
    logger.info("Found %d speakers in %s", len(speakers), source_dir)

    moved_videos = 0
    moved_speakers = 0
    skipped_single_video = 0
    already_present = 0

    for speaker_id in speakers:
        speaker_src = os.path.join(source_dir, speaker_id)
        videos = _subdirs(speaker_src)

        # No fallback: speakers with a single video stay fully in the dev set.
        if len(videos) < 2:
            skipped_single_video += 1
            continue

        held_out_video = videos[-1]  # deterministic: last by sorted name
        src_video_path = os.path.join(speaker_src, held_out_video)
        dest_speaker_dir = os.path.join(dest_dir, speaker_id)
        dest_video_path = os.path.join(dest_speaker_dir, held_out_video)

        if os.path.exists(dest_video_path):
            already_present += 1
            continue

        if dry_run:
            logger.info(
                "[dry-run] would move %s -> %s", src_video_path, dest_video_path
            )
        else:
            os.makedirs(dest_speaker_dir, exist_ok=True)
            shutil.move(src_video_path, dest_video_path)

        moved_videos += 1
        moved_speakers += 1

    verb = "Would move" if dry_run else "Moved"
    logger.info(
        "%s %d videos (one per speaker) for %d speakers into %s",
        verb,
        moved_videos,
        moved_speakers,
        dest_dir,
    )
    logger.info(
        "Left untouched: %d single-video speakers, %d videos already in test set.",
        skipped_single_video,
        already_present,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move one held-out video per speaker from the dev set into a test set."
    )
    parser.add_argument(
        "--source",
        default=os.path.join(project_root, "data", "vox1_dev_wav", "wav"),
        help="Source (enrollment/dev) directory containing <speaker_id>/<video_id>/ folders.",
    )
    parser.add_argument(
        "--dest",
        default=os.path.join(project_root, "data", "vox1_test_wav", "wav"),
        help="Destination (test/probe) directory; mirrors the source structure.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the moves without touching any files.",
    )
    args = parser.parse_args()

    make_split(args.source, args.dest, dry_run=args.dry_run)
