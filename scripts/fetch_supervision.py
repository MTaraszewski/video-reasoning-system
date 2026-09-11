"""Fetch Roboflow's `supervision` sample videos.

LICENCE: unstated. The supervision docs page carries only "Roboflow 2023. All
rights reserved." So these are FETCHED AT RUN TIME AND NEVER REDISTRIBUTED —
nothing downloaded here is committed to the repo. Our labels over them are our
own work and ship normally. See DATASETS.md §6.

    python scripts/fetch_supervision.py --out data/supervision
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Why these three. MILK_BOTTLING_PLANT is the only source in the whole eval set
# covering absence-of-motion events ("the machine stops moving"); SUBWAY and
# MARKET_SQUARE add crowded fixed-camera scenes, which stress false positives.
DEFAULT = ["MILK_BOTTLING_PLANT", "SUBWAY", "MARKET_SQUARE"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/supervision")
    ap.add_argument("--assets", nargs="*", default=DEFAULT)
    args = ap.parse_args()

    try:
        from supervision.assets import VideoAssets, download_assets
    except ImportError:
        print("supervision not installed. It is an optional extra:", file=sys.stderr)
        print("  uv sync --extra data", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    cwd = os.getcwd()
    os.chdir(out)
    try:
        for name in args.assets:
            asset = getattr(VideoAssets, name, None)
            if asset is None:
                available = [a.name for a in VideoAssets]
                print(f"unknown asset {name!r}. Available: {available}", file=sys.stderr)
                return 1
            path = download_assets(asset)
            print(f"  {path}")
    finally:
        os.chdir(cwd)

    print("\nLicence is UNSTATED for these files. Do not commit or redistribute them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
