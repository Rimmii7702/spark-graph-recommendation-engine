"""Download original SNAP Wikipedia voting network (not a generated substitute)."""
from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

URL = "https://snap.stanford.edu/data/wiki-Vote.txt.gz"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/wiki-Vote.txt.gz"))
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        print(f"Already exists: {a.output}; delete it to download again")
        return
    partial = a.output.with_name(a.output.name + ".partial")
    try:
        urllib.request.urlretrieve(URL, partial)
        if partial.stat().st_size < 1000:
            raise ValueError("Unexpectedly small download")
        partial.replace(a.output)
        print(f"Downloaded {a.output}: {a.output.stat().st_size:,} bytes")
    finally:
        partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
