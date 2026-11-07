"""Fetch the review app's weights from the repository's weights release and check them.

    python packaging/weights.py build/desktop/weights                 # download (public release) and verify
    python packaging/weights.py build/desktop/weights --from DIR      # take the files from DIR (e.g. gh release download)

The files and their SHA-256 are listed in ``packaging/weights.json``; the target folder gets the ``weights/``
layout the registry expects (``positioning/...``, ``segmentation/...``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).with_name("weights.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="weights folder to fill")
    parser.add_argument("--from", dest="source", type=Path, help="folder that already holds the release assets")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base = f"https://github.com/{manifest['repository']}/releases/download/{manifest['release']}"
    problems = []
    for item in manifest["files"]:
        out = args.target / item["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        if not (out.is_file() and sha256(out) == item["sha256"]):
            if args.source:
                shutil.copyfile(args.source / item["asset"], out)
            else:
                print(f"downloading {item['asset']} ...", flush=True)
                with urllib.request.urlopen(f"{base}/{item['asset']}", timeout=120) as r, out.open("wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
        if sha256(out) != item["sha256"]:
            problems.append(f"{item['path']}: SHA-256 does not match the manifest")
        else:
            print(f"ok  {item['path']}")
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
