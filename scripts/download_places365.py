#!/usr/bin/env python3
"""
Download Places365 model weights + labels for the per-keyframe scene
classifier.

References
----------
- Places365 dataset:  http://places.csail.mit.edu/
- Model repo:         https://github.com/CSAILVision/places365
- Default destination: ./models/places365/

Per the spec's Open Question 4 the weights are downloaded once at
setup time, then `Places365Classifier(labels_path=...)` loads them
from disk on first inference.

About the default model
-----------------------
The lifeXplore paper (lifeXplore 2024 §3 Table 1) references
"EfficientNet-B2 + Places365", but the official CSAILPlaces365
release ships only AlexNet / ResNet{18,50,152} / DenseNet161 /
GoogLeNet / VGG16 checkpoints. We default to **ResNet50** as the
closest canonical option that ships with the upstream repo; to use
a different backbone (e.g. an EfficientNet-B2 fine-tune hosted on
HuggingFace), pass `--weights-url <url>`.

Usage
-----
    # Default: ResNet50 weights from places2.csail.mit.edu
    python scripts/download_places365.py

    # Custom weight (e.g. an EfficientNet-B2 HuggingFace upload)
    python scripts/download_places365.py --weights-url \\
        https://huggingface.co/<user>/<repo>/resolve/main/weights.pth

    # Override output dir / base URL
    python scripts/download_places365.py --output /tmp/p365 \\
        --url-prefix https://example.com/

    # Just labels
    python scripts/download_places365.py --weights-only \\
        --labels-only  # both false
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

# Canonical CSAIL Places365 host. ResNet50 is the lightest backbone with
# a publicly hosted checkpoint (~98 MB).
DEFAULT_WEIGHTS_URL: str = (
    "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
)
DEFAULT_LABELS_URL: str = (
    "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
)
WEIGHTS_FILENAME: str = "resnet50_places365.pth.tar"
LABELS_FILENAME: str = "categories_places365.txt"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, chunk_bytes: int = 1 << 20) -> None:
    """Stream-download `url` → `dest`, raising a clear error on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {url}")
    print(f"    → {dest}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            with dest.open("wb") as out:
                downloaded = 0
                while True:
                    chunk = resp.read(chunk_bytes)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100.0 * downloaded / total
                        print(
                            f"    {downloaded / 1e6:6.1f} MB / {total / 1e6:6.1f} MB ({pct:5.1f}%)",
                            end="\r",
                        )
            if total:
                print()  # newline after progress
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output",
        "-o",
        default="models/places365",
        help="Destination directory (default: ./models/places365)",
    )
    p.add_argument(
        "--weights-url",
        default=DEFAULT_WEIGHTS_URL,
        help=f"URL of the model weights (default: {DEFAULT_WEIGHTS_URL})",
    )
    p.add_argument(
        "--labels-url",
        default=DEFAULT_LABELS_URL,
        help=f"URL of the labels file (default: {DEFAULT_LABELS_URL})",
    )
    p.add_argument(
        "--weights-only",
        action="store_true",
        help="Only download the model weights file",
    )
    p.add_argument(
        "--labels-only",
        action="store_true",
        help="Only download the labels file",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists locally",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    do_weights = not args.labels_only
    do_labels = not args.weights_only

    if do_labels:
        labels_path = out_dir / LABELS_FILENAME
        if labels_path.exists() and not args.force:
            print(f"✓ Labels already present: {labels_path}")
        else:
            try:
                _download(args.labels_url, labels_path)
            except Exception as exc:
                print(f"❌ {exc}", file=sys.stderr)
                return 1
            print(
                f"✓ Labels downloaded ({labels_path.stat().st_size:,} bytes, "
                f"sha256={_sha256(labels_path)[:12]}…)"
            )

    if do_weights:
        weights_path = out_dir / WEIGHTS_FILENAME
        if weights_path.exists() and not args.force:
            print(f"✓ Weights already present: {weights_path}")
        else:
            print("⏳ Downloading Places365 weights (may take a minute)…")
            try:
                _download(args.weights_url, weights_path)
            except Exception as exc:
                print(f"❌ {exc}", file=sys.stderr)
                # Clean up partial download
                if weights_path.exists():
                    weights_path.unlink()
                return 1
            size_mb = weights_path.stat().st_size / 1e6
            print(f"✓ Weights downloaded ({size_mb:.1f} MB, sha256={_sha256(weights_path)[:12]}…)")

    print()
    print("=" * 60)
    print("Places365 setup complete")
    print("=" * 60)
    print(f"  Directory: {out_dir}")
    if do_labels:
        print(f"  Labels:    {(out_dir / LABELS_FILENAME).exists()}")
    if do_weights:
        print(f"  Weights:   {(out_dir / WEIGHTS_FILENAME).exists()}")
    print()
    print("Next: in .env, set  PLACES365_WEIGHTS_PATH=<that directory>")
    print("      so Places365Classifier picks up the local cache.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
