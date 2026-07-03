"""
Pre-build a unified metadata cache for fast server startup.

Reads all per-keyframe JSON sidecars once and serialises them into
a single pickle file.  BM25 and the API metadata-attach step both
consume this cache instead of reading 41 000+ individual JSON files.

Usage:
    py -3.10 scripts/build_meta_cache.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

# Ensure src is on sys.path so `aic26` is importable
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aic26.config import settings


def build_cache(metadata_dir: Path | None = None, output: Path | None = None) -> Path:
    meta_dir = Path(metadata_dir or settings.metadata_dir)
    out_path = Path(output or meta_dir / "_cache.pkl")

    print(f"Scanning {meta_dir} ...")
    t0 = time.perf_counter()

    cache: dict[str, dict] = {}
    json_files = sorted(meta_dir.rglob("*.json"))

    for jf in json_files:
        key = f"{jf.parent.name}/{jf.stem}"       # e.g. K11_V001/003
        try:
            cache[key] = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            pass

    elapsed = time.perf_counter() - t0
    print(f"  Loaded {len(cache)} entries in {elapsed:.1f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Cache saved to {out_path} ({size_mb:.1f} MB)")
    return out_path


if __name__ == "__main__":
    build_cache()
