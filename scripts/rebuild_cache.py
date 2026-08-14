"""Rebuild _cache.pkl from updated JSON sidecars."""
import json, pickle
from pathlib import Path

metadata_dir = Path(r"D:\My\University\AIC2026\AIC-26\data\metadata")
cache = {}
for jf in sorted(metadata_dir.rglob("*.json")):
    key = f"{jf.parent.name}/{jf.stem}"
    try:
        cache[key] = json.loads(jf.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Skip {jf}: {e}")

cache_path = metadata_dir / "_cache.pkl"
with open(cache_path, "wb") as f:
    pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

with_ocr = sum(1 for v in cache.values() if v.get("ocr_text"))
size_mb = cache_path.stat().st_size / 1024 / 1024
print(f"Rebuilt _cache.pkl: {len(cache)} entries, {with_ocr} with ocr_text, {size_mb:.1f} MB")
